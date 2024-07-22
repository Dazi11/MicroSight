# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from timm.layers import to_2tuple
from torch import nn, Tensor
from util.misc import NestedTensor
import torch.nn.init as init


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.similarity = PartAttention(d_model)
        self.Conv1 = nn.Conv2d(num_encoder_layers + 1, 8, 1,1)
        self.Conv2 = nn.Conv2d(8, 1, 1,1)
        self.MGCNNet = MGCNNet(input_dim=d_model, output_dim=d_model)
        self.ConvCombined = nn.Conv1d(d_model * 2, d_model, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, mask_position):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        # src -> W*H B D (625 2 256)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # pos_embed -> W*H B D (625 2 256)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed -> 100 B D (100 2 256)
        mask = mask.flatten(1)
        # mask -> B W*H (2 625)

        tgt = torch.zeros_like(query_embed)
        # tgt -> 100 B D (100 2 256)
        encoderRes = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory = encoderRes[0]
        attn_contri = encoderRes[1]
        # memory -> W*H B D (625 2 256)

        # MGCNet
        MGC_Input = src.transpose(0, 1)
        similarityMatrix = self.similarity(MGC_Input).unsqueeze(1)
        # adjacency -> B depth+1 Nodes Nodes
        adjacency = torch.cat([similarityMatrix, attn_contri], dim=1)
        adjacency = self.Conv2(self.Conv1(adjacency)).squeeze(1)
        partOutput = self.MGCNNet(adjacency=adjacency, x=MGC_Input, masked_pos=mask_position)
        # partOutput -> B W*H D (2 625 256)
        memory = self.ConvCombined(torch.cat([memory.permute(1, 2, 0), partOutput.permute(0, 2, 1)], dim=1))\
            .permute(2,0,1)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # hs -> layers B 100 D (6 2 100 256)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class LSTM(nn.Module):
    def __init__(self, embedding_dim=256, hidden_size=256, num_layers=2, bidirectional=True):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

    def forward(self, x):
        # x-> Nodes batch D (650 2 256)
        batch_size = x.shape[1]
        NodesNum = x.shape[0]

        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        # 初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        # 维度[layers, batch, hidden_len]
        if self.bidirectional:
            h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
            c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        # x = self.embedding(x)
        # Bi-lstm:-> [num_patches + 1, B * num_heads, (num_patches + 1)*2]
        output, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        # print(output.shape)
        output = output.reshape(NodesNum, batch_size, 2, self.hidden_size)
        forward_output = output[:, :, 0, :]
        backward_output = output[:, :, 1, :]
        output = (forward_output + backward_output) / 2
        # output:-> [num_patches, B * num_heads, num_patches]
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        attn_contri = []
        for layer in self.layers:
            res = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)
            output = res[0]
            attn_contri.append(res[1].unsqueeze(1))

        if self.norm is not None:
            output = self.norm(output)
        attn_contri = torch.cat(attn_contri, dim=1)
        return output, attn_contri


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.lstm = LSTM(d_model, d_model, 2, True)
        self.lstmNorm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # q,k,v -> W*H B D (650 2 256)
        q = k = self.with_pos_embed(src, pos)
        k = self.lstmNorm(self.lstm(k))
        # attn_contri -> B Nodes Nodes (2 650 650)
        attn_contri = (q.permute(1, 0, 2)) @ (k.permute(1, 2, 0))
        # attn_contri = torch.mean(l_attn, 1)
        # src_mask = None
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # src -> W*H B D (650 2 256)
        return src, attn_contri

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        k = self.lstmNorm(self.lstm(k))
        attn_contri = (q.permute(1, 0, 2)) @ (k.permute(1, 2, 0))
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_contri

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MaskedPosition(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self):
        super().__init__()
        self.masked_pos = nn.Parameter(torch.Tensor(2500, 2500))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.masked_pos)

    def forward(self, x):
        h, w = x.shape[-2:]
        pos = self.masked_pos[:h * w, :h * w]
        return pos


class PartAttention(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.dim = embed_dim
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.layerNorm1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x -> [Batch,patches,dim]
        patches = x.shape[1]
        B = x.shape[0]
        similarityMatrix = self.CosineDistance(x, B, patches, self.dim)
        # x -> [Batch, patches + 1(197), dim]
        # similarityMatrix -> [Batch,patches,patches]
        return similarityMatrix

    def EuclideanDistance(self, x, Batch, patches, dim):  # 欧几里得距离（自己与自己的距离为0，可在后续使用均值计算）
        x1 = x.reshape(Batch, patches, 1, dim)
        x2 = x.reshape(Batch, 1, patches, dim)
        y = torch.sum((x1 - x2) * (x1 - x2), dim=3)
        y = torch.div(1.0, y)
        return y

    def CosineDistance(self, x, Batch, patches, dim):  # 余弦相似度（分母为0的情况用浮点数表示）
        x1 = x.reshape(Batch, patches, 1, dim)
        x2 = x.reshape(Batch, 1, patches, dim)
        # x_up -> [B,patches,patches]
        x_up = torch.sum(x1 * x2, dim=3)
        # x1_norm -> [B,patches,1]
        # x2_norm -> [B,1,patches]
        x_down = x1.norm(p=2, dim=3) * x2.norm(p=2, dim=3)
        y = torch.div(x_up, x_down+ 1e-8)
        return y

    def CanberraDistance(self, x, Batch, patches, dim):  # 堪培拉距离
        x1_re = x.reshape(Batch, patches, 1, dim)
        x2_re = x.reshape(Batch, 1, patches, dim)
        # x_up ->[B,patches,patches,dim]
        x_up = torch.abs(x1_re - x2_re)
        x_down = torch.abs(x1_re) + torch.abs(x2_re)
        y = torch.sum(torch.div(x_up, x_down), dim=3)
        y = torch.div(1.0, y)
        return y

    def PearsonCorrelation(self, x, Batch, patches, dim):  # 皮尔逊距离（线性相关度）
        x_mean = torch.mean(x, dim=2).unsqueeze(2)
        x_pre = (x.permute(0, 2, 1) - x_mean.permute(0, 2, 1)).permute(0, 2, 1)
        x1 = x_pre.reshape(Batch, patches, 1, dim)
        x2 = x_pre.reshape(Batch, 1, patches, dim)
        # x_up -> [B,patches,patches]
        x_up = torch.sum(x1 * x2, dim=3)
        # x1_norm -> [B,patches,1]
        # x2_norm -> [B,1,patches]
        x_down = x1.norm(p=2, dim=3) * x2.norm(p=2, dim=3)
        y = torch.div(x_up, x_down + 1e-8)
        return y

    def Bray_CurtisBistance(self, x, Batch, patches, dim):  # 布雷柯蒂斯距离（将生态学与环境科学中的距离定义引入，也可以用来计算样本之间的差异性）
        x1_re = x.reshape(Batch, patches, 1, dim)
        x2_re = x.reshape(Batch, 1, patches, dim)
        x_up = torch.sum(torch.abs(x1_re - x2_re), dim=3)
        x_down = torch.sum(torch.abs(x1_re + x2_re), dim=3)
        y = torch.div(x_down, x_up)  # 分母与分子互换，值越大，越相关
        return y


class GraphConvolution(nn.Module):  # 图卷积块
    def __init__(self, input_dim=768, output_dim=768, use_bias=True):
        """图卷积：H*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # self.w = nn.Parameter(torch.Tensor([0.8, 0.2]))  # 自适应学习权重，初始化为1
        self.w = 0.6
        self.act = nn.Sigmoid()
        if self.use_bias:  # 添加偏置
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.kaiming_uniform_神经网络权重初始化，神经网络要优化一个非常复杂的非线性模型，而且基本没有全局最优解，
        # 初始化在其中扮演着非常重要的作用，尤其在没有BN等技术的早期，它直接影响模型能否收敛。

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature, masked_pos):
        """
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        # 设置为超参数
        # w1 = torch.exp(self.w[0] / torch.sum(torch.exp(self.w)))
        # w2 = torch.exp(self.w[1] / torch.sum(torch.exp(self.w)))
        # adjacency[np.diag_indices_from((adjacency))] += 1  # 进行拉普拉斯平滑，对角线自连
        # 控制位置掩码的权重在0到1之间
        self.w = 0.6
        adjacency = self.w * adjacency + (1 - self.w) * (self.act(masked_pos) * adjacency)
        # 图卷积运算
        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class MGCNNet(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.GCNconv1 = GraphConvolution(input_dim=input_dim, output_dim=32)
        self.GCNconv2 = GraphConvolution(input_dim=32, output_dim=output_dim)

        self.layerNorm1 = nn.LayerNorm(input_dim)
        self.layerNorm2 = nn.LayerNorm(32)
        self.layerNorm3 = nn.LayerNorm(output_dim)

    def forward(self, adjacency, x, masked_pos):
        x = self.layerNorm1(x)
        x = F.relu(self.layerNorm2(self.GCNconv1(adjacency=adjacency, input_feature=x, masked_pos=masked_pos)))
        x = F.relu(self.layerNorm3(self.GCNconv2(adjacency=adjacency, input_feature=x, masked_pos=masked_pos)))
        return x
