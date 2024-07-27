import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from mmdet.models.utils.transformer import Transformer, DeformableDetrTransformer, DeformableDetrTransformerDecoder
from mmdet.models.utils.builder import TRANSFORMER
import torch.nn.init as init


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CoDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, look_forward_twice=False, **kwargs):

        super(CoDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                                         torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                                                    ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class CoDeformableDetrTransformer(DeformableDetrTransformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 mixed_selection=True,
                 with_pos_coord=True,
                 with_coord_feat=True,
                 num_co_heads=1,
                 **kwargs):
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(CoDeformableDetrTransformer, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                # bug: this code should be 'self.head_pos_embed = nn.Embedding(self.num_co_heads, self.embed_dims)', we keep this bug for reproducing our results with ResNet-50.
                # You can fix this bug when reproducing results with swin transformer.
                self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1, self.embed_dims)
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims * 2))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims * 2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                return_encoder_output=False,
                attn_masks=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk = query_embed.shape[0]
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            if return_encoder_output:
                return inter_states, init_reference_out, \
                       inter_references_out, enc_outputs_class, \
                       enc_outputs_coord_unact, memory
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_coord_unact
        if return_encoder_output:
            return inter_states, init_reference_out, \
                   inter_references_out, None, None, memory
        return inter_states, init_reference_out, \
               inter_references_out, None, None

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, \
               inter_references_out


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    # TODO: It can be implemented by add an out_channel arg of
    #  mmcv.cnn.bricks.transformer.FFN
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            pos_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class CoDinoTransformer(CoDeformableDetrTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDinoTransformer, self).__init__(*args, **kwargs)
        self.lstm = LSTM(self.embed_dims, self.embed_dims, 2, True)
        # self.similarity = PartAttention(self.embed_dims)
        self.Conv1 = nn.Conv2d(2, 8, 1, 1)
        self.Conv2 = nn.Conv2d(8, 1, 1, 1)
        self.ConvCombined = nn.Conv1d(self.embed_dims * 2, self.embed_dims, 1)
        self.MGCNNet = MGCNNet(input_dim=self.embed_dims, output_dim=self.embed_dims)
        self.up1 = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 4, 2, 1)
        self.up4 = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 4, 2, 1)
        self.act = nn.GELU()
        self.masked_pos = nn.Parameter(torch.Tensor(2500, 2500))
        self.lstmNorm = nn.LayerNorm(self.embed_dims)
        self.layerNormS = nn.LayerNorm(self.embed_dims)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.masked_pos)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)

    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.query_embed.weight.data)

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
        y = torch.div(x_up, x_down)
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
        del x_mean
        del x_pre
        del x1
        del x2
        del x_up
        del x_down
        return y

    def Bray_CurtisBistance(self, x, Batch, patches, dim):  # 布雷柯蒂斯距离（将生态学与环境科学中的距离定义引入，也可以用来计算样本之间的差异性）
        x1_re = x.reshape(Batch, patches, 1, dim)
        x2_re = x.reshape(Batch, 1, patches, dim)
        x_up = torch.sum(torch.abs(x1_re - x2_re), dim=3)
        x_down = torch.sum(torch.abs(x1_re + x2_re), dim=3)
        y = torch.div(x_down, x_up)  # 分母与分子互换，值越大，越相关
        return y

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        first_size = 0
        first_h = 0
        first_w = 0
        second_h = 0
        second_w = 0
        first_bs = 0
        first_c = 0
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            if lvl == len(mlvl_feats) - 1:
                first_size = feat.shape[1]
                first_h = h
                first_w = w
                first_bs = bs
                first_c = c
            if lvl == len(mlvl_feats) - 2:
                second_w = w
                second_h = h
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        # print(feat_flatten.shape)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)

        all_size = feat_flatten.shape[0]
        MGC_Input = feat_flatten.transpose(0, 1)[:, all_size - first_size:, :]

        lstm_feature = feat_flatten[all_size - first_size:, :, :]
        lstm_feature = self.lstmNorm(self.lstm(lstm_feature))
        pre_feature = feat_flatten
        feat_flatten = torch.cat([lstm_feature, pre_feature[: all_size - first_size, :, :]], dim=0)
        # q = k = feat_flatten
        # attn_contri -> B Nodes Nodes (2 650 650)
        # attn_contri = (q.permute(1, 0, 2)) @ (k.permute(1, 2, 0))
        new_feat = lstm_feature
        q = new_feat.permute(1, 0, 2)
        k = new_feat.permute(1, 2, 0)
        attn_contri = torch.matmul(q, k).unsqueeze(1)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        # memory w*h b c
        patches = MGC_Input.shape[1]
        B = MGC_Input.shape[0]
        x = self.layerNormS(MGC_Input)
        similarityMatrix = self.CosineDistance(x, B, patches, self.embed_dims).unsqueeze(1)
        # similarityMatrix = self.similarity(MGC_Input).unsqueeze(1)
        # adjacency -> B depth+1 Nodes Nodes
        adjacency = torch.cat([similarityMatrix, attn_contri], dim=1)
        # adjacency = similarityMatrix.squeeze(1)
        adjacency = self.Conv2(self.Conv1(adjacency)).squeeze(1)
        partOutput = self.MGCNNet(adjacency=adjacency, x=MGC_Input,
                                  masked_pos=self.masked_pos[:first_size, :first_size])
        part_flatten = []
        # partOutput -> B W*H D (2 625 256)
        partOutput = partOutput.reshape(first_bs, first_h, first_w, first_c).permute(0, 3, 1, 2)
        x1 = self.act(self.up1(partOutput))[:, :, :second_h, :second_w]
        x2 = self.act(self.up2(x1))
        x3 = self.act(self.up3(x2))
        x4 = self.act(self.up4(x3))
        part_flatten.append(x4.flatten(2).transpose(1, 2))
        part_flatten.append(x3.flatten(2).transpose(1, 2))
        part_flatten.append(x2.flatten(2).transpose(1, 2))
        part_flatten.append(x1.flatten(2).transpose(1, 2))
        part_flatten.append(partOutput.flatten(2).transpose(1, 2))
        part_flatten = torch.cat(part_flatten, 1)
        # part_flatten  B H*W C
        memory = self.ConvCombined(torch.cat([memory.permute(1, 2, 0), part_flatten.permute(0, 2, 1)], dim=1)) \
            .permute(2, 0, 1)
        # memory w*h b c
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out, topk_score, topk_anchor, memory

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        memory = feat_flatten
        # enc_inter = [feat.permute(1, 2, 0) for feat in enc_inter]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = (pos_anchors)
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out


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


# class PartAttention(nn.Module):
#     def __init__(self, embed_dim=768):
#         super().__init__()
#         self.dim = embed_dim
#         # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         # self.pos_drop = nn.Dropout(p=drop_rate)

#
#     def forward(self, x):
#         # x -> [Batch,patches,dim]
#
#         # x -> [Batch, patches + 1(197), dim]
#         # similarityMatrix -> [Batch,patches,patches]
#         return similarityMatrix


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
        self.w = 1
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
