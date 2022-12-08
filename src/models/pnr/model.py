import math
from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.nn import Embedding
from transformers import BertPreTrainedModel, BertConfig

from .fpn import PyramidFeatureNet
from .dec import TransformerDecoder, TransformerDecoderLayer
from .predictor import MaskAwareClassifier, MaskAwareSSNFuse, my_cosine_similarity


class PnRNet(BertPreTrainedModel):

    def __init__(
            self,
            config: BertConfig,
            n_classes: int,
            n_queries: int,

            dropout=0.1, pool_type="max",
            use_lstm=False, lstm_layers=3, lstm_drop=0.1,
            token_ebd: Optional[Embedding] = None,
            pos_ebd: Optional[Embedding] = None,
            char_ebd: Optional[Embedding] = None, char_lstm_layers=1, char_lstm_drop=0.1,

            fpn_type: str = "uni", fpn_layers: int = 8, fpn_drop=0.1,
            use_topk_query=True, use_msf=True,
            dec_layers=3, dec_intermediate_size=1024, dec_num_attention_heads=8,
    ):
        super(PnRNet, self).__init__(config)
        self.config = config
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.dropout = nn.Dropout(dropout)

        self.fpn_layers = fpn_layers
        if fpn_type == "uni":
            fpn_cls = PyramidFeatureNet
        else:
            raise NotImplementedError(fpn_type)
        self.fpn = fpn_cls(
            config=config,
            dropout=dropout,
            pool_type=pool_type,

            use_lstm=use_lstm, lstm_layers=lstm_layers, lstm_drop=lstm_drop,
            token_ebd=token_ebd,
            pos_ebd=pos_ebd,
            char_ebd=char_ebd, char_lstm_layers=char_lstm_layers, char_lstm_drop=char_lstm_drop,

            fpn_layers=fpn_layers, fpn_drop=fpn_drop,
        )

        self.use_msf = use_msf
        self.use_topk_query = use_topk_query
        if not use_topk_query:
            # 如果不是two stage，那么类似DETR，使用可学习的query
            self.query_embed = Embedding(n_queries, config.hidden_size * 2)

        self.stage1_classifier = MaskAwareClassifier(config.hidden_size, n_classes)
        self.padding_span_feature = Embedding(1, self.fpn.output_feature_size)
        self.feat_trans = nn.Linear(self.fpn.output_feature_size, config.hidden_size * 2)
        self.pos_feat_norm = nn.LayerNorm(config.hidden_size)
        self.cls_feat_norm = nn.LayerNorm(config.hidden_size)

        self.decoder = TransformerDecoder(
            config,
            TransformerDecoderLayer(
                d_model=config.hidden_size,
                ca_kdim=self.fpn.output_feature_size,
                d_ffn=dec_intermediate_size,
                activation="relu",
                n_heads=dec_num_attention_heads,
            ),
            dec_layers
        )

        self.left_detector = MaskAwareSSNFuse(
            config.hidden_size,
            # 如果use_msf，那么用的是fpn的第一层，否则用的是htoken
            self.fpn.output_feature_size if self.use_msf else config.hidden_size
        )
        self.right_detector = MaskAwareSSNFuse(
            config.hidden_size,
            # 如果use_msf，那么用的是fpn的第一层，否则用的是htoken
            self.fpn.output_feature_size if self.use_msf else config.hidden_size
        )
        self.classifier = MaskAwareClassifier(config.hidden_size, n_classes)

    def init(self):
        for child in self.children():
            if isinstance(child, MaskAwareClassifier):
                prior_prob = 0.01
                child.classifier.bias.data = torch.ones_like(
                    child.classifier.bias.data
                ) * -math.log((1 - prior_prob) / prior_prob)
            else:
                self._init_weights(child)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gather_feat(self, feats: Tensor, indexes: Tensor) -> Tensor:
        # query_pos = torch.gather(
        #     position_feat,
        #     1,
        #     topk_indexes.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        # )
        # tgt = torch.gather(
        #     cls_feat,
        #     1,
        #     topk_indexes.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        # )
        # NOTE: torch.gather with input larger than 1 dim is not deterministic in pytorch 1.11,
        #   for use_deterministic_algorithms checking, we use indexing
        gathered_feats = []
        for s_i, ind in enumerate(indexes):
            gathered_feats.append(feats[s_i, ind, :])
        return torch.stack(gathered_feats)

    def stage1_forward(
        self,
        flat_cls_features: Tensor,
        flat_pos_features: Tensor,
        flat_padding_masks: Tensor,
    ):
        _, candidates, _ = flat_cls_features.size()

        # Classification
        flat_logits = self.stage1_classifier.forward(
            # feat, layer_feature[0], proposal_left, proposal_right, padding_mask, feat_mask
            flat_cls_features, flat_padding_masks
        )

        scores = torch.sum(torch.softmax(flat_logits, dim=-1)[..., 1:], dim=-1)
        # 要保证mask的排名靠后，并且mask部分的边界是不合法的边界，无法在hungarian匹配的时候产生匹配
        scores = torch.masked_fill(scores, flat_padding_masks, -1)

        topk_scores, topk_indexes = torch.topk(
            scores, min(self.n_queries, candidates),
            dim=1, largest=True, sorted=False
        )
        topk_padding_masks = topk_scores < 0
        query_pos = self.gather_feat(flat_pos_features, topk_indexes)
        query = self.gather_feat(flat_cls_features, topk_indexes)

        return flat_logits, (query_pos, query, topk_padding_masks, topk_indexes)

    def stage2_forward(
        self,
        query: Tensor, query_pos: Tensor,
        dec_key: Tensor, dec_key_mask: Tensor,
        sa_q_mask: Tensor,
        boundary_key: Tensor, padding_mask: Tensor, query_padding_masks: Tensor
    ):
        hs = self.decoder.forward(
            query, query_pos, dec_key, dec_key_mask, sa_q_mask=sa_q_mask,
        )

        # 处理输出
        stage2_output = []
        for hidden in hs:
            lay_left = self.left_detector.forward(hidden, boundary_key, padding_mask, query_padding_masks)
            lay_right = self.right_detector.forward(hidden, boundary_key, padding_mask, query_padding_masks)
            # p_cls = self.classifier.forward(hidden, h_token, p_left, p_right, padding_mask, tgt_masks)
            lay_cls_logits = self.classifier.forward(hidden, query_padding_masks)
            stage2_output.append({'hidden': hidden, 'entity_logits': lay_cls_logits, 'p_left': lay_left, 'p_right': lay_right})
        return stage2_output

    def forward(
        self,
        encodings: Tensor, encoding_masks: Tensor,
        token2encoding_masks: Tensor, token2start: Tensor, token_masks: Tensor,
        pos_encoding: Optional[Tensor] = None,
        w2v_encoding: Optional[Tensor] = None,
        char_encoding: Optional[Tensor] = None,
        token_masks_char: Optional[Tensor] = None,
        char_count: Optional[Tensor] = None,
    ):
        bsz, _ = encodings.size()
        padding_mask = ~token_masks

        fpn_outputs = self.fpn.forward(
            encodings=encodings, encoding_masks=encoding_masks,
            token2encoding_masks=token2encoding_masks,
            token2start=token2start,
            token_masks=token_masks, pos_encoding=pos_encoding,
            w2v_encoding=w2v_encoding,
            char_encoding=char_encoding,
            token_masks_char=token_masks_char,
            char_count=char_count,
        )
        h_token = fpn_outputs["encoder_outputs"]
        fpn_features_list = fpn_outputs["features_list"]
        fpn_padding_masks_list = fpn_outputs["padding_masks_list"]
        flat_features = torch.cat(fpn_features_list, dim=1)
        flat_padding_masks = torch.cat(fpn_padding_masks_list, dim=1)
        flat_features = torch.masked_scatter(
            flat_features, flat_padding_masks.unsqueeze(-1),
            source=self.padding_span_feature.weight.view(1, 1, -1).expand(bsz, flat_features.size(1), -1)
        )
        flat_pos_features, flat_cls_features = torch.split(
            self.feat_trans.forward(flat_features),
            self.config.hidden_size, dim=2
        )
        flat_pos_features = self.pos_feat_norm.forward(flat_pos_features)
        flat_cls_features = self.cls_feat_norm.forward(flat_cls_features)

        # Decoding
        flat_logits, (query_pos, query, topk_padding_masks, topk_indexes) = self.stage1_forward(
            flat_cls_features, flat_pos_features, flat_padding_masks
        )
        if self.use_msf:
            # 使用FPN特征
            dec_key = flat_features
            dec_key_mask = flat_padding_masks
            boundary_key = fpn_features_list[0]    # 用FPN的第一层作为预测边界的key
        else:
            # 使用token特征
            dec_key = h_token
            dec_key_mask = padding_mask
            boundary_key = h_token
        if self.use_topk_query:
            # 用FPN的topk来初始化
            query_padding_masks = topk_padding_masks
            sa_q_mask = query_padding_masks.unsqueeze(1).expand(-1, query_padding_masks.size(-1), -1)
        else:
            # 用query embedding来初始化
            query_pos, query = torch.split(self.query_embed.weight, self.config.hidden_size, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bsz, -1, -1)
            query = query.unsqueeze(0).expand(bsz, -1, -1)  # (bs, n_q, C)
            query_padding_masks = None
            sa_q_mask = None

        # 处理输出
        stage2_output = self.stage2_forward(
            query=query, query_pos=query_pos,
            dec_key=dec_key, dec_key_mask=dec_key_mask, sa_q_mask=sa_q_mask,
            boundary_key=boundary_key, padding_mask=padding_mask, query_padding_masks=query_padding_masks,
        )

        stage1_output = None
        if self.use_topk_query:
            stage1_output = {
                "layer_size": [feat.size() for feat in fpn_features_list],      # 虽然也能推出来但还是直接返回比较方便
                "flat_features": flat_features,
                "flat_cls_features": flat_cls_features,
                "flat_pos_features": flat_pos_features,
                "flat_logits": flat_logits,
                "flat_padding_masks": flat_padding_masks,
                "topk_index": topk_indexes,
            }

        return h_token, stage1_output, stage2_output


class FSPnRNetForPreTraining(BertPreTrainedModel):
    """
    Q: 为什么不用PnRNet直接进行预训练？
    A: 因为PnRNet在query进行边界预测和分类的时候并没有

    Args:
        BiPnRNet (_type_): _description_
    """
    def __init__(
        self,
        config: BertConfig,
        n_classes: int,
        n_queries: int,
        dropout=0.1,
        pool_type="max", use_lstm=False, lstm_layers=3, lstm_drop=0.1,
        token_ebd: Optional[Embedding] = None,
        pos_ebd: Optional[Embedding] = None,
        char_ebd: Optional[Embedding] = None, char_lstm_layers=1, char_lstm_drop=0.1,
        fpn_type: str = "uni", fpn_layers: int = 8, fpn_drop=0.1,
        use_topk_query=True, use_msf=True,
        dec_layers=3, dec_intermediate_size=1024, dec_num_attention_heads=8
    ):
        super().__init__(
            config,
            n_classes,
            n_queries,
            dropout,
            pool_type, use_lstm, lstm_layers, lstm_drop,
            token_ebd,
            pos_ebd,
            char_ebd, char_lstm_layers, char_lstm_drop,
            fpn_type, fpn_layers, fpn_drop,
            use_topk_query, use_msf,
            dec_layers, dec_intermediate_size, dec_num_attention_heads
        )


class DSPnRNet(PnRNet):
    def __init__(
        self,
        config: BertConfig,
        n_classes: int,
        n_queries: int,
        type_ebd: Embedding,
        dropout: float = 0.1,
        ebd_drop: float = 0.1,
        pool_type="max", use_lstm=False, lstm_layers=3, lstm_drop=0.1,
        token_ebd: Optional[Embedding] = None,
        pos_ebd: Optional[Embedding] = None,
        char_ebd: Optional[Embedding] = None, char_lstm_layers=1, char_lstm_drop=0.1,
        fpn_type: str = "uni", fpn_layers: int = 8, fpn_drop=0.1,
        dec_intermediate_size=1024, dec_num_attention_heads=8
    ):
        super().__init__(
            config,
            n_classes,
            n_queries,
            dropout,
            pool_type, use_lstm, lstm_layers, lstm_drop,
            token_ebd,
            pos_ebd,
            char_ebd, char_lstm_layers, char_lstm_drop,
            fpn_type, fpn_layers, fpn_drop,
            dec_intermediate_size, dec_num_attention_heads
        )
        # 不使用原来的classifier而是进行相似度比较分类
        del self.classifier
        del self.stage1_classifier
        self.class_embedding = Embedding(n_classes, config.hidden_size)
        self.query_cls_projector = nn.Linear(config.hidden_size, config.hidden_size)

    def stage1_forward(
        self,
        flat_cls_features: Tensor,
        flat_pos_features: Tensor,
        flat_padding_masks: Tensor,
    ):
        _, candidates, _ = flat_cls_features.size()

        # Classification
        flat_logits = my_cosine_similarity(flat_cls_features, self.class_embedding.weight)

        scores = torch.sum(torch.softmax(flat_logits, dim=-1)[..., 1:], dim=-1)
        # 要保证mask的排名靠后，并且mask部分的边界是不合法的边界，无法在hungarian匹配的时候产生匹配
        scores = torch.masked_fill(scores, flat_padding_masks, -1)

        topk_scores, topk_indexes = torch.topk(
            scores, min(self.n_queries, candidates),
            dim=1, largest=True, sorted=False
        )
        topk_padding_masks = topk_scores < 0
        query_pos = self.gather_feat(flat_pos_features, topk_indexes)
        query = self.gather_feat(flat_cls_features, topk_indexes)

        return flat_logits, (query_pos, query, topk_padding_masks, topk_indexes)

    def stage2_forward(
        self,
        query: Tensor, query_pos: Tensor,
        dec_key: Tensor, dec_key_mask: Tensor,
        sa_q_mask: Tensor,
        boundary_key: Tensor, padding_mask: Tensor, query_padding_masks: Tensor
    ):
        hs = self.decoder.forward(
            query, query_pos, dec_key, dec_key_mask, sa_q_mask=sa_q_mask,
        )

        # 处理输出
        stage2_output = []
        for hidden in hs:
            lay_left = self.left_detector.forward(hidden, boundary_key, padding_mask, query_padding_masks)
            lay_right = self.right_detector.forward(hidden, boundary_key, padding_mask, query_padding_masks)
            # p_cls = self.classifier.forward(hidden, h_token, p_left, p_right, padding_mask, tgt_masks)
            lay_cls_logits = my_cosine_similarity(
                self.query_cls_projector.forward(hidden),
                self.class_embedding.weight
            )
            lay_cls_logits.masked_fill_(query_padding_masks.unsqueeze(-1), 0)
            stage2_output.append({'hidden': hidden, 'entity_logits': lay_cls_logits, 'p_left': lay_left, 'p_right': lay_right})
        return stage2_output
