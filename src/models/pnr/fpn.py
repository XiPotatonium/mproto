from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Embedding
from transformers import BertConfig

from ..pyramid.init_util import init_linear
from ..trf_enc import TrfEncoder


class BasePyramidFeatureNet(nn.Module, ABC):
    def __init__(
            self, config: BertConfig,
            dropout=0.1, pool_type="max",
            use_lstm=False, lstm_layers=3, lstm_drop=0.1,
            token_ebd: Optional[Embedding] = None,
            pos_ebd: Optional[Embedding] = None,
            char_ebd: Optional[Embedding] = None,
            char_lstm_layers=1, char_lstm_drop=0.1,

            fpn_layers: int = 8,
    ):
        super(BasePyramidFeatureNet, self).__init__()
        self.encoder = TrfEncoder(
            config=config,
            ebd_drop=dropout,
            pool_type=pool_type,
            use_lstm=use_lstm, lstm_layers=lstm_layers, lstm_drop=lstm_drop,
            token_ebd=token_ebd,
            pos_ebd=pos_ebd,
            char_ebd=char_ebd, char_lstm_layers=char_lstm_layers, char_lstm_drop=char_lstm_drop,
        )
        self.config = config
        self.num_layer = fpn_layers

    @property
    @abstractmethod
    def output_feature_size(self) -> int:
        pass

    def max_positions(self):
        return self.encoder.max_positions()


class PyramidFeatureNet(BasePyramidFeatureNet):
    def __init__(
            self, config: BertConfig,
            dropout=0.1, pool_type="max",
            use_lstm=False, lstm_layers=3, lstm_drop=0.1,
            token_ebd: Optional[Embedding] = None,
            pos_ebd: Optional[Embedding] = None,
            char_ebd: Optional[Embedding] = None,
            char_lstm_layers=1, char_lstm_drop=0.1,

            fpn_layers: int = 8, fpn_drop: float = 0.1,
    ):
        super(PyramidFeatureNet, self).__init__(
            config,
            dropout, pool_type,
            use_lstm, lstm_layers, lstm_drop,
            token_ebd, pos_ebd, char_ebd, char_lstm_layers, char_lstm_drop,
            fpn_layers,
        )
        self.dropout_layer = nn.Dropout(fpn_drop)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.input_encoding_norm = nn.LayerNorm(config.hidden_size)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 其实卷积和linear也没有什么差别吧
        init_linear(self.combine_layer)

    @property
    def output_feature_size(self):
        return self.config.hidden_size

    def forward(
        self,
        encodings: Tensor,
        encoding_masks: Tensor,
        token2encoding_masks: Tensor,
        token2start: Tensor,
        token_masks: Tensor,
        pos_encoding: Optional[Tensor] = None,
        w2v_encoding: Optional[Tensor] = None,
        char_encoding: Optional[Tensor] = None,
        token_masks_char: Optional[Tensor] = None,
        char_count: Optional[Tensor] = None,
    ):
        # padding_mask = ~token_masks
        padding_mask = ~token_masks
        encoder_outputs = self.encoder.forward(
            encodings=encodings,
            encoding_masks=encoding_masks,
            token2encoding_masks=token2encoding_masks,
            token2start=token2start,
            token_masks=token_masks,
            pos_encoding=pos_encoding,
            w2v_encoding=w2v_encoding,
            char_encoding=char_encoding,
            token_masks_char=token_masks_char,
            char_count=char_count,
        )
        h_lay = encoder_outputs

        layer_features = []
        layer_padding_mask = []
        h_lay = self.input_encoding_norm.forward(h_lay)

        num_layer = min(self.num_layer, h_lay.shape[1])

        for i in range(num_layer):
            if i == 0:
                mask = padding_mask
            else:
                h_lay = torch.cat([h_lay[:, :-1], h_lay[:, 1:]], dim=-1)  # (B, T, 2*H)
                h_lay = self.combine_layer.forward(h_lay)  # (B, T, H)

                mask = padding_mask[:, i:]
            layer_padding_mask.append(mask)

            h_lay = self.norm.forward(h_lay)
            h_lay = self.dropout_layer.forward(h_lay)

            layer_features.append(h_lay)

        return {
            "encoder_outputs": encoder_outputs,
            "features_list": layer_features,
            "padding_masks_list": layer_padding_mask,
        }
