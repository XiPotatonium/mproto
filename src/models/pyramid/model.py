from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Embedding
from transformers import BertPreTrainedModel, BertConfig
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .init_util import init_linear, init_lstm
from ..trf_enc import TrfEncoder


class NGramEncoding(nn.Module):

    def __init__(self, hidden_size: int, input_dim=None, ngram=2, padding=0):
        super().__init__()

        if input_dim is None:
            input_dim = hidden_size

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_size,
            kernel_size=ngram,
            padding=padding,
        )

    def forward(self, x):
        # x (B, T, in)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class LSTMEncoding(nn.Module):

    def __init__(self, hidden_size: int, dropout: float, input_dim=None, num_layers=1):
        super().__init__()
        self.bias = True  # config.bias
        self.batch_first = True  # config.batch_first
        self.dropout = dropout
        self.bidirectional = True  # config.bidirectional
        self.input_dim = hidden_size if input_dim is None else input_dim

        k_bidirectional = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            self.input_dim, hidden_size // k_bidirectional,
            num_layers, self.bias, self.batch_first, dropout, self.bidirectional
        )
        init_lstm(self.lstm)

    def forward(self, inputs, return_cls=False, mask=None, lens=None):
        batch_size = inputs.shape[0] if self.batch_first else input.shape[1]
        hidden = None

        if mask is not None or lens is not None:
            if lens is not None:
                word_seq_lens = lens
            else:
                word_seq_lens = mask.sum(dim=-1)
            word_seq_lens = word_seq_lens + (word_seq_lens == 0).long()  # avoid length == 0
            word_rep = inputs
            sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
            _, recover_idx = permIdx.sort(0, descending=False)
            sorted_seq_tensor = word_rep[permIdx]

            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, (h, _) = self.lstm.forward(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            outputs = lstm_out[recover_idx]
            hidden = torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1)
            hidden = hidden[recover_idx]
        else:
            outputs, (h, c) = self.lstm(inputs, hidden)
            hidden = torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1)

        if return_cls:
            return outputs, hidden
        else:
            return outputs


class BiPyramid(BertPreTrainedModel):
    """这个和原来的BiPyramidNestedNER不一样，顶层不做tagging了直接分类，大于枚举长度的放弃

    Args:
        BertPreTrainedModel (_type_): _description_

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(
            self, config: BertConfig, num_types: int, max_depth: int,

            dropout=0.1, pool_type="max",
            use_lstm=False, lstm_layers=3, lstm_drop=0.1,
            token_ebd: Optional[Embedding] = None,
            pos_ebd: Optional[Embedding] = None,
            char_ebd: Optional[Embedding] = None,
            char_lstm_layers=1, char_lstm_drop=0.1,
    ):
        super(BiPyramid, self).__init__(config)
        self.encoder = TrfEncoder(
            config=config,
            ebd_drop=dropout,
            pool_type=pool_type,
            use_lstm=use_lstm, lstm_layers=lstm_layers, lstm_drop=lstm_drop,
            token_ebd=token_ebd,
            pos_ebd=pos_ebd,
            char_ebd=char_ebd, char_lstm_layers=char_lstm_layers, char_lstm_drop=char_lstm_drop
        )
        self.max_depth = max_depth      # 注意如果要生成长度为8的span的表示的话，depth是7

        self.predictor = nn.Linear(config.hidden_size * 2, num_types)

        init_linear(self.predictor)
        self.split_layer = NGramEncoding(config.hidden_size, config.hidden_size * 2, ngram=2, padding=1)

        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        init_linear(self.combine_layer)

        self.dropout = nn.Dropout(dropout)
        #         self.combine_layer = NGramEncoding(self.config, ngram=2)
        self.reuse_decoding = LSTMEncoding(config.hidden_size, lstm_drop)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
            self,
            encodings: Tensor,
            encoding_masks: Tensor,
            token2encoding_masks: Tensor,
            token2start: Tensor,
            token_masks: Tensor,
            pos_encoding: Optional[Tensor] = None,
            w2v_encoding: Optional[Tensor] = None,
    ):
        # padding_mask = ~token_masks
        bsz, _ = encodings.size()
        padding_mask = ~token_masks
        h_token = self.encoder.forward(
            encodings=encodings,
            encoding_masks=encoding_masks,
            token2encoding_masks=token2encoding_masks,
            token2start=token2start,
            token_masks=token_masks,
            pos_encoding=pos_encoding,
            w2v_encoding=w2v_encoding,
        )

        embeddings_list = []
        embeddings_list_inv = []
        mask_list = []

        max_depth = self.max_depth if self.max_depth is not None else h_token.shape[1] - 1
        for i in range(max_depth + 1):

            if i == 0:
                mask = padding_mask
                mask_list.append(mask)
            else:
                if h_token.shape[1] == 1:
                    max_depth = i - 1  # reduce the max_depth if the sentence is too short
                    break

                h_token = torch.cat([h_token[:, :-1], h_token[:, 1:]], dim=-1)  # (B, T, 2*H)
                h_token = self.combine_layer(h_token)  # (B, T, H)

                mask = padding_mask[:, i:]
                mask_list.append(mask)

            h_token = self.norm.forward(h_token)
            h_token = self.dropout.forward(h_token)

            h_token = self.reuse_decoding(h_token)

            h_token = self.dropout.forward(h_token)

            embeddings_list.append(h_token)

        for i in range(max_depth, -1, -1):

            if i == max_depth:

                h_token = torch.zeros_like(h_token)
                embeddings_list_inv.append(h_token)

                continue

            else:
                h_token = self.split_layer(torch.cat([
                    h_token, embeddings_list[i + 1]
                ], dim=-1))

            h_token = self.norm.forward(h_token)
            h_token = self.dropout.forward(h_token)

            h_token = self.reuse_decoding(h_token)

            h_token = self.dropout.forward(h_token)

            embeddings_list_inv.append(h_token)

        logits_list = []

        for i in range(max_depth + 1):
            logits_list.append(
                self.predictor.forward(
                    torch.cat([embeddings_list[i], embeddings_list_inv[-i - 1]], dim=-1)
                )
            )

        return logits_list, mask_list
