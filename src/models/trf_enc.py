from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding
from loguru import logger
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from alchemy.util import new_huggingface_model


def pooling(sub: Tensor, sup_mask: Tensor, pool_type: str = "max"):
    sup = None
    if len(sub.shape) == len(sup_mask.shape):
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else:
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    return sup


class TrfEncoder(nn.Module):

    def __init__(
            self,
            config,
            ebd_drop: float = 0.1,
            pool_type: str = "mean",
            use_lstm: bool = False,
            lstm_layers: int = 3,
            lstm_drop: float = 0.1,
            token_ebd: Optional[Embedding] = None,
            pos_ebd: Optional[Embedding] = None,
            char_ebd: Optional[Embedding] = None,
            char_lstm_layers: int = 1,
            char_lstm_drop: float = 0.2,
    ):
        super(TrfEncoder, self).__init__()
        self.config = config
        self.ebd_drop = nn.Dropout(ebd_drop)
        self.pool_type = pool_type
        self.use_lstm = use_lstm
        lstm_input_dim = config.hidden_size
        self.token_ebd = token_ebd
        if self.token_ebd is not None:
            lstm_input_dim += token_ebd.embedding_dim
        self.pos_ebd = pos_ebd
        if self.pos_ebd is not None:
            lstm_input_dim += pos_ebd.embedding_dim
        self.char_ebd = char_ebd
        if self.char_ebd is not None:
            lstm_input_dim += char_ebd.embedding_dim * 2        # BiLSTM
            self.char_lstm = nn.LSTM(
                input_size=char_ebd.embedding_dim, hidden_size=char_ebd.embedding_dim,
                num_layers=char_lstm_layers, bidirectional=True,
                dropout=char_lstm_drop, batch_first=True
            )
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim, hidden_size=config.hidden_size // 2, num_layers=lstm_layers,
                bidirectional=True, dropout=lstm_drop, batch_first=True
            )
        elif self.token_ebd is not None or self.pos_ebd is not None or self.char_ebd is not None:
            self.reduce_dimension = nn.Linear(lstm_input_dim, config.hidden_size)

        self.trf = new_huggingface_model(config)

    def max_positions(self):
        """和Bert一样

        Returns:
            _type_: _description_
        """
        return self.config.max_position_embeddings

    def with_pretrained(self, path: str):
        logger.info("Use \"{}\" to initialize {} encoder".format(
            path, self.trf.__class__.__name__
        ))
        self.trf = self.trf.__class__.from_pretrained(path)

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
        bsz, _ = encodings.size()
        hidden = self.trf.forward(input_ids=encodings, attention_mask=encoding_masks.float())
        if isinstance(hidden, BaseModelOutputWithPoolingAndCrossAttentions):
            hidden = hidden.last_hidden_state
        token_count = token_masks.long().sum(-1, keepdim=True)
        # 到这里embed变成token序列而不是BPE序列
        if self.pool_type == "start":
            _, _, hidden_size = hidden.size()
            token2start = token2start.unsqueeze(-1).expand(-1, -1, hidden_size)
            h_token = torch.gather(hidden, dim=1, index=token2start)
        else:
            h_token = pooling(hidden, token2encoding_masks, self.pool_type)
        embeds = [h_token]

        if self.token_ebd is not None and w2v_encoding is not None:
            word_embed = self.token_ebd.forward(w2v_encoding)
            word_embed = self.ebd_drop.forward(word_embed)
            embeds.append(word_embed)
        if self.pos_ebd is not None and pos_encoding is not None:
            pos_embed = self.pos_ebd.forward(pos_encoding)
            pos_embed = self.ebd_drop.forward(pos_embed)
            embeds.append(pos_embed)
        if (
            self.char_ebd is not None and
            char_encoding is not None and
            token_masks_char is not None and
            char_count is not None
        ):
            char_count = char_count.view(-1)
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count * bsz, max_char_count)

            char_encoding[char_count == 0][:, 0] = 101
            char_count[char_count == 0] = 1
            char_embed = self.char_ebd.forward(char_encoding)
            char_embed = self.ebd_drop.forward(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(
                input=char_embed, lengths=char_count.tolist(),
                enforce_sorted=False, batch_first=True
            )
            char_embed_packed_o, (_, _) = self.char_lstm.forward(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(bsz, max_token_count, max_char_count, self.char_ebd.embedding_dim * 2)
            h_token_char = pooling(char_embed, token_masks_char, "mean")
            embeds.append(h_token_char)

        h_token = torch.cat(embeds, dim=-1)

        if self.use_lstm:
            h_token = nn.utils.rnn.pack_padded_sequence(
                input=h_token, lengths=token_count.squeeze(-1).cpu().tolist(),
                enforce_sorted=False, batch_first=True
            )
            h_token, (_, _) = self.lstm.forward(h_token)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)
        elif len(embeds) > 1:
            h_token = self.reduce_dimension.forward(h_token)

        return h_token
