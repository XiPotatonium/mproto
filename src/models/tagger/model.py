from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Embedding
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from alchemy.util import new_huggingface_model


class Tagger(BertPreTrainedModel):
    def __init__(
        self,
        config,
        num_tags: int,
        dropout=0.1,
    ):
        super().__init__(config)

        self.encoder = new_huggingface_model(config)

        self.num_tags = num_tags
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(config.hidden_size, self.num_tags)

    def forward(
        self,
        encodings: Tensor,
        encoding_masks: Tensor,
        token2start: Tensor,
    ):
        # padding_mask = ~token_masks
        hidden = self.encoder.forward(input_ids=encodings, attention_mask=encoding_masks.float())
        if isinstance(hidden, BaseModelOutputWithPoolingAndCrossAttentions):
            hidden = hidden.last_hidden_state

        # Decoding
        _, _, hidden_size = hidden.size()
        start_index = token2start.unsqueeze(-1).expand(-1, -1, hidden_size)
        hidden = torch.gather(hidden, dim=1, index=start_index)
        logits = self.linear.forward(self.dropout.forward(hidden))

        return hidden, logits


class CosineProtoTagger(BertPreTrainedModel):
    def __init__(
        self,
        config,
        type_ebd: Tensor,
        use_learnable_scalar: bool,
        dropout=0.1,
    ):
        super().__init__(config)

        self.encoder = new_huggingface_model(config)

        self.type_ebd = nn.Parameter(type_ebd)
        self.dropout = nn.Dropout(dropout)
        if use_learnable_scalar:
            self.scalar = nn.Parameter(torch.randn(1))
        else:
            self.register_buffer("scalar", torch.ones(1))

    def similarity(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(F.normalize(a, dim=-1), F.normalize(b, dim=-1).transpose(-1, -2))

    def forward(
        self,
        encodings: Tensor,
        encoding_masks: Tensor,
        token2start: Tensor,
    ):
        # padding_mask = ~token_masks
        hidden = self.encoder.forward(input_ids=encodings, attention_mask=encoding_masks.float())
        if isinstance(hidden, BaseModelOutputWithPoolingAndCrossAttentions):
            hidden = hidden.last_hidden_state

        # Decoding
        _, _, hidden_size = hidden.size()
        start_index = token2start.unsqueeze(-1).expand(-1, -1, hidden_size)
        hidden = torch.gather(hidden, dim=1, index=start_index)
        logits = self.similarity(self.dropout.forward(hidden), self.type_ebd.unsqueeze(0))

        return hidden, logits, self.scalar * logits


class L2ProtoTagger(CosineProtoTagger):
    def similarity(self, a: Tensor, b: Tensor) -> Tensor:
        return -torch.cdist(a, b, p=2)

    def forward(self, encodings: Tensor, encoding_masks: Tensor, token2start: Tensor):
        # padding_mask = ~token_masks
        hidden = self.encoder.forward(input_ids=encodings, attention_mask=encoding_masks.float())
        if isinstance(hidden, BaseModelOutputWithPoolingAndCrossAttentions):
            hidden = hidden.last_hidden_state

        # Decoding
        bsz, seq_len, hidden_size = hidden.size()
        start_index = token2start.unsqueeze(-1).expand(-1, -1, hidden_size)
        hidden = torch.gather(hidden, dim=1, index=start_index)
        logits = self.similarity(
            self.dropout.forward(hidden),
            self.type_ebd.unsqueeze(0).repeat(bsz, 1, 1),
        )

        return hidden, logits, self.scalar * logits
