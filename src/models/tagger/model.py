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
