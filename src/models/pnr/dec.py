import copy
from typing import Optional
from torch import nn
import torch.nn.functional as F
import torch
from transformers import BertConfig


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def repeat_modules(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, ca_kdim: Optional[int] = None, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, kdim=ca_kdim, vdim=ca_kdim, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self, tgt: torch.Tensor, pos: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor,
        sa_q_mask: Optional[torch.Tensor] = None, ca_q_mask: Optional[torch.Tensor] = None,
    ):
        """

        Args:
            tgt:
            pos:
            src:
            src_mask:
            sa_q_mask (Optional[torch.Tensor], optional): (bsz, n_queries, n_queries)
            ca_q_mask (Optional[torch.Tensor], optional): (bsz, n_queries, len_seq)

        Returns:

        """
        # self attention
        q = k = self.with_pos_embed(tgt, pos)
        v = tgt
        if sa_q_mask is not None:
            sa_q_mask = sa_q_mask.unsqueeze(1).expand(-1, self.self_attn.num_heads, -1, -1).flatten(0, 1)
        tgt2 = self.self_attn.forward(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                      attn_mask=sa_q_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, pos)
        k = v = src
        if ca_q_mask is not None:
            ca_q_mask = ca_q_mask.unsqueeze(1).expand(-1, self.cross_attn.num_heads, -1, -1).flatten(0, 1)
        tgt2 = self.cross_attn.forward(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=src_mask, attn_mask=ca_q_mask
        )[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, config: BertConfig, decoder_layer: nn.Module, num_layers: int):
        super(TransformerDecoder, self).__init__()
        self.layers = repeat_modules(decoder_layer, num_layers)

    def forward(
        self, tgt, query_pos, src, src_padding_mask, sa_q_mask: Optional[torch.Tensor] = None,
    ):
        """

        Args:
            tgt:
            query_pos:
            src:
            src_padding_mask:
            sa_q_mask: (bsz, num_queries, num_queries), True for mask

        Returns:

        """

        bs, n_q, _ = tgt.size()
        output = tgt

        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                output, query_pos, src, src_padding_mask, sa_q_mask=sa_q_mask,
            )

            intermediate.append(output)

        return intermediate
