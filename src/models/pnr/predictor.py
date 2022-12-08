from typing import Optional

import torch
from torch import nn, Tensor


def my_cosine_similarity(feat: Tensor, cls_ebd: Tensor, eps=1e-08):
    sim = torch.einsum("bcd,bzd->bcz", feat, cls_ebd.unsqueeze(0).expand(feat.size()[0], -1, -1))
    cls_ebd_l2 = torch.norm(cls_ebd, p=2, dim=-1)
    return sim / torch.max(cls_ebd_l2, torch.ones_like(cls_ebd_l2) * eps)


class MaskAwareClassifier(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int):
        super(MaskAwareClassifier, self).__init__()

        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: Tensor, padding_masks: Optional[Tensor] = None) -> Tensor:
        """

        Args:
            x:
            masks (Optional[Tensor], optional): 

        Returns:

        """
        x = self.classifier.forward(x)
        if padding_masks is not None:
            x.masked_fill_(padding_masks.unsqueeze(-1), 0)
        return x


class MaskAwareSSNFuse(nn.Module):
    """fuse bert embeddings with query embeddings"""

    def __init__(self, embed_dim: int, kdim: Optional[int] = None):
        super(MaskAwareSSNFuse, self).__init__()
        self.W = nn.Linear(embed_dim + (embed_dim if kdim is None else kdim), embed_dim)
        self.v = nn.Linear(embed_dim, 1)

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        key_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """

        Args:
            query: bsz, n_queries, embed_dim
            key: bsz, sent_len, kdim
            key_mask:
            query_mask (Optional[Tensor], optional): (bsz, n_queries)

        Returns:

        """
        key_embed = key.unsqueeze(1).expand(-1, query.size(1), -1, -1)
        query_embed = query.unsqueeze(2).expand(-1, -1, key.size(-2), -1)

        fuse = torch.cat([key_embed, query_embed], dim=-1)
        x = self.W.forward(fuse)
        x = self.v.forward(torch.tanh(x)).squeeze(-1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x[key_padding_mask] = -1e25

        if query_padding_mask is not None:
            x.masked_fill_(query_padding_mask.unsqueeze(-1), 0)  # mask fill一些不可导的值

        return x
