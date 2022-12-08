from typing import Tuple

import torch


def generate_boundary(max_len: int, max_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        max_len:
        max_depth:

    Returns:
        boundary matrix (torch.Tensor), (max_depth, max_position_embedding, 2), dtype=torch.int64
    """
    assert max_depth <= max_len, "Max length ({}) should be greater than max depth ({})".format(max_len, max_depth)
    left_b = torch.arange(0, max_len, dtype=torch.int64)
    right_b = left_b
    left_boundaries = [left_b[:i] for i in range(max_len, max_len - max_depth, -1)]
    right_boundaries = [right_b[i:] for i in range(max_depth)]
    left_boundaries = torch.nn.utils.rnn.pad_sequence(left_boundaries, batch_first=True)
    right_boundaries = torch.nn.utils.rnn.pad_sequence(right_boundaries, batch_first=True)
    return left_boundaries, right_boundaries


def generate_soft_boundary(max_len: int, sigma: float = 1.0) -> torch.Tensor:
    token_indexes = torch.arange(0, max_len, 1)
    soft_boundary = torch.stack([
        torch.distributions.normal.Normal(torch.as_tensor([i]), torch.as_tensor([sigma])).log_prob(token_indexes).exp()
        for i in range(max_len)
    ])
    return soft_boundary
    
