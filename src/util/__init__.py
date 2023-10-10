from typing import List
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def debug_detect_value_anomaly(value: Tensor, tag: str = "DEBUG"):
    if torch.any(torch.isnan(value)):
        raise ValueError("{}: nan {}".format(tag, value))
    if torch.any(torch.isneginf(value)):
        raise ValueError("{}: neginf {}".format(tag, value))
    if torch.any(torch.isinf(value)):
        raise ValueError("{}: inf {}".format(tag, value))


def pad_sequence_numpy(arrays: List[np.ndarray], padding_value=0.0) -> np.ndarray:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(arr) for arr in arrays],
        batch_first=True,
        padding_value=padding_value
    ).numpy()


def js_dist(logits1: Tensor, logits2: Tensor):
    log_prob1 = F.log_softmax(logits1, dim=-1)
    log_prob2 = F.log_softmax(logits2, dim=-1)
    return 0.5 * (
        F.kl_div(log_prob1, log_prob2, log_target=True, reduction="none")
        + F.kl_div(log_prob2, log_prob1, log_target=True, reduction="none")
    )


def sim_matrix(a: Tensor, b: Tensor, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
