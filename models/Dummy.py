import torch
import torch.nn as nn
from torch import Tensor


def dummy():
    return Dummy()


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, c = x.shape
        preds = x[:, -1:, :].repeat(1, tgt_horizon_len, 1)
        return preds
