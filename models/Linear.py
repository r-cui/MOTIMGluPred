import gin
import torch
import torch.nn as nn

from torch import Tensor


def linear(lookback_len: int, horizon_len: int):
    return Linear(lookback_len, horizon_len)


class Linear(nn.Module):
    def __init__(self, lookback_len: int, horizon_len: int):
        super().__init__()
        self.linear = nn.Linear(lookback_len, horizon_len)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        preds = self.linear(x[:, :, 0]).unsqueeze(-1)
        return preds
