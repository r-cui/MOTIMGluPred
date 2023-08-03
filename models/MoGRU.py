import gin
import torch
import torch.nn as nn

from torch import Tensor


def mogru(horizon_len: int):
    return MoGRU(horizon_len)


class MoGRU(nn.Module):
    def __init__(self, horizon_len: int):
        super().__init__()
        layer_size = 512
        self.gru = nn.GRU(input_size=1,
                          hidden_size=layer_size,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True)
        # self.linears = nn.ModuleList([nn.Linear(layer_size, 1) for _ in range(horizon_len)])
        self.linear = nn.Linear(layer_size, horizon_len)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        _, hn = self.gru(x)
        hn = hn[-1]  # last layer
        pred = self.linear(hn)
        # pred = torch.cat([x(hn) for x in self.linears], dim=1)
        return pred.unsqueeze(-1)

# seqmo
# class MoGRU(nn.Module):
#     def __init__(self, horizon_len: int):
#         super().__init__()
#         layer_size = 512
#         self.gru = nn.GRU(
#             input_size=1,
#             hidden_size=layer_size,
#             num_layers=2,
#             bidirectional=False,
#             batch_first=True
#         )
#         self.dec = nn.GRU(
#             input_size=layer_size,
#             hidden_size=layer_size,
#             num_layers=2,
#             bidirectional=False,
#             batch_first=True
#         )
#         # self.linear = nn.Linear(layer_size, 1)
#         self.linear = nn.Sequential(
#             nn.Linear(layer_size, layer_size),
#             nn.ReLU(),
#             nn.Linear(layer_size, 1)
#         )
#         self.horizon_len = horizon_len
#
#     def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
#         _, enc_hn = self.gru(x)
#         enc_out = enc_hn[-1]  # last layer
#         enc_out = torch.unsqueeze(enc_out, 1)
#         x = enc_out
#         x_list = []
#         for i in range(self.horizon_len):
#             if i == 0:
#                 x, dec_hn = self.dec(x)
#             else:
#                 x, dec_hn = self.dec(x, dec_hn)
#             x_list.append(x)
#         x_list = torch.cat(x_list, dim=1)
#         return self.linear(x_list)
