from typing import Union

import torch

from .TIML import timl
from .Dummy import dummy
from .Linear import linear
from .MoGRU import mogru
from .SeqConv import seqconv
from .ReSelfAttn import reselfattn


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'timl':
        model = timl(datetime_feats=kwargs['datetime_feats'])
    elif model_type == 'dummy':
        model = dummy()
    elif model_type == 'linear':
        model = linear(lookback_len=kwargs['lookback_len'], horizon_len=kwargs['horizon_len'])
    elif model_type == 'mogru':
        model = mogru(horizon_len=kwargs['horizon_len'])
    elif model_type == 'seqconv':
        model = seqconv(horizon_len=kwargs['horizon_len'])
    elif model_type == 'reselfattn':
        model = reselfattn(horizon_len=kwargs['horizon_len'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
