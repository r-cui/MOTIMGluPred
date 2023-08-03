import torch
import time
import numpy as np
from thop import profile, clever_format

from utils.ops import default_device, to_tensor


def test_params_flop(model, dataloader):
    model.eval()
    x, y, x_time, y_time = map(to_tensor, next(iter(dataloader)))
    macs, params = profile(model, inputs=(x, x_time, y_time))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'INFO: MACs: {macs}')
    print(f'INFO: Params: {params}')

    model_params = sum(p.numel() for p in model.parameters())
    print('INFO: Parameter count: {:.2f}M'.format(model_params / 1000000.0))

    times = []
    count = 0
    while count < 1000:
        for it, data in enumerate(dataloader):
            x, y, x_time, y_time = map(to_tensor, data)
            start = time.time()
            _ = model(x, x_time, y_time)
            end = time.time()
            times.append((end - start) * 1000.0)
            count += 1
            if count == 100:
                break
    print('INFO: Inference time: {:.2f} ms ({:.2f})'.format(np.mean(times), np.std(times)))
    return