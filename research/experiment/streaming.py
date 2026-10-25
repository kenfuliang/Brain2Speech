"""Stateful, bin-at-a-time causal inference; no retrospective prefix replay."""
import time
import numpy as np
import torch
from research.train import synchronize


def stream(model, x, device):
    model.eval()
    inference_state = generator_state = None
    outputs, costs = [], []
    with torch.no_grad():
        for i, row in enumerate(x):
            synchronize(device)
            started = time.perf_counter()
            bin_value = torch.as_tensor(row[None,None], dtype=torch.float32, device=device)
            readin = torch.tanh(model.readin(model.adapter(bin_value)))
            inferred, inference_state = model.inference(readin, inference_state)
            inputs = inferred if model.kind=='gru' else model.input_posterior(inferred).chunk(2,-1)[0]
            generated, generator_state = model.generator(inputs, generator_state)
            if i % 4 == 3:
                outputs.append(model.head(model.factors(generated))[0,0].cpu().numpy())
            synchronize(device)
            costs.append((time.perf_counter()-started)*1000)
    return np.stack(outputs), {'mean_bin_compute_ms':float(np.mean(costs)),
                              'p95_bin_compute_ms':float(np.quantile(costs,.95)),
                              'acquisition_ms':4, 'bin_ms':20, 'output_interval_ms':80,
                              'output_scheduling_range_ms':[0,60],
                              'scope':'Neural causal-bin processing; excludes WFST and communication endpointing'}
