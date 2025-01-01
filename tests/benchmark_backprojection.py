#!/usr/bin/env python
import torch
import torchbp
import time
import numpy as np
import torch.utils.benchmark as benchmark

device = "cuda"

nbatch = 1
nr = 1024
ntheta = 1024
nsweeps = 1024
nsamples = 1024
data_dtype = torch.complex64

fc = 6e9
r_res = 0.5

grid_polar = {"r": (10, 500), "theta": (-1, 1), "nr": nr, "ntheta": ntheta}

data = torch.randn((nbatch, nsweeps, nsamples), dtype=data_dtype, device=device)

pos = torch.randn((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
vel = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)
att = torch.zeros((nbatch, nsweeps, 3), dtype=torch.float32, device=device)

pos.requires_grad = True

backprojs = nbatch * nr * ntheta * nsweeps

iterations = 10

tf = benchmark.Timer(
    stmt='torchbp.ops.backprojection_polar_2d(data, grid_polar, fc, r_res, pos, vel, att)',
    setup='import torchbp',
    globals={'data': data, 'grid_polar': grid_polar, 'fc': fc, 'r_res': r_res, 'pos': pos, 'vel': vel, 'att': att})

tb = benchmark.Timer(
    stmt='torch.mean(torch.abs(torchbp.ops.backprojection_polar_2d(data, grid_polar, fc, r_res, pos, vel, att))).backward()',
    setup='import torchbp; ',
    globals={'data': data, 'grid_polar': grid_polar, 'fc': fc, 'r_res': r_res, 'pos': pos, 'vel': vel, 'att': att})

f = tf.timeit(iterations).median
print(f"Device {device}, Forward: {backprojs / f:.3g} backprojections/s")
b = tb.timeit(iterations).median
print(f"Device {device}, Backward: {backprojs / (b - f):.3g} backprojections/s")
