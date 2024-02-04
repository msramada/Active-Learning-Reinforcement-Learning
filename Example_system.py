import torch

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.

rx = 3
ru = 1
ry = 1

# Noise covariance, according to Anderson & Moore Optimal Filtering
Q = torch.diag(torch.tensor([0.2, 0.2, 0.2]))
R = torch.tensor([[0.1]])


def stateDynamics(x,u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    f = torch.zeros(rx,)
    A = torch.tensor([[0.92, 0.7, -0.4],[0, 0.95, -0.1],[0, 0, 0.93]])
    B = torch.tensor([[0],[0],[1.0]])
    f = A @ x + B @ u
    return torch.atleast_2d(f.squeeze()).T


def measurementDynamics(x):
    x = torch.atleast_1d(x.squeeze())
    gx = torch.zeros(ry,)
    helper = torch.tanh(torch.tensor([x[0]+x[1]-2.0]))
    gx = helper
    return torch.atleast_2d(gx.squeeze()).T

