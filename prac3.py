import torch

# 1. Activation function
def sigma(x):
    return x.tanh()

def dsigma(x):
    return x.cosh().pow(-2)

# 2. Loss
"""
def loss(v, t):
    return torch.norm(t-v, 2)**2

def dloss(v, t):
    return -2*torch.norm(t-v, 2)
"""
def loss(v,t):
    return (v-t).pow(2).sum()

def dloss(v,t):
    return 2*(v-t)

# 3. Forward and backward passes
def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)
    return x0, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
                  t,
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dl_dx2 * dsigma(s2)
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dl_dx1 * dsigma(s1)

    dl_dw2.add_(dl_ds2.view(-1,1).mm(x1.view(1,-1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1,1).mm(x0.view(1,-1)))
    dl_db1.add_(dl_ds1)

