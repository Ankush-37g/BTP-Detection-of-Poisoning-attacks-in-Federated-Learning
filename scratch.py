import torch
import numpy as np
def linear_cka(X, Y):
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)
    hsic_xy = torch.norm(torch.mm(X_c.t(), Y_c), p='fro') ** 2
    hsic_xx = torch.norm(torch.mm(X_c.t(), X_c), p='fro') ** 2
    hsic_yy = torch.norm(torch.mm(Y_c.t(), Y_c), p='fro') ** 2
    if hsic_xx == 0 or hsic_yy == 0: return 0.0
    return (hsic_xy / torch.sqrt(hsic_xx * hsic_yy)).item()

global_pl = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
torch.manual_seed(42)
b1_pl = global_pl + torch.randn(3, 2) * 0.1
b2_pl = global_pl + torch.randn(3, 2) * 0.1
a1_pl = torch.randn(3, 2)
a2_pl = torch.randn(3, 2)
print("b1:", linear_cka(b1_pl, global_pl))
print("b2:", linear_cka(b2_pl, global_pl))
print("a1:", linear_cka(a1_pl, global_pl))
print("a2:", linear_cka(a2_pl, global_pl))
