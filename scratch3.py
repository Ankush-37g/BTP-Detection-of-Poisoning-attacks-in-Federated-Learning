import os, sys, torch
from proposed.aggregation.baselines import fedcc
global_w = {
    "k1": torch.tensor([0.0]),
    "pl_layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "k3": torch.tensor([0.0]),
    "k4": torch.tensor([0.0]),
    "k5": torch.tensor([0.0]),
    "k6": torch.tensor([0.0])
}
b1_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b2_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b3_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
def make_client(pl):
    return {
        "k1": torch.tensor([0.0]),
        "pl_layer": pl,
        "k3": torch.tensor([0.0]),
        "k4": torch.tensor([0.0]),
        "k5": torch.tensor([0.0]),
        "k6": torch.tensor([0.0])
    }
client_updates = [
    (make_client(b1_pl), 10),
    (make_client(b2_pl), 10),
    (make_client(b3_pl), 10),
    (make_client(torch.randn(3, 2)), 10),
    (make_client(torch.randn(3, 2)), 10)
]
agg = fedcc(client_updates, global_weights=global_w)
print("Aggregated pl_layer:")
print(agg["pl_layer"])
