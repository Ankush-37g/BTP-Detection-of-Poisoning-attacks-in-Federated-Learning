import torch
import numpy as np
from sklearn.cluster import KMeans
from proposed.aggregation.baselines import fedcc, linear_cka

global_w = {
    "k1": torch.tensor([0.0]),
    "k2": torch.tensor([0.0]),
    "pl_layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), # keys[-4]
    "k4": torch.tensor([0.0]),
    "k5": torch.tensor([0.0]),
    "k6": torch.tensor([0.0])
}
torch.manual_seed(42)
b1_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b2_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b3_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
def make_client(pl):
    return {
        "k1": torch.tensor([0.0]),
        "k2": torch.tensor([0.0]),
        "pl_layer": pl,
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

scores = []
for u, _ in client_updates:
    scores.append([linear_cka(u["pl_layer"], global_w["pl_layer"])])
print("Scores:", scores)
