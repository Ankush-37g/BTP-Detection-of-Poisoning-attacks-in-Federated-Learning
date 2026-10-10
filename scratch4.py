import os, sys, torch
import numpy as np
from sklearn.cluster import KMeans
from proposed.aggregation.baselines import fedcc, linear_cka

global_w = {
    "pl_layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
}
torch.manual_seed(42)
b1_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b2_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
b3_pl = global_w["pl_layer"] + torch.randn(3, 2) * 0.1
a1_pl = torch.randn(3, 2) * 10
a2_pl = torch.randn(3, 2) * 10

scores = []
for pl in [b1_pl, b2_pl, b3_pl, a1_pl, a2_pl]:
    scores.append([linear_cka(pl, global_w["pl_layer"])])
    
print("Scores:", scores)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(np.array(scores))
print("Labels:", labels)
