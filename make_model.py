# !pip install torch==2.1.0
import numpy as np
import json
import torch
import torch.nn as nn
from typing import *

class MLTModel(nn.Module):
    def __init__(self, it2ind_mapping: Dict[str, int], vectors):
        super(MLTModel, self).__init__()
        n_items = len(vectors)
        n_factors = len(vectors[0])
        self.emb = nn.Embedding(n_items, n_factors, _weight=torch.from_numpy(vectors))
        self.it2ind = it2ind_mapping
        self.ind2it = {v:k for k,v in self.it2ind.items()}

    # model forward pass
    def forward(self, user: str, size: int = 5)-> Dict[str, float]:
        ind = torch.tensor(self.it2ind[user])
        u = self.emb(ind)
        scores = u @ self.emb.weight.t()
        s, i = scores.topk(size)
        resp = {self.ind2it[ind.item()]: float(score.item()) for ind, score in zip(i.squeeze(), s.squeeze())}
        return resp

    def export(self, path='model.pt'):
        m = torch.jit.script(self)
        m.save(path)
        return m
    



with open('model.json', 'r') as f:
    rows = [json.loads(line) for line in f]

real_model_data = {x['contentId']: x['factors'] for x in rows}

vectors = np.array([x for x in real_model_data.values()])
name2ind = {n:i for i, n in enumerate(real_model_data.keys())}

model = MLTModel(name2ind, vectors)


model("dna")

sm = model.export()

sm("dna")

