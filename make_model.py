# !pip install torch==2.1.0
import numpy as np
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
        self.active_it2ind = self.it2ind
        self.active_ind2it = self.ind2it
        self.active_emb = self.emb.weight


    @torch.jit.export
    def forward(self, item_name: str, size: int = 5)-> Dict[str, float]:
        ind = torch.tensor(self.it2ind[item_name])
        u = self.emb(ind)
        scores = u @ self.active_emb.t()
        if size > len(scores):
            size = len(scores)
        s, i = scores.topk(size)
        resp = {self.active_ind2it[ind.item()]: float(score.item()) 
                for ind, score in zip(i.squeeze(), s.squeeze())}
        return resp


    @torch.jit.export
    def set_active_items(self, active: List[str]):
        new_items = [self.it2ind[x] for x in active]
        new_emb = self.emb.weight[new_items]
        self.active_ind2it = {new_ind: self.ind2it[old_ind] for new_ind, old_ind in enumerate(new_items)}
        self.active_it2ind = {v:k for k,v in self.active_ind2it.items()}
        self.active_emb = new_emb


    def export(self, path='model.pt'):
        m = torch.jit.script(self)
        m.save(path)
        return m
    

model_data = {'dna': [1.0, 1.0, 1.0], 'inntrengeren': [0.2, 0.2, 0.2]}
model_data.update({str(i): np.random.randn(3).tolist() for i in range(1000)})

vectors = np.array([x for x in model_data.values()])
name2ind = {n:i for i, n in enumerate(model_data.keys())}

model = MLTModel(name2ind, vectors)

model("dna")

sm = model.export()

sm("dna")

sm.set_active_items(["dna", "inntrengeren"])

#i = sm.it2ind["dna"]
#it = sm.emb.weight[i]
#print(i, it)