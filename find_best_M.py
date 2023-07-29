import data
import create_graphs
import numpy as np
from tqdm import tqdm
import random
import networkx as nx

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



graphs = create_graphs.create(
    dotdict({"graph_type": "planar"})
)
print(len(graphs))
print(graphs[0].nodes())
print(graphs[1].nodes())
print()

max_Ms = 0
N_ITER = 1000
with tqdm(total=len(graphs) * N_ITER) as pbar:
    for i in range(len(graphs)):
        for _ in range(N_ITER):
            start_n = random.choice(list(graphs[i].nodes())) #np.random.choice([n[0] for n in graphs[i].nodes()], size=1)[0]
            #end_n = np.random.choice([n[1] for n in graphs[i].nodes()], size=1)[0]

            out, distrib_queue = data.bfs_seq(graphs[i], start_n) # (start_n, end_n)
            tmp = max(distrib_queue)
            if tmp > max_Ms:
                max_Ms = tmp
            pbar.update(1)
print(max_Ms)
