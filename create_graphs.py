import networkx as nx
import numpy as np
import random 

import os
from scipy.spatial import Delaunay
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import *
from data import *


class GridDataset(Dataset):
    def __init__(self, grid_start=10, grid_end=20, same_sample=False):
        filename = f'data/grids_{grid_start}_{grid_end}{"_same_sample" if same_sample else ""}.pt'

        if os.path.isfile(filename):
            assert False
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.n_nodes = []
            self.same_sample = same_sample
            for i in range(grid_start, grid_end):
                for j in range(grid_start, grid_end):
                    G = nx.grid_2d_graph(i, j)
                    adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                    self.adjs.append(adj)
                    self.n_nodes.append(len(G.nodes()))
            self.n_max = (grid_end - 1) * (grid_end - 1)
            # torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved with {len(self.adjs)} graphs')

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = self.adjs[idx] #F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        return graph

class PlanarDataset(Dataset):
    def __init__(self, n_nodes, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'planar_{n_nodes}_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            assert False
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(n_graphs):
                # Generate planar graphs using Delauney traingulation
                points = np.random.rand(n_nodes,2)
                tri = Delaunay(points)
                adj = np.zeros([n_nodes,n_nodes])
                for t in tri.simplices:
                    adj[t[0], t[1]] = 1
                    adj[t[1], t[2]] = 1
                    adj[t[2], t[0]] = 1
                    adj[t[1], t[0]] = 1
                    adj[t[2], t[1]] = 1
                    adj[t[0], t[2]] = 1
                G = nx.convert_matrix.from_numpy_matrix(adj)
                adj = torch.from_numpy(adj).float()
            
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
            self.n_max = n_nodes

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = self.adjs[idx] #F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        if self.ignore_first_eigv:
            size_diff += 1
        #graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph
    
class SBMDataset(Dataset):
    def __init__(self, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False, max_comm_size=40):
        filename = f'sbm_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            assert False
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.n_nodes = []
            self.same_sample = same_sample
            for seed in range(n_graphs):
                n_comunities = np.random.random_integers(2, 5)
                comunity_sizes = np.random.random_integers(20, max_comm_size, size=n_comunities)
                probs = np.ones([n_comunities, n_comunities]) * 0.005
                probs[np.arange(n_comunities), np.arange(n_comunities)] = 0.3
                G = nx.stochastic_block_model(comunity_sizes, probs, seed=seed)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()

                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
            self.n_max = max(self.n_nodes)


        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = self.adjs[idx] #F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        if self.ignore_first_eigv:
            size_diff += 1
        return graph
    

class EgoDataset(Dataset):
    def __init__(self, size, same_sample=False):
        assert size in ["small", "large"]
        filename = f'/home/steve.azzolin/DiGress_fork/data/ego/ego_{size}.npy' # TODO: fix this

        self.adjs = []
        self.n_nodes = []
        self.same_sample = same_sample

        graphs = np.load(filename, allow_pickle=True)
        print("Avg num nodes Ego", np.mean([g.shape[0] for g in graphs]))

        for adj in graphs[:]:
            adj = torch.from_numpy(adj).float()
            self.adjs.append(adj)
            self.n_nodes.append(adj.shape[0])
        self.n_max = max(self.n_nodes)
        print("Total num nodes/Avg num  = ", sum(self.n_nodes), max(self.n_nodes), np.mean(self.n_nodes))
        print(f'Dataset {filename} saved with {len(self.adjs)} graphs')

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]
        del graphs

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = self.adjs[idx] #F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        return graph


def create(args):
    ### load datasets
    graphs=[]
    train_idxs, val_idxs, test_idxs = [], [], []
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='tree':
        graphs = []
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    elif args.graph_type=='caveman':
        # graphs = []
        # for i in range(5,10):
        #     for j in range(5,25):
        #         for k in range(5):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graphs.append(caveman_special(i,j, p_edge=0.3))
        args.max_prev_node = 100
    elif args.graph_type=='caveman_small':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(20):
                    graphs.append(caveman_special(i, j, p_edge=0.8)) # default 0.8
        args.max_prev_node = 20
    elif args.graph_type=='caveman_small_single':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graphs.append(caveman_special(i, j, p_edge=0.5))
        args.max_prev_node = 20
    elif args.graph_type.startswith('community') or args.graph_type.startswith('comm'):
        # num_communities = int(args.graph_type[-1])
        # print('Creating dataset with ', num_communities, ' communities')
        # c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # #c_sizes = [15] * num_communities
        # for k in range(3000):
        #     graphs.append(n_community(c_sizes, p_inter=0.01))
        # args.max_prev_node = 80
        adjs = np.load("/home/steve.azzolin/DiGress_fork/data/comm/comm_train_val_test.npy", allow_pickle=True)
        c = 0
        graphs = []
        train_idxs, val_idxs, seltest_idxs = [], [], []
        for split in range(3):
            for adj in adjs[split]:
                adj = nx.convert_matrix.from_numpy_matrix(adj)
                graphs.append(adj)
                if split == 0:
                    train_idxs.append(c)
                elif split == 1:
                    val_idxs.append(c)
                elif split == 2:
                    test_idxs.append(c)
                c += 1
        args.max_prev_node = 80
    elif args.graph_type=='grid':
        # graphs = []
        # for i in range(10,20):
        #     for j in range(10,20):
        #         graphs.append(nx.grid_2d_graph(i,j))
        # args.max_prev_node = 40
        G = GridDataset()
        graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
        [g.remove_nodes_from(list(nx.isolates(g))) for g in graphs] # remove isolated nodes (in GRID we have all the ones  relative to padding)
    
        args.max_prev_node = 40
        train_idxs = G.train_idxs
        val_idxs = G.val_idxs
        test_idxs = G.test_idxs
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,5):
            for j in range(2,6):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 15
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 130
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 20
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    elif 'barabasi_noise' in args.graph_type:
        graphs = []
        for i in range(100,101):
            for j in range(4,5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        graphs = perturb_new(graphs,p=args.noise/10.0)
        args.max_prev_node = 99

    # real graphs
    elif args.graph_type == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    elif args.graph_type == 'enzymes_small':
        graphs_raw = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        graphs = []
        for G in graphs_raw:
            if G.number_of_nodes()<=20:
                graphs.append(G)
        args.max_prev_node = 15
    elif args.graph_type == 'protein':
        graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        args.max_prev_node = 80
    elif args.graph_type == 'DD':
        graphs = Graph_load_batch(min_num_nodes=100, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True)
        args.max_prev_node = 230
    elif args.graph_type == 'citeseer':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
        args.max_prev_node = 250
    elif args.graph_type == 'citeseer_small':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]
        args.max_prev_node = 15
    elif args.graph_type=='planar':
        G = PlanarDataset(n_nodes=64, n_graphs=200, k=2)
        graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
        args.max_prev_node = 30
        train_idxs = G.train_idxs
        val_idxs = G.val_idxs
        test_idxs = G.test_idxs
    elif args.graph_type=='sbm':
        G = SBMDataset(200, k=4, max_comm_size=100)
        graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
        args.max_prev_node = 32
        train_idxs = G.train_idxs
        val_idxs = G.val_idxs
        test_idxs = G.test_idxs
    elif args.graph_type=='ego':
        G = EgoDataset("small")
        graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
        args.max_prev_node = 17
        train_idxs = G.train_idxs
        val_idxs = G.val_idxs
        test_idxs = G.test_idxs
    return graphs, train_idxs, val_idxs, test_idxs
