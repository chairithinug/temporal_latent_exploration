import os

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from physicsnemo.datapipes.gnn.utils import load_json, save_json

class VortexSheddingRe300To1000Dataset(DGLDataset):
    """In-memory Mesh-Reduced-Transformer Dataset for stationary mesh.
    Notes:
        - A single adj matrix is used for each transient simulation.
            Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "val", "test"], by default "train"
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self, name="dataset", data_dir="dataset", split="train", verbose=False
    ):

        super().__init__(
            name=name,
            verbose=verbose,
        )
        self.data_dir = data_dir

        self.split = split
        self.rawData = np.load(
            os.path.join(self.data_dir, "rawData.npy"), allow_pickle=True
        )

        # select training and testing set
        if self.split == "train":
            self.sequence_ids = [i for i in range(80)]
        if self.split == "val":
            self.sequence_ids = [i for i in range(80, 90)]
        if self.split == "test":
            self.sequence_ids = [i for i in range(90, 101)]

        # solution states are velocity and pressure
        self.solution_states = torch.from_numpy(
            self.rawData["x"][self.sequence_ids, :, :, :]
        ).float()

        # edge information
        self.E = torch.from_numpy(self.rawData["edge_attr"]).float()

        # edge connection
        self.A = torch.from_numpy(self.rawData["edge_index"]).type(torch.long)

        # sequence length
        self.sequence_len = self.solution_states.shape[1]
        self.sequence_num = self.solution_states.shape[0]
        self.num_nodes = self.solution_states.shape[2]

        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("dataset/edge_stats.json")

        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("dataset/node_stats.json")

        # handle the normalization
        for i in range(self.sequence_num):
            for j in range(self.sequence_len):
                self.solution_states[i, j] = self.normalize(
                    self.solution_states[i, j],
                    self.node_stats["node_mean"],
                    self.node_stats["node_std"],
                )
        self.E = self.normalize(
            self.E, self.edge_stats["edge_mean"], self.edge_stats["edge_std"]
        )

    def __len__(self):
        return self.sequence_len * self.sequence_num

    def __getitem__(self, idx):
        sidx = idx // self.sequence_len
        tidx = idx % self.sequence_len

        node_features = self.solution_states[sidx, tidx]
        node_targets = self.solution_states[sidx, tidx]
        graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.edata["x"] = self.E
        return graph

    def _get_edge_stats(self):
        stats = {
            "edge_mean": self.E.mean(dim=0),
            "edge_std": self.E.std(dim=0),
        }
        save_json(stats, "dataset/edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "node_mean": self.solution_states.mean(dim=[0, 1, 2]),
            "node_std": self.solution_states.std(dim=[0, 1, 2]),
        }
        save_json(stats, "dataset/node_stats.json")
        return stats

    @staticmethod
    def normalize(invar, mu, std):
        """normalizes a tensor"""
        if invar.size()[-1] != mu.size()[-1] or invar.size()[-1] != std.size()[-1]:
            raise ValueError(
                "invar, mu, and std must have the same size in the last dimension"
            )
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

class TimeVortexSheddingRe300To1000Dataset(VortexSheddingRe300To1000Dataset):
    def __init__(
        self, name="dataset", data_dir="dataset", split="train", verbose=False
    ):

        super().__init__(
            name=name,
            data_dir=data_dir,
            split=split,
            verbose=verbose,
        )
        self.base_graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
        self.base_graph.edata["x"] = self.E

    def __getitem__(self, idx):
        sidx = idx // self.sequence_len # traj 90
        tidx = idx % self.sequence_len
        
        node_features = self.solution_states[sidx, tidx]
        graph = self.base_graph.clone()
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_features
        return graph

class TemporalAwareDataset(VortexSheddingRe300To1000Dataset):
    def __init__(
        self, name="dataset", data_dir="dataset", split="train", verbose=False
    ):

        super().__init__(
            name=name,
            data_dir=data_dir,
            split=split,
            verbose=verbose,
        )
        self.base_graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
        self.base_graph.edata["x"] = self.E
        self.valid_indices = np.arange(0, 401) 

    def __len__(self): #
        return (self.sequence_len - 2) * self.sequence_num

    def __getitem__(self, idx):
        sidx = idx // (self.sequence_len - 2)
        tidx = idx % (self.sequence_len - 2)
        choices = np.delete(self.valid_indices, tidx)
        pos_idx, neg_idx = np.random.choice(choices, size=2, replace=False)
        if abs(pos_idx - tidx) > abs(neg_idx - tidx): # equally distance is ignored for now.
            pos_idx, neg_idx = neg_idx, pos_idx
        graphs = []
        for i in [tidx, pos_idx, neg_idx]:
            node_features = self.solution_states[sidx, i]
            graph = self.base_graph.clone()
            graph.ndata["x"] = node_features
            graph.ndata["y"] = node_features
            graphs.append(graph)
        return tuple(graphs) # anchor, pos, neg
    

class LatentEvolutionDataset(VortexSheddingRe300To1000Dataset):
     def __init__(
        self, name="dataset", data_dir="dataset", split="test", verbose=False
    ):
        self.data_dir = data_dir

        self.split = split
        self.rawData = np.load(
            os.path.join(self.data_dir, "rawData.npy"), allow_pickle=True
        )
        if self.split == "test":
            self.sequence_ids = [i for i in range(90, 91)]

        # solution states are velocity and pressure
        self.solution_states = torch.from_numpy(
            self.rawData["x"][self.sequence_ids, :, :, :]
        ).float()

        # edge information
        self.E = torch.from_numpy(self.rawData["edge_attr"]).float()

        # edge connection
        self.A = torch.from_numpy(self.rawData["edge_index"]).type(torch.long)

        # sequence length
        self.sequence_len = self.solution_states.shape[1]
        self.sequence_num = self.solution_states.shape[0]
        self.num_nodes = self.solution_states.shape[2]

        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("dataset/edge_stats.json")

        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("dataset/node_stats.json")

        # handle the normalization
        for i in range(self.sequence_num):
            for j in range(self.sequence_len):
                self.solution_states[i, j] = self.normalize(
                    self.solution_states[i, j],
                    self.node_stats["node_mean"],
                    self.node_stats["node_std"],
                )
        self.E = self.normalize(
            self.E, self.edge_stats["edge_mean"], self.edge_stats["edge_std"]
        )