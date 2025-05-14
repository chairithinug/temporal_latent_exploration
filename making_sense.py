# from typing import Any, Callable, Dict, Tuple, Union

# import dgl.function as fn
# import torch
# from dgl import DGLGraph
# from torch import Tensor

# from dataset import (
#     VortexSheddingRe300To1000Dataset,
# )

# from dgl.dataloading import GraphDataLoader

# @torch.jit.ignore()
# def agg_concat_dgl(
#     efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
# ) -> Tensor:

#     with graph.local_scope():
#         # populate features on graph edges
#         graph.edata["x"] = efeat

#         # aggregate edge features
#         if aggregation == "sum":
#             graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
#         elif aggregation == "mean":
#             graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
#         else:
#             raise RuntimeError("Not a valid aggregation!")

#         # concat dst-node & edge features
#         cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
#         return cat_feat

# dataset_test = VortexSheddingRe300To1000Dataset(
#             name="vortex_shedding_train", split="test"
#         )

# dataloader_test = GraphDataLoader(
#             dataset_test,
#             batch_size=1,
#             shuffle=False,
#             drop_last=False,
#             pin_memory=True,
#             use_ddp=False,
#             num_workers=1,
#         )
# for graph in dataloader_test:
#     y = agg_concat_dgl(graph.edata["x"], graph.ndata["x"], graph, 'sum')
#     print(graph.edata["x"].shape)
#     print(graph.ndata["x"].shape)
#     print(y.shape)
#     print(y[:2,:])
#     print(graph.ndata["x"][:2])
#     print(graph)
#     break



import numpy as np
import torch

from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from physicsnemo.datapipes.gnn.vortex_shedding_re300_1000_dataset import (
    VortexSheddingRe300To1000Dataset,
)

class LatentDataset(DGLDataset):
    def __init__(
        self,
        name="dataset",
        data_dir="dataset",
        split="train",
        sequence_len=401,
        verbose=False,
    ):
        super().__init__(
            name=name,
            verbose=verbose,
        )
        self.split = split
        self.sequence_len = sequence_len
        self.data_dir = data_dir

        self.z = torch.load("{}/latent_{}.pt".format(self.data_dir, self.split), map_location=torch.device('cpu'))
        self.get_re_number()

    def __len__(self):
        return len(self.z) // self.sequence_len

    def __getitem__(self, idx):
        return (
            self.z[idx * self.sequence_len : (idx + 1) * self.sequence_len],
            self.re[idx : (idx + 1)],
        )

    def get_re_number(self):
        """Get RE number"""
        ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().reshape([-1, 1])
        nuAll = 1 / ReAll
        listCatALL = []
        for i in range(3):
            re = ReAll ** (i + 1)
            nu = nuAll ** (i + 1)
            listCatALL.append(re / re.max())
            listCatALL.append(nu / nu.max())
        if self.split == "train":
            index = [i for i in range(101) if i % 2 == 0]
        else:
            index = [i for i in range(101) if i % 2 == 1]

        self.re = torch.cat(listCatALL, dim=1)[index, :]

dataset_train = LatentDataset(split="test")
dataloader = GraphDataLoader(
            dataset_train,
            batch_size=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

for lc in dataloader:
    print(lc)
    print(lc[0].shape, lc[1].shape)
    break

