
import os
import sys
import time

import numpy as np
import torch
import wandb as wb
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import (
    VortexSheddingRe300To1000Dataset,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced

from new_train import Mesh_ReducedTrainer

from dgl.data import DGLDataset

from physicsnemo.datapipes.gnn.utils import load_json, save_json
import dgl

class LinearInterpDataset(VortexSheddingRe300To1000Dataset):
    def __init__(
        self, name="dataset", data_dir="dataset", split="train", verbose=False
    ):
        super().__init__(
            name=name, 
            data_dir=data_dir, 
            split=split, 
            verbose=verbose
        )

    def __len__(self):
        return (self.sequence_len - 2) * self.sequence_num

    def __getitem__(self, idx):
        sidx = idx // (self.sequence_len - 2)
        tidx = idx % (self.sequence_len - 2)

        node_features = (self.solution_states[sidx, tidx] + self.solution_states[sidx, tidx + 2]) / 2
        node_targets = self.solution_states[sidx, tidx+1]
        graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.edata["x"] = self.E
        return graph

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def test(graph, criterion=torch.nn.MSELoss(), node_stats=None):
    x = VortexSheddingRe300To1000Dataset.denormalize(graph.ndata["x"], node_stats["node_mean"], node_stats["node_std"])
    gt = VortexSheddingRe300To1000Dataset.denormalize(graph.ndata["y"], node_stats["node_mean"], node_stats["node_std"])
    loss = criterion(x, gt)
    relative_error = (
        loss / criterion(gt, gt * 0.0).detach()
    )
    relative_error_s_record = []
    for i in range(3):
        loss_s = criterion(x[:, i], gt[:, i])
        relative_error_s = (
            loss_s
            / criterion(
                gt[:, i], gt[:, i] * 0.0
            ).detach()
        )
        relative_error_s_record.append(relative_error_s)
    return loss, relative_error, relative_error_s_record

if __name__ == "__main__":
    ds = LinearInterpDataset(split='test')
    print(len(ds))
    loss_total = 0
    relative_error_total = 0
    node_stats = load_json("dataset/node_stats.json")
    for x in tqdm(ds):
        loss, relative_error, relative_error_s = test(x, node_stats=node_stats)
        loss_total = loss_total + loss
        relative_error_total = relative_error_total + relative_error
    n = len(ds)
    avg_relative_error = relative_error_total / n
    avg_loss = loss_total / n

    print(avg_loss,avg_relative_error, relative_error_s)

    # addr = os.getenv("MASTER_ADDR", "localhost")
    # port = os.getenv("MASTER_PORT", "12355")
    # DistributedManager._shared_state["_is_initialized"] = True
    # np.random.seed(seed=DistributedManager().rank)
    # dist = DistributedManager()


    # logger = PythonLogger("main")  # General python logger
    # rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    # logger.file_logging()

    # start = time.time()
    # rank_zero_logger.info("Testing started...")
    # rank_zero_logger.info(torch.cuda.is_available())
    # rank_zero_logger.info(dist.device)
    # position_mesh = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt")).to(dist.device)
    # position_pivotal = torch.from_numpy(np.loadtxt("dataset/meshPosition_pivotal_l256.txt")).to(dist.device)

    # config = AttrDict({
    #         'ckpt_path': "checkpoints/new_encoding",
    #         'ckpt_name': "model_l256.pt",
    #         'batch_size': 8,
    #         'epochs': 300,
    #         'lr':  0.00001,
    #         'lr_decay_rate': 0.9999991,
    #         'jit': False,
    #         'amp': True,
    #         'watch_model': False,
    #         'num_input_features': 3,
    #         'num_edge_features': 3,
    #         'num_output_features': 3,
    #         'output_encode_dim': 3,
    #         'processor_size': 15,

    #         'num_layers_node_processor': 2,
    #         'num_layers_edge_processor': 2,
    #         'hidden_dim_processor': 128,
    #         'hidden_dim_node_encoder': 128,
    #         'num_layers_node_encoder': 2,
    #         'hidden_dim_edge_encoder': 128,
    #         'num_layers_edge_encoder': 2,
    #         'hidden_dim_node_decoder': 128,
    #         'num_layers_node_decoder': 2,
    #         'k': 3,
    #     })
    # trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger, config)

    # for graph in tqdm(trainer.dataloader_test, disable=False):
    #     loss, relative_error, relative_error_s = trainer.test(graph, position_mesh, position_pivotal)
    #     loss_total = loss_total + loss
    #     relative_error_total = relative_error_total + relative_error
    # n = len(trainer.dataloader_test)
    # avg_relative_error = relative_error_total / n
    # avg_loss = loss_total / n
    # rank_zero_logger.info(
    #     f"avg_loss: {avg_loss:10.3e}, avg_relative_error: {avg_relative_error:10.3e}, time per epoch: {(time.time()-start):10.3}, relative_error_s: {relative_error_s}"
    # )
    # rank_zero_logger.info("Testing completed!")
    # print(config)
    # print(avg_loss,avg_relative_error, relative_error_s)