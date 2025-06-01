
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

from train import Mesh_ReducedTrainer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == "__main__":
    addr = os.getenv("MASTER_ADDR", "localhost")
    port = os.getenv("MASTER_PORT", "12355")
    DistributedManager._shared_state["_is_initialized"] = True
    np.random.seed(seed=DistributedManager().rank)
    dist = DistributedManager()


    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()
    dim = 2
    start = time.time()
    rank_zero_logger.info("Testing started...")
    rank_zero_logger.info(torch.cuda.is_available())
    rank_zero_logger.info(dist.device)
    position_mesh = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt")).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l{dim}.txt")).to(dist.device)


    config = AttrDict({
            'ckpt_path': "checkpoints/best",
            'ckpt_name': f"model_l2_256hidden_best.pt",
            'batch_size': 1,
            'epochs': 300,
            'lr':  0.00001,
            'lr_decay_rate': 0.9999991,
            'jit': False,
            'amp': True,
            'watch_model': False,
            'num_input_features': 3,
            'num_edge_features': 3,
            'num_output_features': 3,
            'output_encode_dim': 3,
            'processor_size': 15,

            'num_layers_node_processor': 2,
            'num_layers_edge_processor': 2,
            'hidden_dim_processor': 256,
            'hidden_dim_node_encoder': 256,
            'num_layers_node_encoder': 2,
            'hidden_dim_edge_encoder': 256,
            'num_layers_edge_encoder': 2,
            'hidden_dim_node_decoder': 256,
            'num_layers_node_decoder': 2,
            'k': 3,
        })
    trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger, config)

    trainer.epoch_init = load_checkpoint(
        os.path.join(config.ckpt_path, config.ckpt_name),
        models=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        device=dist.device,
    )

    loss_total = 0
    relative_error_total = 0
    relative_error_s_total = []
    for graph in tqdm(trainer.dataloader_test, disable=False):
        loss, relative_error, relative_error_s = trainer.test(graph, position_mesh, position_pivotal)
        relative_error_s = [x.cpu() for x in relative_error_s]
        relative_error_s_total.append(relative_error_s)
        loss_total = loss_total + loss
        relative_error_total = relative_error_total + relative_error
    n = len(trainer.dataloader_test)
    print(n)
    avg_relative_error = relative_error_total / n
    avg_loss = loss_total / n
    avg_relative_error_s = np.array(relative_error_s_total).mean(axis=0)

    rank_zero_logger.info(
        f"avg_loss: {avg_loss:10.3e}, avg_relative_error: {avg_relative_error:10.3e}, time per epoch: {(time.time()-start):10.3}, relative_error_s: {relative_error_s}"
    )
    rank_zero_logger.info("Testing completed!")
    print(config)
    print(avg_loss,avg_relative_error, avg_relative_error_s)
