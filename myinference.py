# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np
import torch
import wandb as wb
from tqdm import tqdm

# from constants import Constants
from dgl.dataloading import GraphDataLoader
from dataset import (
    VortexSheddingRe300To1000Dataset,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from new_train import Mesh_ReducedTrainer

from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced

import os

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

    for dim in [2]:
    #for dim in [2]:
        config = AttrDict({
            #'ckpt_path': "checkpoints/test_embedding_evo_cosine",
            'ckpt_path': "checkpoints/best",
            'ckpt_name': f"model_l{dim}_best.pt",
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
            'hidden_dim_processor': 128,
            'hidden_dim_node_encoder': 128,
            'num_layers_node_encoder': 2,
            'hidden_dim_edge_encoder': 128,
            'num_layers_edge_encoder': 2,
            'hidden_dim_node_decoder': 128,
            'num_layers_node_decoder': 2,
            'k': 3,
        })
        trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger, config)
        rank_zero_logger.info("Testing started...")
        position_mesh = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_all.txt")).to(dist.device)
        position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l{dim}.txt")).to(dist.device)
        sequence_len = 401
        trainer.epoch_init = load_checkpoint(
            os.path.join(config.ckpt_path, config.ckpt_name),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            device=dist.device,
        )
        predicts = np.zeros((11, sequence_len, len(position_mesh), 3))
        latents = np.zeros((11, sequence_len, dim, 3))
        # Batch size must be 1
        for j, graph in tqdm(enumerate(trainer.dataloader_test)):
            sidx = j // sequence_len
            tidx = j % sequence_len
            x, z = trainer.predict(
                graph, position_mesh, position_pivotal
            )
            predicts[sidx, tidx] = x.cpu()
            latents[sidx, tidx] = z.cpu()
        np.save(f'predict/predict_l{dim}best.npy', predicts)
        np.save(f'latent/latent_l{dim}best.npy', latents)
        rank_zero_logger.info("Inference completed!")
