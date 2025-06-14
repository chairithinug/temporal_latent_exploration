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

class Mesh_ReducedTrainer:
    def __init__(self, wb, dist, rank_zero_logger, C):
        self.dist = dist
        dataset_train = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="train"
        )

        dataset_val = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="val"
        )

        dataset_test = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="test"
        )

        self.C = C

        self.dataloader = GraphDataLoader(
            dataset_train,
            batch_size=self.C.batch_size,
            # shuffle=True, # train
            # drop_last=True, # train
            shuffle=False, # inference_all
            drop_last=False, # inference_all
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=8,
        )

        self.dataloader_val = GraphDataLoader(
            dataset_val,
            batch_size=self.C.batch_size,
            # shuffle=False, # train
            # drop_last=True, # train
            shuffle=False, # inference_all
            drop_last=False, # inference_all
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=8,
        )

        self.dataloader_test = GraphDataLoader(
            dataset_test,
            batch_size=self.C.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=1,
        )

        self.model = self.build_model()
        if self.C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)
        if self.C.watch_model and not self.C.jit and dist.rank == 0:
            wb.watch(self.model)
        self.model.train()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: self.C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(self.C.ckpt_path, self.C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def build_model(self):
        model = Mesh_Reduced(
            input_dim_nodes=self.C.num_input_features, 
            input_dim_edges=self.C.num_edge_features, 
            output_decode_dim=self.C.num_output_features,
            output_encode_dim=self.C.output_encode_dim,
            processor_size=self.C.processor_size,
            num_layers_node_processor=self.C.num_layers_node_processor,
            num_layers_edge_processor=self.C.num_layers_edge_processor,
            hidden_dim_processor=self.C.hidden_dim_processor,
            hidden_dim_node_encoder=self.C.hidden_dim_node_encoder,
            num_layers_node_encoder=self.C.num_layers_node_encoder,
            hidden_dim_edge_encoder=self.C.hidden_dim_edge_encoder,
            num_layers_edge_encoder=self.C.num_layers_edge_encoder,
            hidden_dim_node_decoder=self.C.hidden_dim_node_decoder,
            num_layers_node_decoder=self.C.num_layers_node_decoder,
            k=self.C.k
        )
        self.model = model
        if self.C.jit:
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        self.model.train()
        return model

    def forward(self, graph, position_mesh, position_pivotal):
        with autocast(enabled=self.C.amp):
            z = self.model.encode(
                graph.ndata["x"],
                graph.edata["x"],
                graph,
                position_mesh,
                position_pivotal,
            )
            x = self.model.decode(
                z, graph.edata["x"], graph, position_mesh, position_pivotal
            )
            loss = self.criterion(x, graph.ndata["x"])
            return loss

    def train(self, graph, position_mesh, position_pivotal):
        self.model.train()
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph, position_mesh, position_pivotal)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def validation(self, graph, position_mesh, position_pivotal):
        self.model.eval()
        graph = graph.to(self.dist.device)
        with torch.no_grad():
            loss = self.forward(graph, position_mesh, position_pivotal)
        return loss

    @torch.no_grad()
    def test(self, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device)
        with autocast(enabled=self.C.amp):
            z = self.model.encode(
                graph.ndata["x"],
                graph.edata["x"],
                graph,
                position_mesh,
                position_pivotal,
            )
            x = self.model.decode(
                z, graph.edata["x"], graph, position_mesh, position_pivotal
            )
            loss = self.criterion(x, graph.ndata["x"])

            relative_error = (
                loss / self.criterion(graph.ndata["x"], graph.ndata["x"] * 0.0).detach()
            )
            relative_error_s_record = []
            for i in range(self.C.num_input_features):
                loss_s = self.criterion(x[:, i], graph.ndata["x"][:, i])
                relative_error_s = (
                    loss_s
                    / self.criterion(
                        graph.ndata["x"][:, i], graph.ndata["x"][:, i] * 0.0
                    ).detach()
                )
                relative_error_s_record.append(relative_error_s)

        return loss, relative_error, relative_error_s_record
    
    @torch.no_grad()
    def predict(self, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device)
        with autocast(enabled=self.C.amp):
            z = self.model.encode(
                graph.ndata["x"],
                graph.edata["x"],
                graph,
                position_mesh,
                position_pivotal,
            )
            x = self.model.decode(
                z, graph.edata["x"], graph, position_mesh, position_pivotal
            )
        return x, z

    def backward(self, loss):
        # backward pass
        if self.C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

def search(config=None):
    start = time.time()
    # EarlyStopping
    last_val_loss = 1e9
    trig_cnt = 0
    patience = 10
    best_val_loss = 1e9
    with wb.init(config=config):
        config = wb.config
        rank_zero_logger.info(config)
        trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger, config)

        for epoch in range(trainer.epoch_init, config.epochs):
            total_loss = 0
            total_val_loss = 0

            for graph in tqdm(trainer.dataloader, disable=False):
                loss = trainer.train(graph, position_mesh, position_pivotal)
                total_loss += loss
            total_loss = total_loss / len(trainer.dataloader)
            
            for graph in tqdm(trainer.dataloader_val):
                val_loss = trainer.validation(graph, position_mesh, position_pivotal)
                total_val_loss += val_loss
            total_val_loss = total_val_loss / len(trainer.dataloader_val)

            if best_val_loss > total_val_loss:
                best_val_loss = total_val_loss
                save_checkpoint(
                            os.path.join(config.ckpt_path, config.ckpt_name_best),
                            models=trainer.model,
                            optimizer=trainer.optimizer,
                            scheduler=trainer.scheduler,
                            scaler=trainer.scaler,
                            epoch=epoch,
                        )
            
            if total_val_loss > last_val_loss:
                trig_cnt += 1
                if trig_cnt >= patience:
                    if dist.rank == 0:
                        save_checkpoint(
                            os.path.join(config.ckpt_path, config.ckpt_name),
                            models=trainer.model,
                            optimizer=trainer.optimizer,
                            scheduler=trainer.scheduler,
                            scaler=trainer.scaler,
                            epoch=epoch,
                        )
                        return
            else:
                trig_cnt = 0
            last_val_loss = total_val_loss

            wb.log({
                "val_loss": total_val_loss.detach().cpu(),
                "loss": total_loss.detach().cpu()
            })
            # save checkpoint
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0 and epoch % 10 == 0:
                save_checkpoint(
                    os.path.join(config.ckpt_path, config.ckpt_name),
                    models=trainer.model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    scaler=trainer.scaler,
                    epoch=epoch,
                )
if __name__ == "__main__":
    # initialize distributed manager
    # DistributedManager.initialize()
    addr = os.getenv("MASTER_ADDR", "localhost")
    port = os.getenv("MASTER_PORT", "12355")
    DistributedManager._shared_state["_is_initialized"] = True
    np.random.seed(seed=DistributedManager().rank)
    dist = DistributedManager()

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs("checkpoints/new_encoding", exist_ok=True)

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    start = time.time()
    rank_zero_logger.info("Training started...")

    rank_zero_logger.info(torch.cuda.is_available())
    rank_zero_logger.info(dist.device)
    latent = 2
    position_mesh = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt")).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l{latent}.txt")).to(dist.device)

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
            },
        'parameters': {
            'ckpt_path': {
                 'value' : "checkpoints/new_encoding"
            },
            'ckpt_name': {
                 'value' : f"model_l{latent}_3_256.pt"
            },
            'ckpt_name_best': {
                 'value' : f"model_l{latent}_3_256_best.pt"
            },
            'batch_size': {
                 'value' : 8
            },
            'epochs': {
                 'value' : 300
            },
            'lr': {
                 'value' : 0.00001
            },
            'lr_decay_rate': {
                 'value' : 0.9999991
            },
            'jit': {
                'value' : False
            },
            'amp': {
                'value' : True
            },
            'watch_model': {
                'value' : False
            },
            'num_input_features': {
                'value' : 3
            },
            'num_edge_features': {
                'value' : 3
            },
            'num_output_features': {
                'value' : 3
            },
            'output_encode_dim': {
                'value' : 3
            },
            'processor_size': {
                'value' : 15
            },
            'num_layers_node_processor': {
                'values' : [3]
            },
            'num_layers_edge_processor': {
                'values' : [3]
            },
            'hidden_dim_processor': {
                'values' : [256]
            },
            'hidden_dim_node_encoder': {
                'values' : [256]
            },
            'num_layers_node_encoder': {
                'values' : [3]
            },
            'hidden_dim_edge_encoder': {
                'values' : [256]
            },
            'num_layers_edge_encoder': {
                'values' : [3]
            },
            'hidden_dim_node_decoder': {
                'values' : [256]
            },
            'num_layers_node_decoder': {
                'values' : [3]
            },
             'k': {
                'value' : 3
            },
        }
    }

    sweep_id = wb.sweep(sweep_config, entity="cqk769", project="PhysicsNeMo-Launch-Sweep-New")
    wb.agent(sweep_id, search, count=1)