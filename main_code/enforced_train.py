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
import torch.nn.functional as F
import wandb as wb
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import dgl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from constants import Constants
from dataset import (
    VortexSheddingRe300To1000Dataset, TemporalAwareDataset, LatentEvolutionDataset
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced

class TripletBatch:
    def __init__(self, batch):
        anchors, positives, negatives = zip(*batch)
        self.batched_anchors = dgl.batch(anchors)
        self.batched_positives = dgl.batch(positives)
        self.batched_negatives = dgl.batch(negatives)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.batched_anchors = self.batched_anchors.pin_memory_()
        self.batched_positives = self.batched_positives.pin_memory_()
        self.batched_negatives = self.batched_negatives.pin_memory_()
        return self
    
    def __iter__(self):
        return iter((self.batched_anchors, self.batched_positives, self.batched_negatives))

class Mesh_ReducedTrainer:
    def __init__(self, wb, dist, rank_zero_logger, C):
        self.dist = dist
        dataset_train = TemporalAwareDataset(
            name="vortex_shedding_train", split="train"
        )

        dataset_val = TemporalAwareDataset(
            name="vortex_shedding_train", split="val"
        )

        dataset_test = LatentEvolutionDataset(
            name="vortex_shedding_train", split="test"
        )

        self.C = C

        def triplet_collate(batch):
            return TripletBatch(batch)

        self.dataloader = GraphDataLoader(
            dataset_train,
            batch_size=self.C.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=8,
            collate_fn=triplet_collate,
            prefetch_factor=2,
            persistent_workers=True,
        )

        self.dataloader_val = GraphDataLoader(
            dataset_val,
            batch_size=self.C.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=8,
            collate_fn=triplet_collate,
            prefetch_factor=2,
            persistent_workers=True,
        )

        self.dataloader_test = GraphDataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
            num_workers=8,
            persistent_workers=True,
            #collate_fn=triplet_collate,
        )

        self.model = self.build_model()
        if self.C.jit:
            self.model = torch.jit.script(self.model).to(dist.device, non_blocking=True)
        else:
            self.model = self.model.to(dist.device, non_blocking=True)
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
            os.path.join(self.C.ckpt_path, self.C.ckpt_name_best),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )
        if self.C.triplet_loss or self.C.cosine_loss:
            self.epoch_init = 0

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
            self.model = self.model.to(self.dist.device, non_blocking=True)
        self.model.train()
        return model

    def forward(self, graph, position_mesh, position_pivotal, return_z=False):
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
            loss = self.criterion(x, graph.ndata["x"]) # reconstruction loss
            if return_z:
                return loss, z
            return loss
        
    def forward_temporal(self, graphs, position_mesh, position_pivotal):
        total_loss = 0
        zs = []
        for graph in graphs:
            graph = graph.to(self.dist.device, non_blocking=True)
            loss, z = self.forward(graph, position_mesh, position_pivotal, return_z=True) # reconstruction
            zs.append(z.view(self.C.batch_size, -1))
            total_loss += loss
        
        if self.C.triplet_loss:
            total_loss += F.triplet_margin_loss(zs[0], zs[1], zs[2], margin=0.5) # triplet loss
        if self.C.cosine_loss:
            labels = torch.ones(len(zs[0])).to(self.dist.device, non_blocking=True)
            total_loss += F.cosine_embedding_loss(zs[0], zs[1], labels)
            total_loss += F.cosine_embedding_loss(zs[1], zs[2], labels)
            total_loss += F.cosine_embedding_loss(zs[0], zs[2], labels)
        return total_loss 
        
     
    def train(self, graphs, position_mesh, position_pivotal):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.forward_temporal(graphs, position_mesh, position_pivotal)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def validation(self, graphs, position_mesh, position_pivotal):
        self.model.eval()
        with torch.no_grad():
            loss = self.forward_temporal(graphs, position_mesh, position_pivotal)
        return loss

    @torch.no_grad()
    def test(self, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device, non_blocking=True)
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
        graph = graph.to(self.dist.device, non_blocking=True)
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
    
    @torch.no_grad()
    def decode(self, z, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device, non_blocking=True)
        with autocast(enabled=self.C.amp):
            x = self.model.decode(
                    z, graph.edata["x"], graph, position_mesh, position_pivotal
            )
        return x

    def backward(self, loss):
        # backward pass
        if self.C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def plot_latent(self, epoch):
        latents = np.zeros((1, 401, 2, 3))
        # Batch size must be 1
        for j, graph in tqdm(enumerate(self.dataloader_test)):
            _, z = self.predict(
                graph, position_mesh, position_pivotal
            )
            latents[0, j] = z.cpu()
        np.save(f'latent_evolution/{epoch}_{self.C.triplet_loss}_{self.C.cosine_loss}.npy', latents)
        for i, name in zip(range(3), ['x', 'y', 'p']):
            sc = StandardScaler()
            x_reduced = sc.fit_transform(latents[0,:,:,i])

            plt.figure(figsize=(12, 8))
            fig, ax = plt.subplots(1)
            plt.scatter(x_reduced[:,0], x_reduced[:,1], c=np.arange(len(x_reduced)), cmap="rainbow")
            plt.title(f'latent@dim=2 of {name} of Trajectory 90 (Test)')
            plt.ylabel('C_1')
            plt.xlabel('C_0')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = plt.Normalize(0, len(x_reduced))
            sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
            fig.colorbar(sm, cax=cax, orientation="vertical")
            plt.title('Timesteps')
            fname = f'latent_evolution/256_{epoch}_{self.C.triplet_loss}_{self.C.cosine_loss}_{name}.png'
            plt.savefig(fname, bbox_inches='tight')
            plt.close()

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

        trainer.plot_latent(-1)

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
                'best_val_loss': best_val_loss.detach().cpu(),
                "val_loss": total_val_loss.detach().cpu(),
                "loss": total_loss.detach().cpu()
            })

            # plot latent of traj 90
            trainer.plot_latent(epoch)
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

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    start = time.time()
    rank_zero_logger.info("Training started...")
    rank_zero_logger.info(torch.cuda.is_available())
    rank_zero_logger.info(dist.device)
    latent = 2
    position_mesh = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt")).to(dist.device, non_blocking=True)
    position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l{latent}.txt")).to(dist.device, non_blocking=True)

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
            },
        'parameters': {
            'ckpt_path': {
                 'value' : "checkpoints/v2_all"
            },
            'ckpt_name': {
                 'value' : f"model_l{latent}.pt"
            },
            'ckpt_name_best': {
                 'value' : f"model_l{latent}_best.pt"
            },
            'batch_size': {
                 'value' : 8
            },
            'epochs': {
                 'value' : 100
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
                'values' : [2]
            },
            'num_layers_edge_processor': {
                'values' : [2]
            },
            'hidden_dim_processor': {
                'values' : [256]
            },
            'hidden_dim_node_encoder': {
                'values' : [256]
            },
            'num_layers_node_encoder': {
                'values' : [2]
            },
            'hidden_dim_edge_encoder': {
                'values' : [256]
            },
            'num_layers_edge_encoder': {
                'values' : [2]
            },
            'hidden_dim_node_decoder': {
                'values' : [256]
            },
            'num_layers_node_decoder': {
                'values' : [2]
            },
             'k': {
                'value' : 3
            },
            'triplet_loss':{
                'value' : True
            },
            'cosine_loss':{
                'value' : True
            },
        }
    }

    sweep_id = wb.sweep(sweep_config, entity="cqk769", project="PhysicsNeMo-Launch-Sweep-New")
    wb.agent(sweep_id, search, count=1)