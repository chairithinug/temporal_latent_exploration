import numpy as np
import matplotlib.pyplot as plt
from new_train import Mesh_ReducedTrainer
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
import torch
import os
import wandb as wb
from physicsnemo.launch.utils import load_checkpoint
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

addr = os.getenv("MASTER_ADDR", "localhost")
port = os.getenv("MASTER_PORT", "12355")
DistributedManager._shared_state["_is_initialized"] = True
np.random.seed(seed=DistributedManager().rank)
dist = DistributedManager()
logger = PythonLogger("main")  # General python logger
rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

position_mesh = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_all.txt")).to(dist.device)
position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l2.txt")).to(dist.device)
gt = np.load('./latent/test_data.npy', allow_pickle=True)

print(gt.shape) # (11, 401, 1699, 3)

raw = np.load('./dataset/rawData.npy', allow_pickle=True)
latent_train = raw['x'][:80] # shape = 80, 401, 1699, 3
latent_test = raw['x'][90:] # shape = 11, 401, 1699, 3

sc_x = StandardScaler()
sc_y = StandardScaler()
sc_p  = StandardScaler()
reducer_x = PCA(n_components=2)
reducer_y = PCA(n_components=2)
reducer_p = PCA(n_components=2)

latent_train_x = sc_x.fit_transform(latent_train[...,0].reshape(-1, 1699))
reducer_x.fit(latent_train_x)
latent_test_x = sc_x.transform(latent_test[...,0].reshape(-1, 1699))
latent_test_x = reducer_x.transform(latent_test_x)
latent_test_x = latent_test_x.reshape(11, 401, 2, 1)

latent_train_y = sc_y.fit_transform(latent_train[...,1].reshape(-1, 1699))
reducer_y.fit(latent_train_y)
latent_test_y = sc_y.transform(latent_test[...,1].reshape(-1, 1699))
latent_test_y = reducer_y.transform(latent_test_y)
latent_test_y = latent_test_y.reshape(11, 401, 2, 1)

latent_train_p = sc_p.fit_transform(latent_train[...,2].reshape(-1, 1699))
reducer_p.fit(latent_train_p)
latent_test_p = sc_p.transform(latent_test[...,2].reshape(-1, 1699))
latent_test_p = reducer_p.transform(latent_test_p)
latent_test_p = latent_test_p.reshape(11, 401, 2, 1)

latent = np.concatenate([latent_test_x, latent_test_y, latent_test_p], axis=-1)
print(latent.shape)


config = AttrDict({
            'ckpt_path': "checkpoints/test_embedding_evo_all",
            'ckpt_name': f"model_l2_best.pt",
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

idx = np.lib.stride_tricks.sliding_window_view(np.arange(len(gt[0])),window_shape=3)

x_t2 = torch.from_numpy(gt[:, idx[:,2]]).cpu() # 11, 399, 1699, 3
print(x_t2.shape)
zt = latent[:,idx[:,0]]
zt1 = latent[:,idx[:,1]]
z_hat_t2 = torch.from_numpy((2 * zt1 - zt)).cpu() # 11, 399, 2, 3
    
# decoder

trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger, config)
trainer.epoch_init = load_checkpoint(
            os.path.join(config.ckpt_path, config.ckpt_name),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            device=dist.device,
        )

for graph in trainer.dataloader_test:
    break

def test(x, y, criterion=torch.nn.MSELoss()):
    loss = criterion(x, y)
    relative_error = (
        loss / criterion(y, y * 0.0).detach()
    )
    relative_error_s_record = []
    for i in range(3):
        loss_s = criterion(x[:, i], y[:, i])
        relative_error_s = (
            loss_s
            / criterion(
                y[:, i], y[:, i] * 0.0
            ).detach()
        )
        relative_error_s_record.append(relative_error_s)
    return loss, relative_error, relative_error_s_record

from physicsnemo.datapipes.gnn.utils import load_json
node_stats = load_json("dataset/node_stats.json")

def denormalize(invar, mu, std):
    """denormalizes a tensor"""
    denormalized_invar = invar * std.expand(invar.size()) + mu.expand(invar.size())
    return denormalized_invar

x_hats = torch.zeros_like(x_t2, device='cpu') # 11, 399, 1699, 3
graph = graph.to(trainer.dist.device)
loss_total = 0
relative_error_total = 0
relative_error_s_total = []

print(z_hat_t2[...,0].shape) # 11, 399, 2
zx = reducer_x.inverse_transform(z_hat_t2[...,0])
print(zx.shape)
zx = zx.reshape(-1, 1699)
zx = sc_x.inverse_transform(zx)
zx = zx.reshape(11, 399, 1699, 1)

zy = reducer_y.inverse_transform(z_hat_t2[...,1])
zy = zy.reshape(-1, 1699)
zy = sc_y.inverse_transform(zy)
zy = zy.reshape(11, 399, 1699, 1)

zp = reducer_p.inverse_transform(z_hat_t2[...,2])
zp = zp.reshape(-1, 1699)
zp = sc_p.inverse_transform(zp)
zp = zp.reshape(11, 399, 1699, 1)

x_hats = np.concatenate([zx, zy, zp], axis=-1)
print(x_hats.shape)
x_hats = torch.from_numpy(x_hats)

with torch.no_grad():
    with autocast(enabled=trainer.C.amp):
        for i, x_traj in enumerate(x_hats):
            for j, z in enumerate(x_traj):
                x_hats[i,j] = denormalize(x_hats[i,j], node_stats["node_mean"], node_stats["node_std"])
                loss, relative_error, relative_error_s = test(x_hats[i,j], x_t2[i,j])
                relative_error_s = [x.cpu() for x in relative_error_s]
                relative_error_s_total.append(relative_error_s)
                loss_total = loss_total + loss
                relative_error_total = relative_error_total + relative_error
n = 11 * 399
avg_relative_error = relative_error_total / n
avg_loss = loss_total / n

avg_relative_error_s = np.array(relative_error_s_total).mean(axis=0)
print(avg_loss,avg_relative_error, avg_relative_error_s)
np.save(f'extrapolation/reconstruction_pca_per_attr.npy', x_hats.detach().cpu().numpy())
# 1-step rollout