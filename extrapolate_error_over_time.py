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
from physicsnemo.datapipes.gnn.utils import load_json

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

node_stats = load_json("dataset/node_stats.json")

def denormalize(invar, mu, std):
    """denormalizes a tensor"""
    denormalized_invar = invar * std.expand(invar.size()) + mu.expand(invar.size())
    return denormalized_invar

config = AttrDict({
            'ckpt_path': "checkpoints/test_embedding_evo_triplet",
            #'ckpt_path': "checkpoints/best",
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

addr = os.getenv("MASTER_ADDR", "localhost")
port = os.getenv("MASTER_PORT", "12355")
DistributedManager._shared_state["_is_initialized"] = True
np.random.seed(seed=DistributedManager().rank)
dist = DistributedManager()
logger = PythonLogger("main")  # General python logger
rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

latent = np.load(f'latent/latent_l2tripletbest.npy')
#latent = np.load(f'latent/latent_l2best.npy') # (11, 401, 2, 3)
position_mesh = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_all.txt")).to(dist.device)
position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l2.txt")).to(dist.device)
gt = np.load('./latent/test_data.npy', allow_pickle=True) # (11, 401, 1699, 3)

idx = np.lib.stride_tricks.sliding_window_view(np.arange(len(gt[0])),window_shape=3)
x_t2 = torch.from_numpy(gt[:, idx[:,2]]).cpu() # 11, 399, 1699, 3
zt, zt1 = latent[:,idx[:,0]], latent[:,idx[:,1]]
z_hat_t2 = torch.from_numpy((2 * zt1 - zt)).cpu() # 11, 399, 2, 3
 
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

#x_hats = torch.zeros_like(x_t2, device='cpu') # 11, 399, 1699, 3
x_hats = torch.from_numpy(np.load('extrapolation/reconstruction_triplet.npy')).cpu() # 11, 399, 1699, 3
graph = graph.to(trainer.dist.device)
loss_total = 0
relative_error_total = 0
relative_error_s_total = []
n = len(x_hats[0])
avg_losses = []
avg_relative_errors = []
avg_relative_error_ss = []
name = 'extra_err_ot_90'
#name = 'extra_err_ot'
with torch.no_grad():
    with autocast(enabled=trainer.C.amp):
        for i in range(n): # timestep
            loss_total = 0
            relative_error_total = 0
            relative_error_s_total = []
            for j in range(len(x_hats) if name =='extra_err_ot' else 1): # traj
                loss, relative_error, relative_error_s = test(x_hats[j,i], x_t2[j,i])
                relative_error_s = [x.cpu() for x in relative_error_s]
                relative_error_s_total.append(relative_error_s)
                loss_total = loss_total + loss
                relative_error_total = relative_error_total + relative_error
            avg_relative_error = relative_error_total / n
            avg_loss = loss_total / n
            avg_relative_error_s = np.array(relative_error_s_total).mean(axis=0)

            avg_losses.append(avg_loss.cpu())
            avg_relative_errors.append(avg_relative_error.cpu())
            avg_relative_error_ss.append(avg_relative_error_s)

            print(avg_loss,avg_relative_error, avg_relative_error_s)
avg_losses = np.array(avg_losses)
avg_relative_errors = np.array(avg_relative_errors)
avg_relative_error_ss = np.array(avg_relative_error_ss)

times = np.arange(2,len(avg_losses) + 2)
plt.figure(figsize=(15,8))
plt.semilogy(times, avg_losses,label='avg_loss', alpha=0.7, marker='o')
plt.semilogy(times,avg_relative_errors,label='avg_rel_err', alpha=0.7)
plt.semilogy(times,avg_relative_error_ss[:,0],label='avg_rel_x', alpha=0.7)
plt.semilogy(times,avg_relative_error_ss[:,1],label='avg_rel_y', alpha=0.7)
plt.semilogy(times,avg_relative_error_ss[:,2],label='avg_rel_p', alpha=0.7)
plt.legend()
plt.grid()
plt.xlabel('Timesteps')
plt.ylabel('Log error')
if name == 'extra_err_ot_90':
    plt.title('Log errors of extrapolated data over timesteps\nTrajectory 90 (Test)')
else:
    plt.title('Average log errors of extrapolated data over timesteps\nacross all Test trajectories')
plt.savefig(f'{name}.png')
errs = {}
errs['avg_loss'] = avg_losses.tolist()
errs['avg_rel_err'] = avg_relative_errors.tolist()
errs['avg_rel_x'] = avg_relative_error_ss[:,0].tolist()
errs['avg_rel_y'] = avg_relative_error_ss[:,1].tolist()
errs['avg_rel_p'] = avg_relative_error_ss[:,2].tolist()
errs['timesteps'] = times.tolist()
import json
with open(f'{name}.json', 'w') as f:
    json.dump(errs, f, indent=4)
plt.tight_layout()
plt.show()