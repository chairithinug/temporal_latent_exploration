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

latent = np.load(f'latent/latent_l2_3.npy')
position_mesh = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_all.txt")).to(dist.device)
position_pivotal = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_pivotal_l2.txt")).to(dist.device)
gt = np.load('./latent/test_data.npy', allow_pickle=True)

print(latent.shape) # (11, 401, 2, 3)
print(gt.shape) # (11, 401, 1699, 3)

# interpolation

# short term
# grab two random nearby points 
a = 10
u = np.random.randint(-a, a)
t1 = np.random.choice(np.arange(a, 399 - a))
t2= t1 + u
if t1 > t2:
    t1, t2 = t2, t1
print(t1, t2)
z1 = latent[0][t1]
z2 = latent[0][t2]

z_delta = (z2 - z1) / (t2 - t1)
print(z_delta)

z_hat = np.zeros((t2 - t1, 2, 3))
for i in tqdm(range(t2 - t1)):
    z_hat[i] = z1 + z_delta * (i + 1)

# decoder
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

x_hats = np.zeros((t2 - t1, 1699, 3))
for i, z in enumerate(z_hat):
    x_hats[i] = trainer.model.decode(
                z, graph.edata["x"], graph, position_mesh, position_pivotal
    )

np.save(f'interpolation/reconstruction_{t1}_{t2}.npy', x_hats)
# 1-step rollout