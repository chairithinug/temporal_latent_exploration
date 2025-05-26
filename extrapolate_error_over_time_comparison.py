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

gt = np.load('./latent/test_data.npy', allow_pickle=True) # (11, 401, 1699, 3)

idx = np.lib.stride_tricks.sliding_window_view(np.arange(len(gt[0])),window_shape=3)
x_t2 = torch.from_numpy(gt[:, idx[:,2]]).cpu() # 11, 399, 1699, 3
 
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

graph = graph.to(trainer.dist.device)

def evaluate_predictions(x_hats, x_t2, name, trainer):
    n = len(x_hats[0])
    avg_losses = []
    avg_relative_errors = []
    avg_relative_error_ss = []
    with torch.no_grad():
        with autocast(enabled=trainer.C.amp):
            for i in range(n):  # timestep
                loss_total = 0
                relative_error_total = 0
                relative_error_s_total = []
                for j in range(len(x_hats)):  # trajectory
                    loss, relative_error, relative_error_s = test(x_hats[j, i], x_t2[j, i])
                    relative_error_s = [x.cpu() for x in relative_error_s]
                    relative_error_s_total.append(relative_error_s)
                    loss_total += loss
                    relative_error_total += relative_error
                avg_loss = loss_total / n
                avg_relative_error = relative_error_total / n
                avg_relative_error_s = np.array(relative_error_s_total).mean(axis=0)

                avg_losses.append(avg_loss.cpu())
                avg_relative_errors.append(avg_relative_error.cpu())
                avg_relative_error_ss.append(avg_relative_error_s)

    avg_losses = np.array(avg_losses)
    avg_relative_errors = np.array(avg_relative_errors)
    avg_relative_error_ss = np.array(avg_relative_error_ss)
    times = np.arange(2, len(avg_losses) + 2)

    # Save as JSON
    errs = {
        'avg_loss': avg_losses.tolist(),
        'avg_rel_err': avg_relative_errors.tolist(),
        'avg_rel_x': avg_relative_error_ss[:, 0].tolist(),
        'avg_rel_y': avg_relative_error_ss[:, 1].tolist(),
        'avg_rel_p': avg_relative_error_ss[:, 2].tolist(),
        'timesteps': times.tolist()
    }
    import json
    with open(f'{name}.json', 'w') as f:
        json.dump(errs, f, indent=4)

    return times, avg_losses, avg_relative_errors, avg_relative_error_ss

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_errors(all_results, title_prefix='Comparison'):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs = axs.flatten()

    mpl.rcParams.update({'font.size': 12})
    titles = ['Total Relative Error', 'Relative Error (x)', 'Relative Error (y)', 'Relative Error (p)']
    key_map = ['avg_relative_errors', 'avg_rel_x', 'avg_rel_y', 'avg_rel_p']
    colors = plt.get_cmap('tab10')

    for i, (ax, key, title) in enumerate(zip(axs, key_map, titles)):
        for j, (name, (times, _, avg_relative_errors, avg_relative_error_ss)) in enumerate(all_results.items()):
            if key == 'avg_relative_errors':
                data = avg_relative_errors
            elif key == 'avg_rel_x':
                data = avg_relative_error_ss[:, 0]
            elif key == 'avg_rel_y':
                data = avg_relative_error_ss[:, 1]
            elif key == 'avg_rel_p':
                data = avg_relative_error_ss[:, 2]

            ax.semilogy(times, data, label=name, color=colors(j), alpha=0.85)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Log error')
        ax.grid(True, linestyle='--', linewidth=0.5)
            #ax.label_outer()  # Only show outer labels

        # Unified legend outside the subplots
        ax.legend(
            title='Method'
        )

    fig.suptitle(f'{title_prefix}: Average Log-Scaled Error Components Over Time\nAcross All Test Trajectories', fontsize=16)
    plt.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.08, hspace=0.3, wspace=0.2)
    plt.savefig(f'{title_prefix.lower().replace(" ", "_")}_subplots_clean.png', dpi=300)
    plt.show()

x_hat_paths = {
    'triplet': 'extrapolation/reconstruction_triplet.npy',
    'data': 'extrapolation/reconstruction_data.npy',
    'pca_per_attr': 'extrapolation/reconstruction_pca_per_attr.npy',
    '2d': 'extrapolation/reconstruction_2d.npy',
}

all_results = {}
for label, path in x_hat_paths.items():
    x_hat_data = torch.from_numpy(np.load(path)).cpu()
    results = evaluate_predictions(x_hat_data, x_t2, f'extra_err_{label}', trainer)
    all_results[label] = results

plot_errors(all_results, title_prefix="Extrapolation Error Comparison")