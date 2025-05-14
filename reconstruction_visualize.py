from matplotlib.tri import Triangulation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

@torch.no_grad()
def plot_mesh(velocity, mesh_pos, faces, ax=None, fig=None, title='Mesh'):
    """
    Plots two graphs with each other.
    Can be used to plot the predicted graph and the ground truth
    """
    #fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.cla()
    ax.set_aspect("equal")
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    ax.autoscale(enable=True, axis='both', tight=True)
    #ax.set_axis_off()
    
    print(mesh_pos.shape, faces.shape, velocity.shape)

    triang = mtri.Triangulation(mesh_pos[:, 0].cpu(), mesh_pos[:, 1].cpu(), faces.cpu())
    mesh_plot = ax.tripcolor(
        triang, velocity.cpu(), vmin=velocity.min(), vmax=velocity.max(), shading="flat"
    )
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

    ax.set_title(title, fontsize="40")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
    clb.ax.tick_params(labelsize=40)
    clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 40})
    return fig

def triple_plot_mesh():
    gt = torch.from_numpy(raw['x'][traj,time,:,0])
    denom = denormalize(torch.from_numpy(predict[traj-90, time,:]), node_stats["node_mean"], node_stats["node_std"])

    fig = plot_mesh(gt, pos, faces)
    #fig.savefig(f'visualize_traj_{traj}_t{time}.png', bbox_inches='tight')

    fig = plot_mesh(gt - denom[...,0], pos, faces)
    #fig.savefig(f'visualize_diff_pred{dim}_traj_{traj}_t{time}.png', bbox_inches='tight')

    fig = plot_mesh(denom[...,0], pos, faces)
    #fig.savefig(f'visualize_pred{dim}_traj_{traj}_t{time}.png', bbox_inches='tight')

from physicsnemo.datapipes.gnn.utils import load_json, save_json

node_stats = load_json("dataset/node_stats.json")


def denormalize(invar, mu, std):
    """denormalizes a tensor"""
    denormalized_invar = invar * std.expand(invar.size()) + mu.expand(invar.size())
    return denormalized_invar

raw = np.load('./dataset/rawData.npy', allow_pickle=True)
pos = torch.from_numpy(np.loadtxt(f"dataset/meshPosition_all.txt"))
faces = torch.from_numpy(np.load('mat_delaunay_filtered.npy'))
traj = 90

for dim in [2]:
    predict = np.load(f'predict/predict_l{dim}allbest.npy')
    for time in [0, 50, 100, 200, 400]:
        gt = torch.from_numpy(raw['x'][traj,time,:,0])
        denom = denormalize(torch.from_numpy(predict[traj-90, time,:]), node_stats["node_mean"], node_stats["node_std"])

        # fig = plot_mesh(gt, pos, faces)
        # fig.savefig(f'visualize_traj_{traj}_t{time}.png', bbox_inches='tight')

        # fig = plot_mesh(gt - denom[...,0], pos, faces)
        # fig.savefig(f'visualize_diff_pred{dim}_traj_{traj}_t{time}.png', bbox_inches='tight')

        # fig = plot_mesh(denom[...,0], pos, faces)
        # fig.savefig(f'visualize_pred{dim}_traj_{traj}_t{time}.png', bbox_inches='tight')
        fig, axes = plt.subplots(3, 1, figsize=(20, 30))
        plot_mesh(gt, pos, faces, ax=axes[0], fig=fig, title='Ground Truth')
        plot_mesh(denom[..., 0], pos, faces, ax=axes[1], fig=fig, title='Prediction')
        plot_mesh(gt - denom[..., 0], pos, faces, ax=axes[2], fig=fig, title='GT - Prediction')

        plt.tight_layout()
        fig.savefig(f'combined_plot_traj_{traj}_t{time}_dim{dim}allbest.png', bbox_inches='tight')