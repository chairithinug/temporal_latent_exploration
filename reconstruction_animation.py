import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from physicsnemo.datapipes.gnn.utils import load_json

@torch.no_grad()
def plot_mesh(velocity, mesh_pos, faces, vmin=None, vmax=None, ax=None, title='Mesh'):
    ax.cla()
    ax.set_aspect("equal")
    ax.autoscale(enable=True, axis='both', tight=True)

    triang = mtri.Triangulation(mesh_pos[:, 0].cpu(), mesh_pos[:, 1].cpu(), faces.cpu())
    mesh_plot = ax.tripcolor(
        triang, velocity.cpu(), vmin=vmin, vmax=vmax, shading="flat", cmap="viridis"
    )
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
    ax.set_title(title, fontsize=12)

    return mesh_plot

def denormalize(invar, mu, std):
    return invar * std.expand(invar.size()) + mu.expand(invar.size())

# Load data
node_stats = load_json("dataset/node_stats.json")
raw = np.load('./dataset/rawData.npy', allow_pickle=True)
pos = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt"))
faces = torch.from_numpy(np.load('mat_delaunay_filtered.npy'))
predict = np.load('predict/predict_l2tripletbest.npy')  # assuming dim = 2
#predict = np.load('interpolation/reconstruction_90_granular_120_131_triplet.npy')  # assuming dim = 2

traj = 90
attributes = [0, 1, 2]
times = list(range(0, 401))  # Animation frames
#times = list(range(120,131))

fig, axes = plt.subplots(3, 3, figsize=(24, 18))
cbar_axes = [[make_axes_locatable(axes[i][j]).append_axes("right", size="5%", pad=0.05)
              for j in range(3)] for i in range(3)]

def init():
    for row in axes:
        for ax in row:
            ax.clear()
    return []

def update(frame_idx):
    time = times[frame_idx]
    gt_all = torch.from_numpy(raw['x'][traj, time, :, :])  # (num_nodes, 3)
    pred = torch.from_numpy(predict[traj - 90, time, :])  # (num_nodes, 3)
    denom = denormalize(pred, node_stats["node_mean"], node_stats["node_std"])

    mesh_plots = []

    for j, attrib in enumerate(attributes):
        gt = gt_all[:, attrib]
        recon = denom[:, attrib]
        diff = gt - recon

        vmin = min(gt.min(), recon.min()).item()
        vmax = max(gt.max(), recon.max()).item()

        for i, data in enumerate([gt, recon, diff]):
            ax = axes[i][j]
            cax = cbar_axes[i][j]
            mesh = plot_mesh(data, pos, faces, vmin if i < 2 else None, vmax if i < 2 else None,
                             ax=ax, title=f"{['GT','Recon','GT-Recon'][i]} {['X','Y','P'][j]}")
            cax.cla()
            cb = fig.colorbar(mesh, cax=cax)
            cb.ax.tick_params(labelsize=8)
            mesh_plots.append(mesh)

    fig.suptitle(f"Time = {time}", fontsize=16)
    return mesh_plots

ani = FuncAnimation(fig, update, frames=len(times), interval=50, init_func=init, blit=False)
ani.save("mesh_animation_grid.gif", writer='pillow')
