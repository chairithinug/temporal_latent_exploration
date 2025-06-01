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
        triang, velocity.cpu(), vmin=vmin, vmax=vmax, shading="gouraud", cmap="viridis"
    )
    # ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
    ax.set_title(title, fontsize=12)

    return mesh_plot


# Load data
node_stats = load_json("dataset/node_stats.json")
raw = np.load('./dataset/rawData.npy', allow_pickle=True)
pos = torch.from_numpy(np.loadtxt("dataset/meshPosition_all.txt"))
faces = torch.from_numpy(np.load('mat_delaunay_filtered.npy'))
#predict = np.load('interpolation/reconstruction_granular_121_122_10steps_triplet.npy')  # assuming dim = 2
predict = np.load('interpolation/reconstruction_90_coarse_0_400_triplet.npy')  # assuming dim = 2

traj = 90
attributes = [0, 1, 2]
#times = list(range(11))
times = list(range(21))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
cbar_axes = [make_axes_locatable(axes[j]).append_axes("right", size="5%", pad=0.05)
             for j in range(3)]
def init():
    for ax in axes:
        ax.clear()
    return []

def update(frame_idx):
    time = times[frame_idx]
    pred = torch.from_numpy(predict[traj - 90, time, :])  # (num_nodes, 3)

    mesh_plots = []

    for j, attrib in enumerate(attributes):
        recon = pred[:, attrib]

        vmin = recon.min().item()
        vmax = recon.max().item()

        for i, data in enumerate([recon]):
            ax = axes[j]
            cax = cbar_axes[j]
            mesh = plot_mesh(data, pos, faces, vmin if i < 2 else None, vmax if i < 2 else None,
                             ax=ax, title=f"{['Recon'][i]} {['X','Y','P'][j]}")
            cax.cla()
            cb = fig.colorbar(mesh, cax=cax)
            cb.ax.tick_params(labelsize=8)
            mesh_plots.append(mesh)

    #fig.suptitle(f"Time = {121+(time)/10:.2f}", fontsize=16)
    fig.suptitle(f"Coarse walk={time:.2f}", fontsize=16)
    return mesh_plots

#ani = FuncAnimation(fig, update, frames=len(times), interval=50, init_func=init, blit=False)
ani = FuncAnimation(fig, update, frames=len(times), interval=250, init_func=init, blit=False)
#ani.save(f"granular{traj}.gif", writer='pillow')
ani.save(f"coarse{traj}.gif", writer='pillow')
