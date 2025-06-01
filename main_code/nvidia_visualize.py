# import numpy as np
# from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
# from matplotlib import tri as mtri
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import torch

# @torch.no_grad()
# def plot_mesh(velocity, mesh_pos, faces, title="Mesh"):
#     """
#     Plots two graphs with each other.
#     Can be used to plot the predicted graph and the ground truth
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(30, 20))
#     #fig, ax = plt.subplots(1, 1, figsize=(40, 32))

#     ax.cla()
#     ax.set_aspect("equal")
#     ax.tick_params(axis='both', which='major', labelsize=30)
#     ax.tick_params(axis='both', which='minor', labelsize=30)
#     ax.autoscale(enable=True, axis='both', tight=True)
#     #ax.set_axis_off()
    
#     print(mesh_pos.shape, faces.shape, velocity.shape)

#     triang = mtri.Triangulation(mesh_pos[:, 0].cpu(), mesh_pos[:, 1].cpu(), faces.cpu())
#     mesh_plot = ax.tripcolor(
#         triang, velocity.cpu(), vmin=velocity.min(), vmax=velocity.max(), shading="flat"
#     )
#     ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

#     ax.set_title(title, fontsize=30)
#     ax.set_ylabel('y position', fontsize=30)
#     ax.set_xlabel('x position', fontsize=30)
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.5)
#     clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
#     clb.ax.tick_params(labelsize=30)
#     clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 30})
#     return fig

# raw = np.load('./dataset/rawData.npy', allow_pickle=True)
# pos = np.loadtxt(f"dataset/meshPosition_all.txt")
# tri = Delaunay(pos)
# filtered_faces = np.load('mat_delaunay_filtered.npy')
# traj = 50
# time = 0
# fig = plot_mesh(torch.from_numpy(raw['x'][traj,time,:,0]), torch.from_numpy(pos), torch.from_numpy(filtered_faces))
# fig.savefig(f'z.png', bbox_inches='tight')

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

@torch.no_grad()
def plot_multiple_timesteps(velocity_array, mesh_pos, faces, timesteps, title_prefix="Trajectory 79", mode='x'):
    """
    Plots 5 vertical subplots of the mesh at different timesteps.
    """
    fig, axes = plt.subplots(len(timesteps), 1, figsize=(20, 30))
    if len(timesteps) == 1:
        axes = [axes]

    if mode == 'x':
        attribute = 0
        cb_title = "x velocity\n(m/s)"
    elif mode == 'y':
        attribute = 1
        cb_title = "y velocity\n(m/s)"
    else:
        attribute = 2
        cb_title = "pressure\n"
    vmin = velocity_array[:, :, attribute].min()
    vmax = velocity_array[:, :, attribute].max()

    for ax, t in zip(axes, timesteps):
        ax.set_aspect("equal")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.autoscale(enable=True, axis='both', tight=True)

        triang = mtri.Triangulation(
            mesh_pos[:, 0].cpu(), mesh_pos[:, 1].cpu(), faces.cpu()
        )
        velocity = velocity_array[t, :, attribute]
        mesh_plot = ax.tripcolor(
            triang,
            velocity.cpu(),
            vmin=vmin,
            vmax=vmax,
            shading="flat"
        )
        ax.triplot(triang, "ko-", ms=0.5, lw=0.3)


        ax.set_ylabel('y position', fontsize=20)
        if ax == axes[0]:
            ax.set_title(f"{title_prefix}", fontsize=24)
        if ax == axes[-1]:
            ax.set_xlabel('x position', fontsize=20)

        # Add row label (timestep) on the left
        ax.text(-0.15, 0.5, f"t = {t}", fontsize=20, va='center', ha='right',
                transform=ax.transAxes, rotation=90)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
        # clb.ax.tick_params(labelsize=20)
        # clb.ax.set_title("x velocity\n(m/s)", fontsize=18)

    #fig.text(0.5, 0.92, f"Trajectory {1}", ha='center', va='bottom', fontsize=24)
    # Shared colorbar
    cbar_ax = fig.add_axes([0.8, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(mesh_plot, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_title(cb_title, fontsize=18)
    plt.tight_layout()
    return fig

# Load data
raw = np.load('./dataset/rawData.npy', allow_pickle=True)
pos = np.loadtxt(f"dataset/meshPosition_all.txt")
tri = Delaunay(pos)
filtered_faces = np.load('mat_delaunay_filtered.npy')

# Prepare tensors
pos_tensor = torch.from_numpy(pos)
faces_tensor = torch.from_numpy(filtered_faces)

for traj in [0, 40, 79]:
    velocity_tensor = torch.from_numpy(raw['x'][traj])  # shape: [timesteps, nodes, features]

    # Define timesteps
    timesteps = [0, 50, 100, 200, 400]

    # Plot and save
    for mode in ['x','y','p']:
        fig = plot_multiple_timesteps(velocity_tensor, pos_tensor, faces_tensor, timesteps, title_prefix=f"Trajectory {traj}", mode=mode)
        fig.savefig(f'dataset_visualize/{traj}_{mode}.png', bbox_inches='tight')