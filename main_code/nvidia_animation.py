# import numpy as np
# from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
# from matplotlib import tri as mtri
# from matplotlib.animation import FuncAnimation, PillowWriter

# def animate_trajectory(velocity_array, mesh_pos, faces, title_prefix="Trajectory 79", mode='x', save_path="animation.gif"):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.set_aspect("equal")
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.tick_params(axis='both', which='minor', labelsize=15)
#     ax.autoscale(enable=True, axis='both', tight=True)

#     if mode == 'x':
#         attribute = 0
#         cb_title = "x velocity\n(m/s)"
#     elif mode == 'y':
#         attribute = 1
#         cb_title = "y velocity\n(m/s)"
#     else:
#         attribute = 2
#         cb_title = "pressure\n"

#     vmin = velocity_array[:, :, attribute].min()
#     vmax = velocity_array[:, :, attribute].max()

#     print(vmin, vmax)

#     triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)

#     # Initial plot
#     velocity = velocity_array[0, :, attribute]
#     mesh_plot = ax.tripcolor(triang, velocity, vmin=vmin, vmax=vmax, shading="gouraud")
#     ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

#     title = ax.set_title(f"{title_prefix} - t=0", fontsize=18)
#     cbar = fig.colorbar(mesh_plot, ax=ax, orientation="vertical")
#     cbar.ax.set_title(cb_title, fontsize=12)
#     cbar.ax.tick_params(labelsize=12)

#     def update(frame):
#         print(frame, velocity_array.shape)
#         velocity = velocity_array[frame, :, attribute]
#         mesh_plot.set_array(velocity)
#         title.set_text(f"{title_prefix} - t={frame}")

#     anim = FuncAnimation(fig, update, frames=velocity_tensor.shape[0], interval=50)
#     anim.save(save_path, writer='pillow')
#     plt.close(fig)

# # Load data
# raw = np.load('./dataset/rawData.npy', allow_pickle=True)
# pos = np.loadtxt(f"dataset/meshPosition_all.txt")
# filtered_faces = np.load('mat_delaunay_filtered.npy')

# for traj in [0, 40, 79]:
#     velocity_tensor = raw['x'][traj]  # shape: [timesteps, nodes, features]
#     for mode in ['x', 'y', 'p']:
#         animate_trajectory(
#             velocity_tensor, pos, filtered_faces,
#             title_prefix=f"Trajectory {traj}",
#             mode=mode,
#             save_path=f"dataset_visualize/{traj}_{mode}.gif"
#         )

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_trajectory_all_at_once(velocity_array, mesh_pos, faces, title_prefix="Trajectory", save_path="animation.gif"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    attributes = ['x velocity\n(m/s)', 'y velocity\n(m/s)', 'pressure\n']
    mesh_plots = []
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)

    vmins = [velocity_array[:, :, i].min() for i in range(3)]
    vmaxs = [velocity_array[:, :, i].max() for i in range(3)]

    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(attributes[i], fontsize=14)
        ax.autoscale(enable=True, axis='both', tight=True)
        velocity = velocity_array[0, :, i]
        mesh_plot = ax.tripcolor(triang, velocity, vmin=vmins[i], vmax=vmaxs[i], shading="gouraud")
        #ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
        mesh_plots.append(mesh_plot)
        cbar = fig.colorbar(mesh_plot, ax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.autoscale(enable=True, axis='both', tight=True)

    super_title = fig.suptitle(f"{title_prefix} - t=0", fontsize=16)

    def update(frame):
        for i in range(3):
            velocity = velocity_array[frame, :, i]
            mesh_plots[i].set_array(velocity)
        super_title.set_text(f"{title_prefix} - t={frame}")

    anim = FuncAnimation(fig, update, frames=velocity_array.shape[0], interval=50)
    anim.save(save_path, writer='pillow')
    plt.close(fig)

# Load data
raw = np.load('./dataset/rawData.npy', allow_pickle=True)
pos = np.loadtxt(f"dataset/meshPosition_all.txt")
filtered_faces = np.load('mat_delaunay_filtered.npy')

for traj in [0, 40, 79]:
    velocity_tensor = raw['x'][traj]  # shape: [timesteps, nodes, features]
    animate_trajectory_all_at_once(
        velocity_tensor, pos, filtered_faces,
        title_prefix=f"Trajectory {traj}",
        save_path=f"dataset_visualize/{traj}_xyz.gif"
    )