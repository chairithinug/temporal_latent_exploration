# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib import tri as mtri
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# def plot_mesh(velocity, mesh_pos, faces, vmin=None, vmax=None, ax=None, fig=None, title='Mesh'):
#     """
#     Plots a mesh with a color map
#     """
#     if ax is None or fig is None:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
#     ax.set_aspect("equal")
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.tick_params(axis='both', which='minor', labelsize=15)
#     ax.autoscale(enable=True, axis='both', tight=True)
    
#     triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
#     if vmin is None:
#         vmin=velocity.min()
#     if vmax is None:
#          vmax=velocity.max()
#     mesh_plot = ax.tripcolor(triang, velocity, vmin=vmin, vmax=vmax, shading="flat")
#     ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

#     ax.set_title(title, fontsize=25)
    
#     # Create color bar
#     if not hasattr(ax, "colorbar"):
#         divider = make_axes_locatable(ax)
#         ax.cax = divider.append_axes("right", size="5%", pad=0.3)
#         ax.colorbar = fig.colorbar(mesh_plot, cax=ax.cax, orientation="vertical")
#     else:
#         ax.colorbar.update_normal(mesh_plot)
#     ax.colorbar.ax.set_title('x velocity\n(m/s)',fontsize=15)
#     ax.colorbar.ax.tick_params(labelsize=15)

# latent = np.load('latent/latent_l2tripletbest.npy')
# recon = np.load('interpolation/reconstruction_triplet.npy')
# gt = np.load('./latent/test_data.npy')

# position_mesh = np.loadtxt(f"dataset/meshPosition_all.txt")
# faces = np.load('mat_delaunay_filtered.npy')

# import json
# with open(f'inter_err_ot_90.json', 'r') as f:
#     data = json.load(f)
#     loss = data['avg_loss']
#     times = data['timesteps']

# fig = plt.figure(figsize=(18, 12))
# fig.tight_layout(pad=5.0)
# gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)

# # Axes assignment
# lax = fig.add_subplot(gs[:2, 0])       # Latent space spans all rows of left column
# rax = fig.add_subplot(gs[1, 1])       # Row 1, right column
# gax = fig.add_subplot(gs[0, 1])       # Row 2, right column
# dax = fig.add_subplot(gs[2, 1])       # Row 3, right column
# eax = fig.add_subplot(gs[2, 0])

# def update(frame):
#     # Clear plots
#     lax.clear()
#     rax.clear()
#     gax.clear()
#     dax.clear()
#     eax.clear()

#     eax.semilogy(times[:frame+1], loss[:frame+1], label='avg_loss', alpha=0.7, marker='o')
#     eax.set_xlabel('Timesteps', fontsize=15)
#     eax.set_ylabel('Log MSE', fontsize=15)
#     eax.set_title('Log MSE vs Timesteps', fontsize=25)
    
#     # Plot latent space
#     lax.scatter(latent[0, :frame+1, 0, 0], latent[0, :frame+1, 1, 0], alpha=0.5, c='gray', s=25)
#     lax.scatter(latent[0, frame, 0, 0], latent[0, frame, 1, 0], alpha=1, c='red', s=50, label='T0')
#     lax.scatter(latent[0, frame + 1, 0, 0], latent[0, frame + 1, 1, 0], alpha=1, c='green', s=50, label='T1')
#     lax.scatter(latent[0, frame + 2, 0, 0], latent[0, frame + 2, 1, 0], alpha=1, c='blue', s=50, label='T2')

#     # Dashed line and midpoint
#     lax.plot([latent[0, frame, 0, 0], latent[0, frame + 2, 0, 0]], 
#              [latent[0, frame, 1, 0], latent[0, frame + 2, 1, 0]], 
#              linestyle='--', color='black', alpha=0.5)

#     midpoint_x = (latent[0, frame, 0, 0] + latent[0, frame + 2, 0, 0]) / 2
#     midpoint_y = (latent[0, frame, 1, 0] + latent[0, frame + 2, 1, 0]) / 2
#     lax.scatter(midpoint_x, midpoint_y, c='purple', s=50, label='Interpolated')

#     # Residual error line
#     t1_x = latent[0, frame + 1, 0, 0]
#     t1_y = latent[0, frame + 1, 1, 0]
#     lax.plot([midpoint_x, t1_x], [midpoint_y, t1_y], linestyle=':', color='orange', label='Residual Error')
    
#     lax.set_title('Latent Space', fontsize=25)
#     lax.legend()

#     lax.tick_params(axis='both', which='major', labelsize=15)
#     lax.tick_params(axis='both', which='minor', labelsize=15)

#     vmin = min(recon[0, frame, :, 0].min(),gt[0, frame + 1, :, 0].min())
#     vmax = max(recon[0, frame, :, 0].max(),gt[0, frame + 1, :, 0].max())
#     # Plot meshes
#     plot_mesh(recon[0, frame, :, 0], position_mesh, faces, vmin=vmin, vmax=vmax, ax=rax, fig=fig, title='Interpolated (Int)')
#     plot_mesh(gt[0, frame + 1, :, 0], position_mesh, faces, vmin=vmin, vmax=vmax, ax=gax, fig=fig, title='Ground Truth (GT)')
#     plot_mesh(gt[0, frame + 1, :, 0] - recon[0, frame, :, 0], position_mesh, faces, ax=dax, fig=fig, title='GT - Int')
#     fig.suptitle(f'Interpolation timestep {frame+1} of Trajectory 90 (Test)', fontsize=30)
# ani = FuncAnimation(fig, update, frames=399, interval=400)
# ani.save('test_interpolate.gif', writer='pillow')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load data
latent = np.load('latent/latent_l2tripletbest.npy')
recon = np.load('interpolation/reconstruction_triplet.npy')
gt = np.load('./latent/test_data.npy')
position_mesh = np.loadtxt("dataset/meshPosition_all.txt")
faces = np.load('mat_delaunay_filtered.npy')

loss = [None for _ in range(3)]
import json
with open('inter_err_ot_90.json', 'r') as f:
    data = json.load(f)
    loss[0] = data['avg_rel_x']
    loss[1] = data['avg_rel_y']
    loss[2] = data['avg_rel_p']
    times = data['timesteps']

# Attribute configuration
attr_titles = ['x velocity\n(m/s)', 'y velocity\n(m/s)', 'pressure\n']
attr_short = ['X', 'Y', 'P']
plot_titles = ['Ground Truth ({}GT)', 'Interpolated {} ({}Ixt)', '{}GT - {}Ixt']

# Setup mesh triangulation
triang = mtri.Triangulation(position_mesh[:, 0], position_mesh[:, 1], faces)

# Create axes grid and mesh/colorbar containers
axes = [[None]*3 for _ in range(3)]  # 3 rows x 3 columns
meshes = [[None]*3 for _ in range(3)]
colorbars = [[None]*3 for _ in range(3)]

# Create figure
fig = plt.figure(figsize=(15, 20))
gs = fig.add_gridspec(5, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1, 1, 1, 1], wspace=0.3, hspace=0.3)

# Latent and error axes (left column)
xlax = fig.add_subplot(gs[0, 0])   # Latent space (rowspan 2)
ylax = fig.add_subplot(gs[0, 1])   # Latent space (rowspan 2)
plax = fig.add_subplot(gs[0, 2])   # Latent space (rowspan 2)

xeax = fig.add_subplot(gs[1, 0])    # Loss plot
yeax = fig.add_subplot(gs[1, 1])    # Loss plot
peax = fig.add_subplot(gs[1, 2])    # Loss plot

sc_latent_all = [None for _ in range(3)]
sc_t0  = [None for _ in range(3)]
sc_t1  = [None for _ in range(3)]
sc_t2  = [None for _ in range(3)]
sc_inter  = [None for _ in range(3)]
line_dashed  = [None for _ in range(3)]
line_residual  = [None for _ in range(3)]
line_loss = [None for _ in range(3)]

# Initialize plots
for i, this_ax in enumerate([xlax, ylax, plax]):
    sc_latent_all[i] = this_ax.scatter([], [], alpha=0.3, c='gray', s=25)
    sc_t0[i] = this_ax.scatter([], [], alpha=1, c='red', s=25, label='T0')
    sc_t1[i] = this_ax.scatter([], [], alpha=1, c='blue', s=25, label='T1')
    sc_t2[i] = this_ax.scatter([], [], alpha=1, c='green', s=25, label='T2')
    sc_inter[i] = this_ax.scatter([], [], alpha=1, c='purple', s=25, label='Interpolated')
    line_dashed[i], = this_ax.plot([], [], linestyle='--', color='black', alpha=0.5)
    line_residual[i], = this_ax.plot([], [], linestyle='--', color='red', label='Residual')
    this_ax.set_title(f'{attr_short[i]} Latent Space', fontsize=13)
    this_ax.legend()
    this_ax.set_xlim(min(latent[0,:,0,i]), max(latent[0,:,0,i]))
    this_ax.set_ylim(min(latent[0,:,1,i]), max(latent[0,:,1,i]))
    # this_ax.tick_params(axis='both', which='major', labelsize=15)

for i, this_ax in enumerate([xeax, yeax, peax]):
    line_loss[i], = this_ax.semilogy([], [], label='avg_loss', alpha=1, marker='o')
    this_ax.set_xlabel('Timesteps', fontsize=13)
    this_ax.set_ylabel(f'Avg Rel {attr_short[i]}', fontsize=13)
    this_ax.set_title(f'Avg Rel {attr_short[i]} vs Timesteps', fontsize=13)
    this_ax.set_xlim(0, times[-1])
    this_ax.set_ylim(min(loss[i]) * 0.9, max(loss[i]) * 1.1)

def init_mesh_plot(ax, title, cbar_label):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.autoscale(enable=True, axis='both', tight=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    mesh = ax.tripcolor(triang, np.zeros(triang.x.shape), shading="gouraud")
    cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cb.ax.set_title(cbar_label, fontsize=10)
    return mesh, cb


for col, (attr_label, attr_tag) in enumerate(zip(attr_titles, attr_short)):
    for row, title_template in enumerate(plot_titles, start=2):
        ax = fig.add_subplot(gs[row, col])
        title = title_template.format(attr_tag, attr_tag)
        mesh, cb = init_mesh_plot(ax, title, attr_label)
        axes[row-2][col] = ax
        meshes[row-2][col] = mesh
        colorbars[row-2][col] = cb

def update(frame):
    print(frame)
    fig.suptitle(f'Interpolation timestep {frame+1} of Trajectory 90 (Test)', fontsize=30)

    # --- Latent plot update ---
    # Plot all latent points up to current frame (trajectory 90, point 0)
    for i in range(3):
        if frame >= 1:
            latent_points = latent[0, :frame, :, i]  # shape: (frame+1, 3, latent_dim)
            sc_latent_all[i].set_offsets(latent_points[:, :2])

        # Set t0, t1, t2 points
        sc_t0[i].set_offsets(latent[0, frame, :2, i])
        sc_t1[i].set_offsets(latent[0, frame+1, :2, i])
        sc_t2[i].set_offsets(latent[0, frame+2, :2, i])

        # Compute interpolated point 
        inter = (latent[0, frame+2, :, i] + latent[0, frame, :, i]) / 2
        sc_inter[i].set_offsets(inter[:2])

        # # Plot line from t0 interpolated to actual t2
        # line_dashed[i].set_data(
        #     [latent[0, frame, 0, i], latent[0, frame+2, 0, i]],
        #     [latent[0, frame, 1, i], latent[0, frame+2, 1, i]]
        # )
        line_residual[i].set_data(
            [inter[0], latent[0, frame+1, 0, i]],
            [inter[1], latent[0, frame+1, 1, i]]
        )

        # --- Loss plot update ---
        line_loss[i].set_data(times[:frame+1], loss[i][:frame+1])


    for i, attr_idx in enumerate([0, 1, 2]):  # x, y, pressure
        recon_data = recon[0, frame, :, attr_idx]
        gt_data = gt[0, frame + 1, :, attr_idx]
        diff_data = gt_data - recon_data

        vmin = min(recon_data.min(), gt_data.min())
        vmax = max(recon_data.max(), gt_data.max())
        max_diff = diff_data.max()
        min_diff = diff_data.min()

        for j, data in enumerate([gt_data, recon_data, diff_data]):  # GT, Ext, Diff
            mesh = meshes[j][i]
            cb = colorbars[j][i]
            mesh.set_array(data)
            if j != 2:
                mesh.set_clim(vmin, vmax)
            else:
                mesh.set_clim(min_diff, max_diff)
            cb.update_normal(mesh)

    return []

ani = FuncAnimation(fig, update, frames=399, interval=200, blit=False)
ani.save('test_interpolate_optimized.gif', writer='pillow')
plt.show()