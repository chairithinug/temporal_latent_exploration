import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_mesh(velocity, mesh_pos, faces, vmin=None, vmax=None, ax=None, fig=None, title='Mesh'):
    """
    Plots a mesh with a color map
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
    ax.set_aspect("equal")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.autoscale(enable=True, axis='both', tight=True)
    
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
    if vmin is None:
        vmin=velocity.min()
    if vmax is None:
         vmax=velocity.max()
    mesh_plot = ax.tripcolor(triang, velocity, vmin=vmin, vmax=vmax, shading="flat")
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

    ax.set_title(title, fontsize=25)
    
    # Create color bar
    if not hasattr(ax, "colorbar"):
        divider = make_axes_locatable(ax)
        ax.cax = divider.append_axes("right", size="5%", pad=0.3)
        ax.colorbar = fig.colorbar(mesh_plot, cax=ax.cax, orientation="vertical")
    else:
        ax.colorbar.update_normal(mesh_plot)
    ax.colorbar.ax.set_title('x velocity\n(m/s)',fontsize=15)
    ax.colorbar.ax.tick_params(labelsize=15)

latent = np.load('latent/latent_l2tripletbest.npy')
recon = np.load('interpolation/reconstruction_triplet.npy')
gt = np.load('./latent/test_data.npy')

position_mesh = np.loadtxt(f"dataset/meshPosition_all.txt")
faces = np.load('mat_delaunay_filtered.npy')

import json
with open(f'extra_err_ot_90.json', 'r') as f:
    data = json.load(f)
    loss = data['avg_loss']
    times = data['timesteps']

fig = plt.figure(figsize=(18, 12))
fig.tight_layout(pad=5.0)
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)

# Axes assignment
lax = fig.add_subplot(gs[:2, 0])       # Latent space spans all rows of left column
rax = fig.add_subplot(gs[0, 1])       # Row 1, right column
gax = fig.add_subplot(gs[1, 1])       # Row 2, right column
dax = fig.add_subplot(gs[2, 1])       # Row 3, right column
eax = fig.add_subplot(gs[2, 0])

def update(frame):
    # Clear plots
    lax.clear()
    rax.clear()
    gax.clear()
    dax.clear()
    eax.clear()

    eax.semilogy(times[:frame], loss[:frame], label='avg_loss', alpha=0.7, marker='o')
    eax.set_xlabel('Timesteps', fontsize=15)
    eax.set_ylabel('Log MSE', fontsize=15)
    
    # Plot latent space
    lax.scatter(latent[0, :frame, 0, 0], latent[0, :frame, 1, 0], alpha=0.5, c='gray', s=25)
    lax.scatter(latent[0, frame, 0, 0], latent[0, frame, 1, 0], alpha=1, c='red', s=50, label='T0')
    lax.scatter(latent[0, frame + 1, 0, 0], latent[0, frame + 1, 1, 0], alpha=1, c='green', s=50, label='T1')
    lax.scatter(latent[0, frame + 2, 0, 0], latent[0, frame + 2, 1, 0], alpha=1, c='blue', s=50, label='T2')

    # Dashed line and midpoint
    lax.plot([latent[0, frame, 0, 0], latent[0, frame + 2, 0, 0]], 
             [latent[0, frame, 1, 0], latent[0, frame + 2, 1, 0]], 
             linestyle='--', color='black', alpha=0.5)

    midpoint_x = (latent[0, frame, 0, 0] + latent[0, frame + 2, 0, 0]) / 2
    midpoint_y = (latent[0, frame, 1, 0] + latent[0, frame + 2, 1, 0]) / 2
    lax.scatter(midpoint_x, midpoint_y, c='purple', s=50, label='Interpolated')

    # Residual error line
    t1_x = latent[0, frame + 1, 0, 0]
    t1_y = latent[0, frame + 1, 1, 0]
    lax.plot([midpoint_x, t1_x], [midpoint_y, t1_y], linestyle=':', color='orange', label='Residual Error')
    
    lax.set_title('Latent Space', fontsize=25)
    lax.legend()

    lax.tick_params(axis='both', which='major', labelsize=15)
    lax.tick_params(axis='both', which='minor', labelsize=15)

    vmin = min(recon[0, frame, :, 0].min(),gt[0, frame + 1, :, 0].min())
    vmax = max(recon[0, frame, :, 0].max(),gt[0, frame + 1, :, 0].max())
    # Plot meshes
    plot_mesh(recon[0, frame, :, 0], position_mesh, faces, vmin=vmin, vmax=vmax, ax=rax, fig=fig, title='Interppolated (Int)')
    plot_mesh(gt[0, frame + 1, :, 0], position_mesh, faces, vmin=vmin, vmax=vmax, ax=gax, fig=fig, title='Ground Truth (GT)')
    plot_mesh(gt[0, frame + 1, :, 0] - recon[0, frame, :, 0], position_mesh, faces, ax=dax, fig=fig, title='GT - Int')
    fig.suptitle(f'Interpolation timestep {frame+2} of Trajectory 90 (Test)', fontsize=30)
ani = FuncAnimation(fig, update, frames=399, interval=400)
ani.save('test_interpolate.gif', writer='pillow')
plt.show()
