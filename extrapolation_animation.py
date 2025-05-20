import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_mesh(velocity, mesh_pos, faces, ax=None, fig=None, title='Mesh'):
    """
    Plots a mesh with a color map
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
    ax.set_aspect("equal")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.autoscale(enable=True, axis='both', tight=True)
    
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
    mesh_plot = ax.tripcolor(triang, velocity, vmin=velocity.min(), vmax=velocity.max(), shading="flat")
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

    ax.set_title(title, fontsize=20)
    
    # Create color bar
    if not hasattr(ax, "colorbar"):
        divider = make_axes_locatable(ax)
        ax.cax = divider.append_axes("right", size="5%", pad=0.5)
        ax.colorbar = fig.colorbar(mesh_plot, cax=ax.cax, orientation="vertical")
    else:
        ax.colorbar.update_normal(mesh_plot)

latent = np.load('latent/latent_l2tripletbest.npy')
recon = np.load('extrapolation/reconstruction_triplet.npy')
gt = np.load('./latent/test_data.npy')

position_mesh = np.loadtxt(f"dataset/meshPosition_all.txt")
faces = np.load('mat_delaunay_filtered.npy')

fig, (lax, rax, gax, dax) = plt.subplots(1, 4, figsize=(32, 8))
fig.tight_layout(pad=3.0)

def update(frame):
    # Clear plots
    lax.clear()
    rax.clear()
    gax.clear()
    dax.clear()
    
    # Plot latent space
    lax.scatter(latent[0, :frame, 0, 0], latent[0, :frame, 1, 0], alpha=0.5, c='gray', s=25)
    lax.scatter(latent[0, frame, 0, 0], latent[0, frame, 1, 0], alpha=1, c='red', s=50, label='T0')
    lax.scatter(latent[0, frame + 1, 0, 0], latent[0, frame + 1, 1, 0], alpha=1, c='blue', s=25, label='T1')
    lax.scatter(latent[0, frame + 2, 0, 0], latent[0, frame + 2, 1, 0], alpha=1, c='green', s=50, label='T2')

    extra_x = (2 * latent[0, frame + 1, 0, 0] - latent[0, frame, 0, 0])
    extra_y = (2 * latent[0, frame + 1, 1, 0] - latent[0, frame, 1, 0])
    lax.scatter(extra_x, extra_y, c='purple', s=40, label='Extrapolated')

    # Dashed line
    lax.plot([latent[0, frame, 0, 0], extra_x], 
             [latent[0, frame, 1, 0], extra_y], 
             linestyle='--', color='purple', alpha=0.5)

    # Residual error line
    t2_x = latent[0, frame + 2, 0, 0]
    t2_y = latent[0, frame + 2, 1, 0]
    lax.plot([extra_x, t2_x], [extra_y, t2_y], linestyle=':', color='orange', label='Residual Error')
    
    lax.set_title(f'Triplet Animation - Frame {frame}')
    lax.legend()

    # Plot meshes
    plot_mesh(recon[0, frame, :, 0], position_mesh, faces, ax=rax, fig=fig, title='Extrapolated')
    plot_mesh(gt[0, frame + 2, :, 0], position_mesh, faces, ax=gax, fig=fig, title='Ground Truth')
    plot_mesh(gt[0, frame + 2, :, 0] - recon[0, frame, :, 0], position_mesh, faces, ax=dax, fig=fig, title='GT-Ext')
    
ani = FuncAnimation(fig, update, frames=399, interval=400)
ani.save('test_extrapolate.gif', writer='pillow')
plt.show()
