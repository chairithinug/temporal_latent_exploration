import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# raw = np.load('./dataset/rawData.npy', allow_pickle=True)

# image_paths = []
# plt.figure(figsize=(12, 8))
# for dim in [2, 16, 32, 64, 128, 256, 'data']:
#     if dim == 'data':
#         latent = raw['x'][90:]
#     elif dim == 2:
#         latent = np.load(f'latent/latent_l{dim}_3.npy')
#     else:
#         latent = np.load(f'latent/latent_l{dim}.npy')
#     for i, name in zip(range(3), ['x', 'y', 'p']):
#         reducer = PCA(n_components=2)
#         sc = StandardScaler()
#         transformed = sc.fit_transform(latent[0,:,:,i])
#         if dim != 2:
#             x_reduced = reducer.fit_transform(transformed)
#         else:
#             x_reduced = transformed

        
#         fig, ax = plt.subplots(1)
#         plt.scatter(x_reduced[:,0], x_reduced[:,1], c=np.arange(len(x_reduced)), cmap="rainbow")
#         if dim == 2:
#             plt.title(f'Latent@dim={dim} of {name} of Trajectory 90 (Test)')
#         else:
#             plt.title(f'PCA on latent@dim={dim} of {name} of Trajectory 90 (Test)')
#         plt.ylabel('C_1')
#         plt.xlabel('C_0')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         norm = plt.Normalize(0, len(x_reduced))
#         sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
#         fig.colorbar(sm, cax=cax, orientation="vertical")
#         plt.title('Timesteps')
#         fname = f'visualization/pca_test_latent_{name}_dim{dim}.png'
#         plt.savefig(fname, bbox_inches='tight')
#         plt.clf()
#         image_paths.append(fname)
        
# fig, axes = plt.subplots(7, 3, figsize=(36, 56))
# for i, ax in enumerate(axes.flat):
#     img = plt.imread(image_paths[i])  # Read image
#     ax.imshow(img)                      # Display image
#     ax.axis('off')                      # Hide axes
# plt.tight_layout()
# plt.savefig('visualization/pca_test_latent.png', bbox_inches='tight')
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np

# Row and column titles
row_titles = ["dim=2", "dim=16", "dim=32", "dim=64", "dim=128", "dim=256", "Original"]
col_titles = ["x-vel", "y-vel", "p"]

xlim = (-30, 30)
ylim = (-30, 30)

# Setup
fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(7, 3, figure=fig, wspace=0.4, hspace=0.6)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(7)])

# Plot each subplot
for i, dim in enumerate([2, 16, 32, 64, 128, 256, 'data']):
    if dim == 'data':
        latent = np.load('./latent/test_data.npy')
    else:
        latent = np.load(f'latent/latent_l{dim}best.npy')
    for j in range(3):
        ax = axs[i][j]
        #ax.set_aspect('equal', adjustable='box')
        x = latent[0]
        reducer = PCA(n_components=2)
        sc = StandardScaler()
        transformed = sc.fit_transform(x[:,:,j])
        if dim != 2:
            x_reduced = reducer.fit_transform(transformed)
        else:
            x_reduced = x[:,:,j]
        #print(x_reduced.shape)
        sc = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=np.arange(len(x_reduced)), cmap='rainbow', s=10)
        #ax.set_xlim(xlim)
        #ax.set_ylim(ylim)
        ax.set_xlabel('C_0')
        ax.set_ylabel('C_1')

# Add column titles above the top row
for j, title in enumerate(col_titles):
    axs[0][j].set_title(title, fontsize=14, pad=20)

# Add row titles to the left of each row
for i, title in enumerate(row_titles):
    axs[i][0].annotate(title, xy=(0, 0.5), xytext=(-60, 0),
                xycoords='axes fraction', textcoords='offset points',
                size=14, ha='right', va='center', rotation=90)

# Shared colorbar
cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
norm = plt.Normalize(vmin=0, vmax=len(x_reduced))
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), cax=cbar_ax)
cbar.set_label('Timesteps')

plt.suptitle(f'Trajectory 90 (Test)')
plt.tight_layout(rect=[0.06, 0.03, 0.9, 0.97])  # leave room for labels
plt.savefig('test_combined.png')
plt.show()