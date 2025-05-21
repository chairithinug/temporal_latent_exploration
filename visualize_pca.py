# import numpy as np
# import torch

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# raw = np.load('./dataset/rawData.npy', allow_pickle=True)

# image_paths = []
# plt.figure(figsize=(12, 8))
# traj = 50
# for i, name in zip(range(3), ['x', 'y', 'p']):
#     reducer = PCA(n_components=2)
#     sc = StandardScaler()
#     transformed = sc.fit_transform(raw['x'][{traj},:,:,i])
#     x_reduced = reducer.fit_transform(transformed)

#     fig, ax = plt.subplots(1)
#     plt.scatter(x_reduced[:,0], x_reduced[:,1], c=np.arange(len(x_reduced)), cmap="inferno")
#     plt.title(f'PCA on {name} data of Trajectory {traj} (Training)')
#     plt.ylabel('C_1')
#     plt.xlabel('C_0')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     norm = plt.Normalize(0, len(x_reduced))
#     sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
#     fig.colorbar(sm, cax=cax, orientation="vertical")
#     plt.title('Timesteps')
#     fname = f'nvidia_visualizations/pca_{name}_{traj}.png'
#     plt.savefig(fname)
#     plt.clf()
#     image_paths.append(fname)
    
# fig, axes = plt.subplots(1, 3, figsize=(12, 24))
# for i, ax in enumerate(axes.flat):
#     img = plt.imread(image_paths[i])  # Read image
#     ax.imshow(img)                      # Display image
#     ax.axis('off')                      # Hide axes
# plt.tight_layout()
# plt.savefig('pca_train_dataspace.png', bbox_inches='tight', dpi=300)
# plt.show()

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load data
raw = np.load('./dataset/rawData.npy', allow_pickle=True)

traj = 40
variable_names = ['x velocity', 'y velocity', 'pressure']
colors = np.arange(raw['x'].shape[1])  # assuming shape = (N_traj, N_timesteps, N_particles, 3)

# Create shared figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Normalize for shared colorbar
norm = plt.Normalize(colors.min(), colors.max())
sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)

# Loop over each attribute (x, y, p)
for i, (name, ax) in enumerate(zip(variable_names, axes)):
    reducer = PCA(n_components=2)
    sc = StandardScaler()

    # Apply PCA on the selected trajectory and component
    transformed = sc.fit_transform(raw['x'][traj, :, :, i])
    x_reduced = reducer.fit_transform(transformed)

    scatter = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=colors, cmap="rainbow", norm=norm)
    ax.set_title(f'PCA on {name}')
    ax.set_xlabel('C_0')
    ax.set_ylabel('C_1')

# Add one shared colorbar
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', shrink=1, label='Timesteps')
fig.suptitle(f'PCA on Each Component of Trajectory {traj} (Training)', fontsize=16)

# Save and show
plt.savefig('pca_train_dataspace_sharedcolorbar.png', dpi=300)
plt.show()