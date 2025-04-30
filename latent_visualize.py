import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

raw = np.load('./dataset/rawData.npy', allow_pickle=True)

image_paths = []
plt.figure(figsize=(12, 8))
for dim in [2, 16, 32, 64, 128, 256, 'data']:
    if dim == 'data':
        latent = raw['x'][90:]
    elif dim == 2:
        latent = np.load(f'latent/latent_l{dim}_3.npy')
    else:
        latent = np.load(f'latent/latent_l{dim}.npy')
    for i, name in zip(range(3), ['x', 'y', 'p']):
        reducer = PCA(n_components=2)
        sc = StandardScaler()
        transformed = sc.fit_transform(latent[0,:,:,i])
        if dim != 2:
            x_reduced = reducer.fit_transform(transformed)
        else:
            x_reduced = transformed

        
        fig, ax = plt.subplots(1)
        plt.scatter(x_reduced[:,0], x_reduced[:,1], c=np.arange(len(x_reduced)), cmap="rainbow")
        if dim == 2:
            plt.title(f'Latent@dim={dim} of {name} of Trajectory 90 (Test)')
        else:
            plt.title(f'PCA on latent@dim={dim} of {name} of Trajectory 90 (Test)')
        plt.ylabel('C_1')
        plt.xlabel('C_0')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = plt.Normalize(0, len(x_reduced))
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
        fig.colorbar(sm, cax=cax, orientation="vertical")
        plt.title('Timesteps')
        fname = f'visualization/pca_test_latent_{name}_dim{dim}.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
        image_paths.append(fname)
        
fig, axes = plt.subplots(7, 3, figsize=(36, 56))
for i, ax in enumerate(axes.flat):
    img = plt.imread(image_paths[i])  # Read image
    ax.imshow(img)                      # Display image
    ax.axis('off')                      # Hide axes
plt.tight_layout()
plt.savefig('visualization/pca_test_latent.png', bbox_inches='tight')
plt.show()