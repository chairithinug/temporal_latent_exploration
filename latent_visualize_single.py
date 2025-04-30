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
for dim in [2]:
    latent = np.load(f'latent/latent_l{dim}_3layers_forced_best.npy')
    for i, name in zip(range(3), ['x', 'y', 'p']):
        sc = StandardScaler()
        transformed = sc.fit_transform(latent[0,:,:,i])
        x_reduced = transformed

        
        fig, ax = plt.subplots(1)
        plt.scatter(x_reduced[:,0], x_reduced[:,1], c=np.arange(len(x_reduced)), cmap="rainbow")
        plt.title(f'latent@dim={dim} of {name} of Trajectory 90 (Test)')
        plt.ylabel('C_1')
        plt.xlabel('C_0')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = plt.Normalize(0, len(x_reduced))
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
        fig.colorbar(sm, cax=cax, orientation="vertical")
        plt.title('Timesteps')
        fname = f'visualization/pca_test_latent_{name}_dim{dim}_3layers_forced_best.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
        image_paths.append(fname)
        