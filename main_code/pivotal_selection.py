import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

raw = np.load('./dataset/rawData.npy', allow_pickle=True)
position_mesh = torch.from_numpy(np.loadtxt('./dataset/meshPosition_all.txt'))
position_pivotal = torch.from_numpy(np.loadtxt('./dataset/meshPosition_pivotal_l256.txt'))

# plt.figure(figsize=(15,8))
# plt.scatter(position_mesh[:,0], position_mesh[:,1], alpha=0.2)
# plt.scatter(position_pivotal_2[:,0], position_pivotal_2[:,1], label='2', alpha=0.6)
# plt.legend()
# plt.title('All Node Positions and Sampled Positions')
# plt.ylabel('Y')
# plt.xlabel('X')

# position_pivotal_128 = np.random.choice(np.arange(len(position_mesh)), 128, replace=False)
# position_pivotal_128 = position_mesh[position_pivotal_128]
# position_pivotal_64 = np.random.choice(np.arange(len(position_mesh)), 64, replace=False)
# position_pivotal_64 = position_mesh[position_pivotal_64]
# position_pivotal_32 = np.random.choice(np.arange(len(position_mesh)), 32, replace=False)
# position_pivotal_32 = position_mesh[position_pivotal_32]
# position_pivotal_16 = np.random.choice(np.arange(len(position_mesh)), 16, replace=False)
# position_pivotal_16 = position_mesh[position_pivotal_16]
# position_pivotal_2 = np.random.choice(np.arange(len(position_mesh)), 2, replace=False)
# position_pivotal_2 = position_mesh[position_pivotal_2]
# np.savetxt('./dataset/meshPosition_pivotal_l128.txt', position_pivotal_128)
# np.savetxt('./dataset/meshPosition_pivotal_l64.txt', position_pivotal_64)
# np.savetxt('./dataset/meshPosition_pivotal_l32.txt', position_pivotal_32)
# np.savetxt('./dataset/meshPosition_pivotal_l16.txt', position_pivotal_16)
# np.savetxt('./dataset/meshPosition_pivotal_l2.txt', position_pivotal_2)

position_pivotal_2 = np.loadtxt('./dataset/meshPosition_pivotal_l2.txt')
position_pivotal_16 = np.loadtxt('./dataset/meshPosition_pivotal_l16.txt')
position_pivotal_32 = np.loadtxt('./dataset/meshPosition_pivotal_l32.txt')
position_pivotal_64 = np.loadtxt('./dataset/meshPosition_pivotal_l64.txt')
position_pivotal_128 = np.loadtxt('./dataset/meshPosition_pivotal_l128.txt')
position_pivotal_256 = np.loadtxt('./dataset/meshPosition_pivotal_l256.txt')

fig = plt.figure(figsize=(12,6))
ax = plt.gca()
#ax.set_aspect("equal")
plt.scatter(position_mesh[:,0], position_mesh[:,1], alpha=0.2)
plt.scatter(position_pivotal[:,0], position_pivotal[:,1], label='256', alpha=0.6, s=25)
plt.scatter(position_pivotal_128[:,0], position_pivotal_128[:,1], label='128', alpha=0.6, s=25)
plt.scatter(position_pivotal_64[:,0], position_pivotal_64[:,1], label='64', alpha=0.6, s=25)
plt.scatter(position_pivotal_32[:,0], position_pivotal_32[:,1], label='32', alpha=0.6, s=25)
plt.scatter(position_pivotal_16[:,0], position_pivotal_16[:,1], label='16', alpha=0.6, s=25)
plt.scatter(position_pivotal_2[:,0], position_pivotal_2[:,1], label='2', alpha=0.6, s=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.autoscale(enable=True, axis='both', tight=True)
plt.legend()
plt.title('All Node Positions and Sampled Positions', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.savefig('sampled_location.png')