import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np

# Dummy data
N = 400
timesteps = np.arange(N)
colors = timesteps

# Replace this with your actual PCA results
xs = [np.random.randn(N, 2) * i for i in range(1, 5)]

# Setup
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)
axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

# Plot each subplot
for ax, x, title in zip(
    axs, xs,
    ["latent x", "latent y", "data x", "data y"]
):
    sc = ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='inferno', s=10)
    ax.set_title(title)
    ax.set_xlabel('C_0')
    ax.set_ylabel('C_1')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=400)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='inferno'), cax=cbar_ax)
cbar.set_label('Timesteps')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
plt.savefig('test_combined.png')
plt.show()