import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np

# Dummy data
N = 400
timesteps = np.arange(N)
colors = timesteps

# Replace with your actual data (21 datasets)
xs = [np.random.randn(N, 2) * (i + 1) for i in range(21)]

# Row and column titles
row_titles = ["dim=2", "dim=16", "dim=32", "dim=62", "dim=128", "dim=256", "Original"]
col_titles = ["V_x", "V_y", "p"]

# Setup
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(7, 3, figure=fig, wspace=0.4, hspace=0.6)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(7)])

# Plot each subplot
for i in range(7):
    for j in range(3):
        idx = i * 3 + j
        ax = axs[i][j]
        x = xs[idx]
        sc = ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='rainbow', s=10)
        ax.set_xlabel('C_0')
        ax.set_ylabel('C_1')

# Add column titles above the top row
for j, title in enumerate(col_titles):
    ax = axs[0][j]
    ax.set_title(title, fontsize=14, pad=20)

# Add row titles to the left of each row
for i, title in enumerate(row_titles):
    ax = axs[i][0]
    ax.annotate(title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 35, 0),
                xycoords='axes fraction', textcoords='offset points',
                size=14, ha='right', va='center', rotation=90)

# Shared colorbar
cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
norm = plt.Normalize(vmin=0, vmax=400)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), cax=cbar_ax)
cbar.set_label('Timesteps')

plt.tight_layout(rect=[0.06, 0.03, 0.9, 0.97])  # leave room for labels
plt.savefig('test_combined.png')
plt.show()