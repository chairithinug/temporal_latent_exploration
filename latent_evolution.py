import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

epochs = 23
sequence_len = 401

latents = np.empty((epochs+1, sequence_len, 2))

for i in range(-1, epochs):
    latents[i+1] = np.load(f'latent_evolution/{i}_True_False.npy').squeeze()[...,0]

# Standard scaling all data at once for consistency
# sc = StandardScaler()
# all_data = latents.reshape(-1, 2)
# sc.fit(all_data)
# latents = sc.transform(all_data).reshape(epochs, sequence_len, 2)


# After loading and scaling latents
x_min = np.min(latents[..., 0])
x_max = np.max(latents[..., 0])
y_min = np.min(latents[..., 1])
y_max = np.max(latents[..., 1])

# Add a bit of margin (optional)
margin = 0.1  # or more depending on density
x_range = (x_min - margin, x_max + margin)
y_range = (y_min - margin, y_max + margin)


from scipy.interpolate import interp1d

# interp_steps = 5  # Number of extra frames between two real frames
# # Interpolated data container
# smooth_latents = []

# for i in range(len(latents) - 1):
#     start = latents[i]
#     end = latents[i + 1]

#     # Linear interpolation
#     for t in np.linspace(0, 1, interp_steps, endpoint=False):
#         interpolated = start * (1 - t) + end * t
#         smooth_latents.append(interpolated)

# # Add the final frame explicitly
# smooth_latents.append(latents[-1])
# smooth_latents = np.array(smooth_latents)  # shape: (new_frames, num_points, 2)

T, N, D = latents.shape  # epochs, num_points, dim
interp_steps = 10
pause_frames = 5

# Precompute: total frames per pair = pause_frames + interp_steps
frames_per_segment = pause_frames + interp_steps
total_frames = (T - 1) * frames_per_segment + pause_frames

# Preallocate arrays
smooth_latents = np.empty((total_frames, N, D), dtype=latents.dtype)
frame_labels = np.empty((total_frames,), dtype=np.float32)

idx = 0
for i in tqdm(range(T - 1)):
    start = latents[i]
    end = latents[i + 1]

    # Pause frames (repeat start)
    smooth_latents[idx:idx+pause_frames] = start
    frame_labels[idx:idx+pause_frames] = float(i)
    idx += pause_frames

    # Interpolation frames
    t_values = np.linspace(0, 1, interp_steps, endpoint=False)
    interp = start[None] * (1 - t_values)[:, None, None] + end[None] * t_values[:, None, None]
    smooth_latents[idx:idx+interp_steps] = interp
    frame_labels[idx:idx+interp_steps] = i + t_values
    idx += interp_steps

# Final pause at last frame
smooth_latents[idx:idx+pause_frames] = latents[-1]
frame_labels[idx:idx+pause_frames] = float(T - 1)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
ax.grid(True)
scat = ax.scatter([], [], s=50, c=[], cmap="rainbow", alpha=0.7)
ax.set_xlim(*x_range)
ax.set_ylim(*y_range)
ax.set_xlabel('C_0')
ax.set_ylabel('C_1')
main_title = ax.set_title("latent@dim=2 of Trajectory 90 (Test)")

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
norm = plt.Normalize(0, sequence_len)
sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
fig.colorbar(sm, cax=cax, orientation="vertical")
cbar = sm

texts = [ax.text(0, 0, '', fontsize=8, ha='center', va='center') for _ in range(sequence_len)]

# --- Animation ---
def init():
    scat.set_offsets(np.empty((0, 2)))
    scat.set_array(np.array([]))
    for txt in texts:
        txt.set_text('')
        txt.set_position((0, 0))
    return scat,

# def update(frame):
#     # data = latents[frame]
#     data = smooth_latents[frame]
#     scat.set_offsets(data)
#     scat.set_array(np.arange(sequence_len))  # colors
#     main_title.set_text(f"latent@dim=2 of Trajectory 90 (Test) - Time={frame//interp_steps}, Interp={frame%interp_steps}")
#     return scat,
def update(frame):
    data = smooth_latents[frame]
    current_label = frame_labels[frame]
    
    scat.set_offsets(data)
    scat.set_array(np.arange(len(data)))
    main_title.set_text(f"latent@dim=2 - Step {current_label:.2f}")
    for i, txt in enumerate(texts):
        x, y = data[i]
        if i % 10 == 0:
            txt.set_text(str(i))
            txt.set_position((x, y + 0.02))
            txt.set_fontsize(6)
            txt.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        else:
            txt.set_text('')
    return scat, main_title

#ani = animation.FuncAnimation(fig, update, frames=len(smooth_latents), init_func=init, blit=True, repeat=False)
ani = animation.FuncAnimation(
    fig, update,
    frames=len(smooth_latents),
    init_func=init,
    blit=True,
    interval=11.111111,  # 50 ms per frame = 20 fps
)

ani.save('tracking_animation_new.gif', writer='PillowWriter', fps=90)