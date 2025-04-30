import os

# Paths to your files
file_paths = {
    'latent_l2_3': 'latent/latent_l2_3.npy',
    'latent_l16': 'latent/latent_l16.npy',
    'latent_l32': 'latent/latent_l32.npy',
    'latent_l64': 'latent/latent_l64.npy',
    'latent_l128': 'latent/latent_l128.npy',
    'latent_l256': 'latent/latent_l256.npy',
    'test_data': 'latent/test_data.npy'  # original uncompressed
}

# Measure file sizes
file_sizes = {name: os.path.getsize(path) for name, path in file_paths.items()}

# Get the original (uncompressed) size
original_size = file_sizes['test_data']

print(f"Original file (test_data) size: {original_size} Bytes\n")

# Print header
print(f"{'Latent Name':<15} {'Size (Bytes)':<15} {'% of Original':<15} {'Compression Factor':<20}")
print("-" * 65)

# Compute and print results
for name, size in file_sizes.items():
    percent = (size / original_size) * 100
    factor = original_size / size
    print(f"{name:<15} {size:<15} {percent:<15.3f} {factor:<20.2f}")

# import numpy as np

# latent = np.load('dataset/rawData.npy', allow_pickle=True)
# np.save(f'latent/test_data.npy', latent['x'][90:])
# print(latent['x'][90:].shape)