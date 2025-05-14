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

model_paths = {
    'latent_l2_3': 'checkpoints/final/model_l2_3layers_best.pt/Mesh_Reduced.0.285.pt',
    'latent_l16': 'checkpoints/final/model_l16_best.pt/Mesh_Reduced.0.250.pt',
    'latent_l32': 'checkpoints/final/model_l32_best.pt/Mesh_Reduced.0.298.pt',
    'latent_l64': 'checkpoints/final/model_l64_best.pt/Mesh_Reduced.0.298.pt',
    'latent_l128': 'checkpoints/final/model_l128_best.pt/Mesh_Reduced.0.270.pt',
    'latent_l256': 'checkpoints/final/model_l256_best.pt/Mesh_Reduced.0.220.pt',
    'test_data': 'dummy'  # original uncompressed
}

model_sizes = {name: os.path.getsize(path) for name, path in model_paths.items()}

meshpos_paths = 'dataset/meshPosition_all.txt'

meshpos_size = os.path.getsize(meshpos_paths)

pivotpos_paths = {
    'latent_l2_3': 'dataset/meshPosition_pivotal_l2.txt',
    'latent_l16': 'dataset/meshPosition_pivotal_l16.txt',
    'latent_l32': 'dataset/meshPosition_pivotal_l32.txt',
    'latent_l64': 'dataset/meshPosition_pivotal_l64.txt',
    'latent_l128': 'dataset/meshPosition_pivotal_l128.txt',
    'latent_l256': 'dataset/meshPosition_pivotal_l256.txt',
    'test_data': 'dummy'  # original uncompressed
}

pivot_sizes = {name: os.path.getsize(path) for name, path in pivotpos_paths.items()}

# print(model_sizes)
# print(meshpos_size)
# print(pivot_sizes)

print(f"Original file (test_data) size: {original_size} Bytes\n")

# Print header
print(f"{'Latent Name':<15} {'Size (Bytes)':<15} {'% of Original':<15} {'Compression Factor':<20}")
print("-" * 65)

original_size += meshpos_size
# Compute and print results
for name in file_sizes.keys():
    size = file_sizes[name]
    size += model_sizes[name]
    size += pivot_sizes[name]
    size += meshpos_size
    percent = (size / original_size) * 100
    factor = original_size / size
    print(f"{name:<15} {size:<15} {percent:<15.3f} {factor:<20.2f}")