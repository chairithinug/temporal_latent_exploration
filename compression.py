import os

MB = 1024 ** 2  # Conversion factor

# Paths to your files
file_paths = {
    'latent_l2': 'latent/latent_l2best.npy',
    'latent_l16': 'latent/latent_l16best.npy',
    'latent_l32': 'latent/latent_l32best.npy',
    'latent_l64': 'latent/latent_l64best.npy',
    'latent_l128': 'latent/latent_l128best.npy',
    'latent_l256': 'latent/latent_l256best.npy',
    'test_data': 'latent/test_data.npy'
}

file_sizes = {name: os.path.getsize(path) / MB for name, path in file_paths.items()}
original_size = file_sizes['test_data']

print(f"Original file (test_data) size: {original_size:.2f} MB\n")

print(f"{'Latent Name':<15} {'Size (MB)':<15} {'% of Original':<15} {'Compression Factor':<20}")
print("-" * 65)

for name, size in file_sizes.items():
    percent = (size / original_size) * 100
    factor = original_size / size
    print(f"{name:<15} {size:<15.3f} {percent:<15.3f} {factor:<20.2f}")

# Model and mesh/pivot sizes
model_paths = {
    'latent_l2': 'checkpoints/best/model_l2_best.pt/Mesh_Reduced.0.292.pt',
    'latent_l16': 'checkpoints/best/model_l16_best.pt/Mesh_Reduced.0.250.pt',
    'latent_l32': 'checkpoints/best/model_l32_best.pt/Mesh_Reduced.0.298.pt',
    'latent_l64': 'checkpoints/best/model_l64_best.pt/Mesh_Reduced.0.298.pt',
    'latent_l128': 'checkpoints/best/model_l128_best.pt/Mesh_Reduced.0.281.pt',
    'latent_l256': 'checkpoints/best/model_l256_best.pt/Mesh_Reduced.0.279.pt',
    'test_data': 'dummy'
}

model_sizes = {name: (os.path.getsize(path) / MB if os.path.exists(path) else 0) for name, path in model_paths.items()}
meshpos_size = os.path.getsize('dataset/meshPosition_all.txt') / MB

pivotpos_paths = {
    'latent_l2': 'dataset/meshPosition_pivotal_l2.txt',
    'latent_l16': 'dataset/meshPosition_pivotal_l16.txt',
    'latent_l32': 'dataset/meshPosition_pivotal_l32.txt',
    'latent_l64': 'dataset/meshPosition_pivotal_l64.txt',
    'latent_l128': 'dataset/meshPosition_pivotal_l128.txt',
    'latent_l256': 'dataset/meshPosition_pivotal_l256.txt',
    'test_data': 'dummy'
}

pivot_sizes = {name: (os.path.getsize(path) / MB if os.path.exists(path) else 0) for name, path in pivotpos_paths.items()}

print(f"\nOriginal file (test_data + meshPosition_all) size: {original_size + meshpos_size:.2f} MB\n")

print(f"{'Latent Name':<15} {'Size (MB)':<15} {'% of Original':<15} {'Compression Factor':<20}")
print("-" * 65)

original_size_total = original_size + meshpos_size

for name in file_sizes.keys():
    size_total = file_sizes[name] + model_sizes[name] + pivot_sizes[name] + meshpos_size
    percent = (size_total / original_size_total) * 100
    factor = original_size_total / size_total
    print(f"{name:<15} {size_total:<15.3f} {percent:<15.3f} {factor:<20.2f}")

# Create Markdown report
markdown_lines = []

markdown_lines.append(f"# Compression Report\n")
markdown_lines.append(f"Original file (test_data + meshPosition_all): **{original_size_total:.2f} MB**\n")
markdown_lines.append("## Summary Table\n")
markdown_lines.append("| Latent Name   | Size (MB)     | % of Original | Compression Factor |")
markdown_lines.append("|---------------|---------------|----------------|---------------------|")

for name in file_sizes.keys():
    size_total = file_sizes[name] + model_sizes[name] + pivot_sizes[name] + meshpos_size
    percent = (size_total / original_size_total) * 100
    factor = original_size_total / size_total
    markdown_lines.append(f"| {name:<13} | {size_total:<13.3f} | {percent:<14.3f} | {factor:<19.2f} |")

# Save to .md file
with open("compression_report.md", "w") as f:
    f.write("\n".join(markdown_lines))

print("\n✅ Markdown report saved to `compression_report.md`")