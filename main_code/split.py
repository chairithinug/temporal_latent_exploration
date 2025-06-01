import numpy as np

raw = np.load('./dataset/rawData.npy', allow_pickle=True)
gt = np.load('./latent/test_data.npy', allow_pickle=True)
print(raw.keys())
print([(k, raw[k].shape) for k in raw.keys()])
print(raw['x'].shape)
print(gt.shape)
print(np.allclose(gt,raw['x'][90:]))

np.save('./latent/edge_index.npy', raw['edge_index'])
np.save('./latent/edge_attr.npy', raw['edge_attr'])
# np.save('./latent/raw_target_dataset.npy', raw['x'])
