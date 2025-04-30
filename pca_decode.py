import numpy as np
import torch
from tqdm import tqdm

import os
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

def test(x, y, criterion=torch.nn.MSELoss()):
    loss = criterion(y, x)
    relative_error = (
        loss / criterion(y, y * 0.0).detach()
    )
    relative_error_s_record = []
    for i in range(3):
        loss_s = criterion(y[:, i], x[:, i])
        relative_error_s = (
            loss_s
            / criterion(y[:, i], y[:, i] * 0.0).detach()
        )
        relative_error_s_record.append(relative_error_s.cpu())
    return loss, relative_error, relative_error_s_record

raw = np.load('./dataset/rawData.npy', allow_pickle=True)

latent_train = raw['x'][:80] # shape = 80, 401, 1699, 3
latent_test = raw['x'][90:] # shape = 11, 401, 1699, 3
loss_total = 0
relative_error_total = 0

reshaped_train = latent_train.reshape(-1, 1699 * 3)
reshaped_test = latent_test.reshape(-1, 1699 * 3)
# print(np.allclose(reshaped.reshape(11, 401, 1699, 3), latent))
sc = StandardScaler()

reshaped_train = sc.fit_transform(reshaped_train)
reshaped_test = sc.transform(reshaped_test)

reducer = PCA(n_components=2)
reducer.fit(reshaped_train)
x_reduced = reducer.transform(reshaped_test)
x_recon = reducer.inverse_transform(x_reduced)
x_recon = x_recon.reshape(11, 401, 1699, 3)
reshaped_test = reshaped_test.reshape(11, 401, 1699, 3)
relative_error_s_total = []
for j, traj in enumerate(x_recon):
    for k, step in enumerate(traj):
        step = sc.inverse_transform(step.reshape(-1, 1699 * 3))    
        step = torch.from_numpy(step)
        step = step.reshape(1699, 3)
        loss, relative_error, relative_error_s = test(step, torch.from_numpy(latent_test[j, k]))
        relative_error_s_total.append(relative_error_s)
        loss_total = loss_total + loss
        relative_error_total = relative_error_total + relative_error

n =  11 * 401
avg_relative_error = relative_error_total / n
avg_loss = loss_total / n
avg_relative_error_s = np.array(relative_error_s_total).mean(axis=0)

print(avg_loss, avg_relative_error, avg_relative_error_s)