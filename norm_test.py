import numpy as np

x = np.load('predict/predict_l2cosinebest.npy')

print(np.linalg.norm(x[0,0]))
