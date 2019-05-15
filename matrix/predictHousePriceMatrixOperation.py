import numpy as np

data = np.array([
   [80,200],
   [95,230],
   [104,245],
   [112,274],
   [125,259],
   [135,262],
])
# 纬度扩展
# print(np.expand_dims(data[:, 0], axis=1))
# print(data[:, 0:1])

feature = data[:,0:1]
# label = data[:,-1:]
label = np.expand_dims(data[:, -1], axis=1)
m = 1
b = 1
weight = np.array([
    [m],
    [b],
])
featureMatrix = np.append(feature, np.ones(shape=(6, 1)),axis=1)
dMatrix = np.dot(featureMatrix, weight) - label
print(np.dot(featureMatrix.T, dMatrix) * 2 / len(featureMatrix))