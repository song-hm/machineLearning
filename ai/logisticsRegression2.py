import numpy as np
# 逻辑回归 预测
weight = np.array([
    [0.96926278],
    [-19.12845077],
])

feature = np.array([
    [20,1]
])

print(1 / (1 + np.exp(-np.dot(feature, weight))))