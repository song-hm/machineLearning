import numpy as np
# 数据的标准化
data = np.array([
    [1,2],
    [3,4],
    [5,6]
])

mean = np.mean(data[:,1])
std = np.std(data[:,1])
print((data[:,1]-mean)/std)