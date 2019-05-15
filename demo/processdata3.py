import numpy as np
#数据归一化
data = np.array([23,12,35,111,36,44,67,74])
print((data-np.min(data))/(np.max(data)-np.min(data)))