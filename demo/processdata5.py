import numpy as np
# 数据的标准化
data = np.array([5,5,5,5,5,5])

mean = np.mean(data)
print(mean)
std = np.std(data)
print((data-mean)/std)

#   print((data-mean)/std)
# [nan nan nan nan nan nan]
# nan错误 1、除数为零 2、数字过大