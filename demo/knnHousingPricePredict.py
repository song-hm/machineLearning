import numpy as np
#房价预测
feature = np.array([
    [-121,47],
    [-121.2, 46.5],
    [-122, 46.3],
    [-120.9, 46.7],
    [-120.1, 46.2]
])
label = np.array([
    200,210,250,215,232
])

predictPoint = np.array([-121,46])
matrixTemp = (feature - predictPoint)
matrixTemp2 = (np.square(matrixTemp))
#axis=1 表示逐行相加
sortIndex = np.argsort(np.sum(matrixTemp2, axis=1))
sortLabel = label[sortIndex]
k = 3
predictPrice = np.sum(sortLabel[0:k])/k
print("预测的房价{}万".format(predictPrice))

# [[ 0.   1. ]
#  [-0.2  0.5]
#  [-1.   0.3]
#  [ 0.1  0.7]
#  [ 0.9  0.2]]