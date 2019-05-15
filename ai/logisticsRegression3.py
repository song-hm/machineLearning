import numpy as np
#多分类问题逻辑回归
data = np.array([
    [5,1,0,0],
    [15,1,0,0],
    [25,0,1,0],
    [35,0,1,0],
    [45,0,1,0],
    [55,0,0,1],
    [65,0,0,1],
])

m = 1
b = 1
weight = np.array([
    [m,m,m],
    [b,b,b],
])
learningRate = 0.001
feature = data[:,0:1]
featureMatrix = np.append(feature,np.ones(shape=(len(data),1)),axis=1)
label = data[:,1:]

def gradentDecent():
    #sigmoid函数，激活函数
    predict = 1 / (1 + np.exp(-np.dot(featureMatrix, weight)))
    return np.dot(featureMatrix.T, predict - label)

def train():
    for i in range(1,1000000):
        slop = gradentDecent()
        global weight
        weight = weight - learningRate*slop
    print(weight)

if __name__ == '__main__':
    train()

#看电视
# [[-0.9697223 ]
#  [19.13764586]]

#读书
# [[ 0.02997379]
#  [-0.15720643]]

#广场舞
# [[ 0.60928238]
#  [-30.35856277]]

# [[-9.69722297e-01  2.99737907e-02  6.09282382e-01]
#  [ 1.91376459e+01 -1.57206430e-01 -3.03585628e+01]]