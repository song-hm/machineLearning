import numpy as np
import time
# data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,1))
data = np.array([
   [80,200],
   [95,230],
   [104,245],
   [112,274],
   [125,259],
   [135,262]
])
m = 1
b = 1
weight = np.array([
    [m],
    [b],
])
feature = data[:,0:1] # 保留了维度信息的feature
featureMatrix = np.append(feature, np.ones(shape=(6, 1)),axis=1)
label = np.expand_dims(data[:, -1], axis=1) # 保留了维度信息的label
learningRate = 0.00001

#梯度下降的函数
def grandentdecent():
    result = np.dot(featureMatrix.T,np.dot(featureMatrix,weight) - label)/len(featureMatrix)*2
    return result #结果矩阵 第0行第0列是对m的偏导，第0行第1列是对b的偏导
#训练
def train():
    startTime = time.time()
    for i in range(1,10000000):
        result = grandentdecent()
        # print(result)
        global weight
        weight = weight - result*learningRate
        if (abs(result[0][0])<0.5 and abs(result[1][0])<0.5):
            break
    endTime = time.time()
    print("weight={}".format(weight))
    print("消耗的时间={}".format(endTime-startTime))


if __name__ == '__main__':
    train()
