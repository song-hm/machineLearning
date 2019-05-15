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
xArray = data[:,0]
yReal = data[:,-1]
learningRate = 0.00001

#梯度下降的函数
def grandentdecent():
    bslop = 0
    for index,x in enumerate(xArray):
        bslop = bslop + (m*x+b - yReal[index])
    bslop = bslop*2/len(xArray)
    # print("mse对b求导={}".format(bslop))

    mslop = 0
    for index, x in enumerate(xArray):
        mslop = mslop + (m * x + b - yReal[index])*x
    mslop = mslop * 2 / len(xArray)
    # print("mse对m求导={}".format(mslop))
    return (bslop,mslop)
#训练
def train():
    startTime = time.time()
    for i in range(1,10000000):
        bslop, mslop = grandentdecent()
        global m
        m = m - mslop*learningRate
        global b
        b = b - bslop*learningRate
        if (abs(mslop)<0.5 and abs(bslop)<0.5):
            break
    endTime = time.time()
    print("m={},b={}".format(m,b))
    print("消耗的时间={}".format(endTime - startTime))
if __name__ == '__main__':
    train()


# m=1.2144708175984846,b=112.97991644806174
# 消耗的时间=87.97550368309021

# weight=[[  1.21447082]
#  [112.97991645]]
# 消耗的时间=52.911609411239624