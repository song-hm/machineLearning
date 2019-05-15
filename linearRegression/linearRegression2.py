import numpy as np

data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,1))
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
    print("mse对b求导={}".format(bslop))

    mslop = 0
    for index, x in enumerate(xArray):
        mslop = mslop + (m * x + b - yReal[index])*x
    mslop = mslop * 2 / len(xArray)
    # print("mse对m求导={}".format(mslop))
    return (bslop,mslop)
#训练 多次运行梯度下降算法
def train():
    for i in range(1,10000000):
        bslop, mslop = grandentdecent()
        global m
        m = m - mslop*learningRate
        global b
        b = b - bslop*learningRate
        if (abs(mslop)<0.5 and abs(bslop)<0.5):
            break
    print("m={},b={}".format(m,b))

if __name__ == '__main__':
    train()
