import numpy as np

#数据标准化
def dataStandardization(data):
    for i in range(0, len(data.T) - 1):
        feature = data.T[i]
        mean = np.mean(feature)
        std = np.std(feature)
        data.T[i] = ((feature - mean) / std)
    return data

data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,5,1))
data = dataStandardization(data)
m1 = 1
m2 = 1
b = 1
xArray = data[:,0:2]
yReal = data[:,-1]
learningRate = 0.00001

#梯度下降的函数
def grandentdecent():
    bslop = 0
    for index,x in enumerate(xArray):
        bslop = bslop + (m1*x[0] + m2*x[1] + b - yReal[index])
    bslop = bslop*2/len(xArray)
    print("mse对b求导={}".format(bslop))

    m1slop = 0
    for index, x in enumerate(xArray):
        m1slop = m1slop + (m1*x[0] + m2*x[1] + b - yReal[index])*x[0]
    m1slop = m1slop * 2 / len(xArray)
    # print("mse对m求导={}".format(mslop))

    m2slop = 0
    for index, x in enumerate(xArray):
        m2slop = m2slop + (m1*x[0] + m2*x[1] + b - yReal[index])*x[1]
    m2slop = m2slop * 2 / len(xArray)
    # print("mse对m求导={}".format(mslop))
    return (bslop,m1slop,m2slop)

# 训练多次运行梯度下降算法
def train():
    for i in range(1,10000000):
        bslop, m1slop, m2slop = grandentdecent()
        global m1
        m1 = m1 - m1slop*learningRate
        global m2
        m2 = m2 - m2slop*learningRate
        global b
        b = b - bslop*learningRate
        if (abs(m1slop)<0.5 and abs(bslop)<0.5 and abs(m2slop)<0.5):
            break
    print("m1={},m2={},b={}".format(m1,m2,b))

if __name__ == '__main__':
    train()
