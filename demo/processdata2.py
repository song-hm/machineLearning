import numpy as np
def color2num(str):
    dict = {"红":0.50,"黄":0.51,"蓝":0.52,"绿":0.53,"紫":0.54,"粉":0.55}
    return dict[str]
data = np.loadtxt("data1.csv",delimiter=",",converters={1:color2num},encoding="gbk")
np.savetxt("data2.csv",data,delimiter=",",fmt="%.2f")
data2 = np.loadtxt("data2.csv",delimiter=",")
np.random.shuffle(data2)
testdata = data2[0:100]
traindata = data2[100:-1]
#保存测试集
np.savetxt("data2-test.csv",testdata,delimiter=",",fmt="%.2f")
#保存训练集
np.savetxt("data2-train.csv",traindata,delimiter=",",fmt="%.2f")