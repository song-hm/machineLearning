#处理数据，把数据打散，拆分成训练集和测试集
import numpy as np

data = np.loadtxt("data0.csv",delimiter=",")
np.random.shuffle(data)
#测试集
testdata = data[0:100]
#训练集
traindata = data[100:-1]
#保存测试集
np.savetxt("data0-test.csv",testdata,delimiter=",",fmt="%d")
#保存训练集
np.savetxt("data0-train.csv",traindata,delimiter=",",fmt="%d")