import numpy as np

data = np.loadtxt("kc_house_data.csv",delimiter=",",skiprows=1,usecols=(5,17,18,2))
np.random.shuffle(data)
testdata = data[0:1960]
traindata = data[1960:-1]
#保存测试集
np.savetxt("kc_house_testdata.csv",testdata,delimiter=",",fmt="%.4f")
#保存训练集
np.savetxt("kc_house_traindata.csv",traindata,delimiter=",",fmt="%.4f")