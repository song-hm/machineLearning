import numpy as np
#房价预测 经度、纬度、房屋使用面积、房价

def knn(k,predictPoint,feature,label):
    matrixTemp = (feature - predictPoint)
    matrixTemp2 = (np.square(matrixTemp))
    # axis=1 表示逐行相加
    sortIndex = np.argsort(np.sum(matrixTemp2, axis=1))
    sortLabel = label[sortIndex]
    predictPrice = np.sum(sortLabel[0:k]) / k
    return predictPrice

#数据标准化
def dataStandardization(data):
    for i in range(0, len(data.T) - 1):
        feature = data.T[i]
        mean = np.mean(feature)
        std = np.std(feature)
        data.T[i] = ((feature - mean) / std)
    return data


if __name__ == '__main__':
    traindata = np.loadtxt("kc_house_traindata.csv",delimiter=",")
    traindata = dataStandardization(traindata)
    feature = traindata[:,0:3]
    label = traindata[:,-1]
    k = 140
    count = 0
    testdata = dataStandardization(np.loadtxt("kc_house_traindata.csv",delimiter=","))
    for item in testdata:
        predictPrice = knn(k,item[0:3],feature,label)
        realPrice = item[-1]
        if (abs(predictPrice - realPrice))<1000:
            count = count + 1
    print("准确率={}%".format(count * 100.0 / len(testdata)))