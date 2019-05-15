import numpy as np
#房价预测
def knn(k,predictPoint,feature,label):
    matrixTemp = (feature - predictPoint)
    matrixTemp2 = (np.square(matrixTemp))
    # axis=1 表示逐行相加
    sortIndex = np.argsort(np.sum(matrixTemp2, axis=1))
    sortLabel = label[sortIndex]
    predictPrice = np.sum(sortLabel[0:k]) / k
    return ("预测的房价{}万".format(predictPrice))
if __name__ == '__main__':
    feature = np.loadtxt("kc_house_data.csv",delimiter=",",skiprows=1,usecols=(17,18,6))
    label = np.loadtxt("kc_house_data.csv",delimiter=",",skiprows=1,usecols=(2))
    predictPoint = np.array([47.5112,-122.257,5650])
    print(knn(450, predictPoint, feature, label))
