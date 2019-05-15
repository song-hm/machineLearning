import numpy as np
# 进行预测

# 模型
# m1 = -0.1025721
# m2 = -4.27357477
# b = 40.29786641

def calc(featureMatrix,weight):
    return np.dot(featureMatrix,weight)


# 评估模型准确性
if __name__ == '__main__':
    testdata = np.loadtxt("cars-test.csv",delimiter=",")
    feature = testdata[:,0:2]
    featureMatrix = np.append(feature,np.ones(shape=(len(feature),1)),axis=1)
    label = testdata[:,-1:]

    weight = np.array([
        [-0.1025721],
        [-4.27357477],
        [40.29786641],
    ])
    result = calc(featureMatrix,weight)
    error = (label - result)/label
    totalError = np.sum(abs(error))/len(error)
    print("错误率={}".format(totalError))

# 错误率=0.15613893091963713