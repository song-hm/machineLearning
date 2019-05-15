import numpy as np
# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
# np.set_printoptions(threshold = np.inf)
# 若想不以科学计数显示:
np.set_printoptions(suppress = True)


# 手写数字识别-训练
def filter(str):
    # if str == b'4':
    if int(str.decode()) == 4:
        return 1
    else:
        return 0

#注意归一化
feature = np.loadtxt("train_image.csv",delimiter=",",max_rows=3000)/255
featureMatrix = np.append(feature,np.ones(shape=(len(feature),1)),axis=1)
weight = np.ones(shape=(785,1))
learningRate = 0.3
#获取label 假设识别数字4
label = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter})
labelMatrix = np.expand_dims(label,axis=1)

def gradentDecent():
    predict = 1 / (1 + np.exp(-np.dot(featureMatrix, weight)))
    slop = np.dot(featureMatrix.T, predict - labelMatrix) / len(label)
    return slop

def train():
    for i in range(1,5000):
        slop = gradentDecent()
        global weight
        weight = weight - slop*learningRate
        # print(slop.T)
    return weight

if __name__ == '__main__':
    #训练
    weight = train()
    #测试
    testfeature = np.loadtxt("test_image.csv", delimiter=",", max_rows=20) / 255
    testfeatureMatrix = np.append(testfeature, np.ones(shape=(len(testfeature), 1)), axis=1)
    predict = 1 / (1 + np.exp(-np.dot(testfeatureMatrix,weight)))
    print(predict)
    # label = np.loadtxt("train_labels.csv", delimiter=",", max_rows=5, converters={0: filter})