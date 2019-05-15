import numpy as np
# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
# np.set_printoptions(threshold = np.inf)
# 若想不以科学计数显示:
np.set_printoptions(suppress = True)


# 手写数字识别-训练
# def filter(str,num):
#     if int(str.decode()) == num:
#         return 1
#     else:
#         return 0

def filter0(str):
    if int(str.decode()) == 0:
        return 1
    else:
        return 0
def filter1(str):
    if int(str.decode()) == 1:
        return 1
    else:
        return 0

def filter2(str):
    if int(str.decode()) == 2:
        return 1
    else:
        return 0

def filter3(str):
    if int(str.decode()) == 3:
        return 1
    else:
        return 0

def filter4(str):
    if int(str.decode()) == 4:
        return 1
    else:
        return 0
def filter5(str):
    if int(str.decode()) == 5:
        return 1
    else:
        return 0
def filter6(str):
    if int(str.decode()) == 6:
        return 1
    else:
        return 0
def filter7(str):
    if int(str.decode()) == 7:
        return 1
    else:
        return 0
def filter8(str):
    if int(str.decode()) == 8:
        return 1
    else:
        return 0
def filter9(str):
    if int(str.decode()) == 9:
        return 1
    else:
        return 0

#注意归一化
feature = np.loadtxt("train_image.csv",delimiter=",",max_rows=3000)/255
featureMatrix = np.append(feature,np.ones(shape=(len(feature),1)),axis=1)
weight = np.ones(shape=(785,10))
learningRate = 0.3
#获取label
label0 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter0})
label1 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter1})
label2 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter2})
label3 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter3})
label4 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter4})
label5 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter5})
label6 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter6})
label7 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter7})
label8 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter8})
label9 = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter9})
labelMatrix = np.array([
    label0,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
])

# labelMatrix = np.append(np.append(np.append(np.expand_dims(label0,axis=1),np.expand_dims(label1,axis=1),axis=1),np.expand_dims(label2,axis=1)),np.expand_dims(label3,axis=1))

def gradentDecent():
    predict = 1 / (1 + np.exp(-np.dot(featureMatrix, weight)))
    slop = np.dot(featureMatrix.T, predict - labelMatrix.T) / len(feature)
    return slop

def train():
    for i in range(1,5000):
        slop = gradentDecent()
        global weight
        weight = weight - slop*learningRate
        # print(slop.T)
    return weight

if __name__ == '__main__':
    #手写数字训练
    weight = train()
    #手写数字测试
    testfeature = np.loadtxt("test_image.csv", delimiter=",", max_rows=50) / 255
    testfeatureMatrix = np.append(testfeature, np.ones(shape=(len(testfeature), 1)), axis=1)
    predict = 1 / (1 + np.exp(-np.dot(testfeatureMatrix,weight)))
    print(predict)
    # label = np.loadtxt("train_labels.csv", delimiter=",", max_rows=5, converters={0: filter})