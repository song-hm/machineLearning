#抽取一个函数  classification 分类问题 plinko游戏
import numpy as np
import collections as c
#考虑的是一维的数据，（position） feature[130,126,25...]
def knn(k,predictPoint,feature,label):
    # 计算每个投掷点与predictPoint的距离
    distance = list(map(lambda x: abs(predictPoint - x), feature))
    # 对distance的集合元素从小到大排列（返回元素排序的下标位置）
    sortindex = (np.argsort(distance))
    # 用排序的sortindex操作label集合
    sortedlabel = (label[sortindex])
    # knn算法的k取最近的三个邻居
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

#二维空间，feature[ [135,0.53] ,[126,0.51] ,[25,0.52] ]
def knn2(k,predictPoint,ballcolor,feature,lable):
    # 计算每个投掷点与（predictPoint，ballcolor)的距离
    distance = list(map(lambda item:((item[0]-predictPoint)**2 + (item[1]-ballcolor)**2)**0.5,feature))
    # 对distance的集合元素从小到大排列（返回元素排序的下标位置）
    sortindex = (np.argsort(distance))
    # 用排序的sortindex操作label集合
    sortedlabel = (label[sortindex])
    # knn算法的k取最近的三个邻居
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

#二维空间，feature[ [135,0.53] ,[126,0.51] ,[25,0.52] ] 数据归一化的knn
def knn3(k,predictPoint,ballcolor,feature,lable):
    # 计算每个投掷点与（predictPoint，ballcolor)的距离
    distance = list(map(lambda item:((item[0]/475-predictPoint/475)**2 + ((item[1]-0.50)/0.05-(ballcolor-0.50)/0.05)**2)**0.5,feature))
    # 对distance的集合元素从小到大排列（返回元素排序的下标位置）
    sortindex = (np.argsort(distance))
    # 用排序的sortindex操作label集合
    sortedlabel = (label[sortindex])
    # knn算法的k取最近的三个邻居
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

if __name__ == '__main__':
    traindata = np.loadtxt("data2-train.csv", delimiter=",")
    # 输入值
    feature = (traindata[:, 0:2])
    # 结果label ，-1表示取最后一列
    label = traindata[:, -1]
    # 预测
    #经验k值的选择是数据量的开平方
    k = 36
    count = 0
    testdata = np.loadtxt("data2-test.csv",delimiter=",")
    for item in testdata:
        predict = knn3(k,item[0],item[1],feature,label)
        real = item[-1]
        if predict == real:
            count = count + 1
    print("准确率={}%".format(count*100.0/len(testdata)))