#抽取一个函数
import numpy as np
import collections as c

def knn(k,predictPoint,feature,label):
    # 计算每个投掷点与predictPoint的距离
    distance = list(map(lambda x: abs(predictPoint - x), feature))
    # 对distance的集合元素从小到大排列（返回元素排序的下标位置）
    sortindex = (np.argsort(distance))
    # 用排序的sortindex操作label集合
    sortedlabel = (label[sortindex])
    # knn算法的k取最近的三个邻居
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

if __name__ == '__main__':
    traindata = np.loadtxt("data0-train.csv", delimiter=",")
    # 输入值
    feature = (traindata[:, 0])
    # 结果label ，-1表示取最后一列
    label = traindata[:, -1]
    # 预测
    for k in range(1,100):
        count = 0
        testdata = np.loadtxt("data0-test.csv", delimiter=",")
        for item in testdata:
            predict = knn(k, item[0], feature, label)
            real = item[1]
            if predict == real:
                count = count + 1
        print("k={},准确率：{}%".format(k,count * 100.0 / len(testdata)))
#经验k值的选择是数据量的开平方
print(knn(36,200,feature,label))