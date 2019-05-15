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
    data = np.loadtxt("data0.csv", delimiter=",")
    # 输入值
    feature = (data[:, 0])
    # 结果label ，-1表示取最后一列
    label = data[:, -1]
    # 预测点
    predictPoint = 500
    for k in range(1,100):
        print("k={},预计落入{}".format(k,knn(k,predictPoint,feature,label)))

