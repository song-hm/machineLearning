import numpy as np
import collections as c
data = np.array([
        [154,1],
        [126,2],
        [70,2],
        [196,2],
        [161,2],
        [371,4]
])
# 输入值
feature = (data[:,0])
# 结果label ，-1表示取最后一列
label = data[:,-1]
# 预测点
predictPoint = 200
# 计算每个投掷点与predictPoint的距离
distance = list(map(lambda x:abs(predictPoint-x),feature))
#对distance的集合元素从小到大排列（返回元素排序的下标位置）
sortindex = (np.argsort(distance))
#用排序的sortindex操作label集合
sortedlabel = (label[sortindex])
#knn算法的k取最近的三个邻居
k = 3
print(c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

