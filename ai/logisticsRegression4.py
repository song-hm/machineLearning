import numpy as np
# 逻辑回归 预测
#看电视的权重
weight1 = np.array([
    [-0.9697223],
    [19.13764586],
])
#读书的权重
weight2 = np.array([
    [0.02997379],
    [-0.15720643],
])
#广场舞的权重
weight3 = np.array([
    [0.60928238],
    [-30.35856277],
])

weight = np.array([
    [-9.69722297e-01 , 2.99737907e-02 , 6.09282382e-01],
    [1.91376459e+01 ,-1.57206430e-01 ,-3.03585628e+01],
])
feature = np.array([
    [96,1]
])
# tv = np.dot(feature, weight1)
# book = np.dot(feature, weight2)
# dance = np.dot(feature, weight3)
predict = np.dot(feature,weight)
print(predict)
# softMax函数，计算多个分类的概率
print("看电视")
# print(np.exp(tv)/(np.exp(tv)+np.exp(book)+np.exp(dance)))
print(np.exp(predict[0][0])/(np.exp(predict[0][0])+np.exp(predict[0][1])+np.exp(predict[0][2])))

print("看书")
# print(np.exp(book)/(np.exp(tv)+np.exp(book)+np.exp(dance)))
print(np.exp(predict[0][1])/(np.exp(predict[0][0])+np.exp(predict[0][1])+np.exp(predict[0][2])))

print(("广场舞"))
# print(np.exp(dance)/(np.exp(tv)+np.exp(book)+np.exp(dance)))
print(np.exp(predict[0][2])/(np.exp(predict[0][0])+np.exp(predict[0][1])+np.exp(predict[0][2])))
