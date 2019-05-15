import numpy as np
# 进行预测

# 模型
def calculation(x1,x2):
    return  -0.1025721*x1 + (-4.27357477)*x2 + 40.29786641

# 评估模型准确性
if __name__ == '__main__':
    testdata = np.loadtxt("cars-test.csv",delimiter=",")
    feature = testdata[:,0:2]
    label = testdata[:,-1]
    totalError = 0
    for index,item in enumerate(feature):
        predict = calculation(item[0],item[1])
        real = label[index]
        errorRate = (real-predict)/real
        totalError = totalError + abs(errorRate)

    error = totalError/len(label)
    print(error)

#  0.15613893091963718