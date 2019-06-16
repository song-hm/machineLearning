from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

def linear1():
    """
    正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    #获取数据
    boston = load_boston()

    #划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=6)
    #标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #预估器
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    #得出模型
    print("正规方程权重系数为：\n",estimator.coef_)
    print("正规方程偏置为：\n",estimator.intercept_)

    #模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n",y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程均方误差：\n",error)

    return None

def linear2():
    """
    梯度下降的优化方法对波士顿房价进行预测
    :return:
    """
    #获取数据
    boston = load_boston()
    print("特征数量：\n",boston.data.shape)

    #划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=6)
    #标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #预估器
    estimator = SGDRegressor(learning_rate="constant",eta0=0.01,max_iter=10000)
    estimator.fit(x_train,y_train)

    #得出模型
    print("梯度下降权重系数为：\n",estimator.coef_)
    print("梯度下降偏置为：\n",estimator.intercept_)

    #模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差：\n", error)
    return None

def linear3():
    """
    岭回归的优化方法对波士顿房价进行预测
    :return:
    """
    #获取数据
    boston = load_boston()
    print("特征数量：\n",boston.data.shape)

    #划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=6)
    #标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #预估器
    # estimator = Ridge(alpha=0.5,max_iter=10000)
    # estimator.fit(x_train,y_train)

    #保存模型
    # joblib.dump(estimator,"ridge_boston.pkl")
    #加载模型
    estimator = joblib.load("ridge_boston.pkl")

    #得出模型
    print("岭回归权重系数为：\n",estimator.coef_)
    print("岭回归偏置为：\n",estimator.intercept_)

    #模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归均方误差：\n", error)
    return None

if __name__ == '__main__':
    #正规方程的优化方法对波士顿房价进行预测
    linear1()
    #梯度下降的优化方法对波士顿房价进行预测
    linear2()
    #岭回归的优化方法对波士顿房价进行预测
    linear3()

