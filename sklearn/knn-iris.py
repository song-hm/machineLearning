from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_iris():
    """
    用knn算法对鸢尾花进行分类
    :return:
    """
    #获取数据
    iris = load_iris()
    #划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #knn算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    #模型评估
    #1、比对真实值和与预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和与预测值:\n",y_predict == y_test)

    #2、计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)
    return None
def knn_ir_gscv():
    """
    用knn算法对鸢尾花进行分类,添加网格搜索和交叉验证
    :return:
    """
    #获取数据
    iris = load_iris()
    #划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #knn算法预估器
    estimator = KNeighborsClassifier()
    #添加网格搜索和交叉验证
    #参数准备
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)
    #模型评估
    #1、比对真实值和与预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和与预测值:\n",y_predict == y_test)

    #2、计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率：\n",score)

    # 最佳参数：best_params_
    print("最佳参数：\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果:\n", estimator.cv_results_)
    return None

if __name__ == '__main__':
    #用knn算法对鸢尾花进行分类
    # knn_iris()
    #用knn算法对鸢尾花进行分类,添加网格搜索和交叉验证
    knn_ir_gscv()
