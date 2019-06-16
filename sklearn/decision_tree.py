from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
def decision_iris():
    """
    用决策树对鸢尾花数据分类
    :return:
    """
    #1、获取数据集
    iris = load_iris()

    #2、划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)
    #3、使用决策树预估器分类
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    #4、模型评估
    # 1)比对真实值和与预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和与预测值:\n", y_predict == y_test)

    # 2)计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：\n", score)

    #可视化决策树
    export_graphviz(estimator,out_file="iris_tree.dot",feature_names=iris.feature_names)
    return None

if __name__ == '__main__':
    #用决策树对鸢尾花数据分类
    decision_iris()
