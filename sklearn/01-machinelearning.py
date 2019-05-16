import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def sklearn_demo():
    """
    sklearn数据集使用
    :return:
    """
    #获取数据集
    iris = load_iris()
    # print("鸢尾花数据集：\n",iris)
    # print("鸢尾花的描述：\n",iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)
    print("查看特征值：\n",iris.data,iris.data.shape)

    #数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值：\n",x_train,x_train.shape)
    return None

def dict_demo():
    """
    #字典特征抽取
    :return:
    """
    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    #实例化一个转换器
    transfer = DictVectorizer(sparse=False)

    #调用 fit_transform
    data_new = transfer.fit_transform(data)

    print("data_new:\n",data_new)
    print("特征名字：\n",transfer.get_feature_names())

    return None

if __name__ == '__main__':
    # code1：sklearn数据集使用
    # sklearn_demo()

    #字典特征抽取
    dict_demo()
