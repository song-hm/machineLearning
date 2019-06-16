import sklearn
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer,text
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import jieba
import pandas as pd

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

    print("data_new:\n",data_new,type(data_new))
    print("特征名字：\n",transfer.get_feature_names())

    return None

def count_demo():
    """
    文本特征提取
    :return:
    """
    data = ["lift is short,i like like python",
            "lift is too long,i dislike python"]
    # 实例化一个转换器
    transfer = text.CountVectorizer(stop_words=["is","too"])
    # 调用 fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new: \n",data_new.toarray())
    print("特征名字: \n",transfer.get_feature_names())

    return None

def cut_word(text):
    """
    中文分词:"我爱北京天安门" ==> "我 爱 北京 天安门"
    :return:
    """
    return " ".join(list(jieba.cut(text)))

def count_chinese_demo():
    """
    中文文本特征提取
    :return:
    """
    data = ["我 爱 北京 天安门",
            "天安门 上 太阳 升"]
    # 实例化一个转换器
    transfer = text.CountVectorizer()
    # 调用 fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new: \n",data_new.toarray())
    print("特征名字: \n",transfer.get_feature_names())

    return None

def count_chinese_demo2():
    """
    中文文本特征提取,自动分词
    :return:
    """
    data = ["我爱北京天安门",
            "天安门上太阳升"]
    #中文分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 实例化一个转换器
    transfer = text.CountVectorizer()
    # 调用 fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new: \n",data_final.toarray())
    print("特征名字: \n",transfer.get_feature_names())

    return None
def tfidf_demo():
    """
    使用TF-IDF进行中文文本特征提取,自动分词
    :return:
    """
    data = ["我爱北京天安门",
            "天安门上太阳升"]
    #中文分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 实例化一个转换器
    transfer = text.TfidfVectorizer()
    # 调用 fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new: \n",data_final.toarray())
    print("特征名字: \n",transfer.get_feature_names())

    return None

def minmax_demo():
    """
    数据归一化
    :return:
    """
    #读取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:,:3]
    # 实例化一个转换器
    transfer = MinMaxScaler(feature_range=(0,1))
    # 调用 fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None
def standard_demo():
    """
    数据标准化
    :return:
    """
    #读取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:,:3]
    # 实例化一个转换器
    transfer = StandardScaler()
    # 调用 fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None

def variance_demo():
    """
    低方差特征过滤
    :return:
    """
    #读取数据
    data  = pd.read_csv("factor_returns.csv")
    data = data.iloc[:,1:-2]
    # print(data)
    #实例化一个转换器
    transfer = VarianceThreshold(threshold=10)
    #调用fit_transform方法
    data_new = transfer.fit_transform(data)
    # print("data_new: \n",data_new,data_new.shape)
    #计算两个变量之间的相关系数
    r = pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("相关系数r= \n",r)

def pca_demo():
    """
    PCA（主成分分析法）降维
    :return:
    """
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    #实例化一个转换器
    transfer = PCA(n_components=0.95)
    #调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)


if __name__ == '__main__':
    # code1：sklearn数据集使用
    # sklearn_demo()

    #字典特征抽取
    # dict_demo()

    #文本特征提取
    # count_demo()

    #中文文本特征提取
    # count_chinese_demo()

    #中文分词
    # print(cut_word("我爱北京天安门"))

    #中文文本特征提取,自动分词
    # count_chinese_demo2()

    #使用TF-IDF进行中文文本特征提取,自动分词
    # tfidf_demo()

    #数据归一化
    # minmax_demo()

    #数据标准化
    # standard_demo()

    #低方差特征过滤
    # variance_demo()

    #PCA（主成分分析法）降维
    pca_demo()
