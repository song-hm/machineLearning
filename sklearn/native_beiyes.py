from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    """
    用朴素贝叶斯算法对数据进行分类
    :return:
    """
    #1）获取数据
    news = fetch_20newsgroups(subset="all")
    #2)划分数据集
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target)
    #3)特征工程：文本特征抽取-tfidf
    transfer = text.TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #4)朴素贝叶斯预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    #5)模型评估
    # 1、比对真实值和与预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和与预测值:\n", y_predict == y_test)

    # 2、计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：\n", score)

    return None
if __name__ == '__main__':
    #用朴素贝叶斯算法对数据进行分类
    nb_news()
