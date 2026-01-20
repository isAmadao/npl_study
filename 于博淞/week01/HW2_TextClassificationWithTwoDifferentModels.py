import jieba
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 从数据库读数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=None)
# print(dataset[1].value_counts())

# 中文分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print("分词处理后：", input_sentence)

# 提取特征
vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词，不是模型 不保留语序
vector.fit(input_sentence.values)  # 统计词表
input_feature = vector.transform(input_sentence.values)  # 进行转换 100 * 词表大小

# KNN模型
knn_model = KNeighborsClassifier()
knn_model.fit(input_feature, dataset[1].values)

# 逻辑回归模型
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(input_feature, dataset[1].values)

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(input_feature, dataset[1].values)


def text_classify_using_knn(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return knn_model.predict(test_feature)[0]


def text_classify_using_lr(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return lr_model.predict(test_feature)[0]


def text_classify_using_rf(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return rf_model.predict(test_feature)[0]


if __name__ == "__main__":
    test_text = "查询洛杉矶明天的天气情况"
    print("KNN模型:", text_classify_using_knn(test_text))
    print("逻辑回归模型:", text_classify_using_lr(test_text))
    print("随机森林模型:", text_classify_using_rf(test_text))
