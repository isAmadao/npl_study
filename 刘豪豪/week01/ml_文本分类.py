import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(data)

input_sententce = data[0].apply(lambda x: " ".join(jieba.lcut(x)))   # 用jieba分词
print(input_sententce)

vector = CountVectorizer()  # CountVectorizer 将文本转换为词频向量
vector.fit(input_sententce.values)  # 分析所有输入句子，建立词汇表
input_feature = vector.transform(input_sententce.values)  # 将文本转换为数值矩阵,每行代表一个文档,每列代表一个词语的出现次数
print(vector)
print(input_feature[0])

model = KNeighborsClassifier()
model.fit(input_feature, data[1].values)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习的方式），输入文本完成文本的划分
    """
    text_sentence = "".join(jieba.lcut(text))
    test_feature = vector.transform([text_sentence])
    return model.predict(test_feature)[0]

if __name__ == "__main__":
    print("机器学习: ", text_calssify_using_ml("帮我导航到天安门"))
