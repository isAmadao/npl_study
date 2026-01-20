import jieba # 中文分词
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN模型

from typing import Union
from fastapi import FastAPI

app = FastAPI()

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset.head(6)) #输出前6行

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

@app.get("/text-cls/knn")
def text_calssify_using_knn(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# http://127.0.0.1:8000/text-cls/knn?text=

test_query = "麻烦帮我创建一个周一上午9点的会议提醒"
print("待预测的文本：", test_query)
print("KNN模型预测结果: ", text_calssify_using_knn(test_query))
