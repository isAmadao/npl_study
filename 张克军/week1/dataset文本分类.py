import time

import pandas as pd #用于数据处理和分析的库
import jieba as jb #用于分词的库
from fastapi import FastAPI
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

#读取本地文件
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.shape)
# dataset 是一个文本的一个文类的数据集，现在要求对这个文本进行训练，然后使用这个模型用于对文本进行分类
# 1. 数据预处理
# 1.1 分词
#这个input_sententce 是对每个字符串再进行分词，讲每行字符串变成每行的分词列表
input_sententce = dataset[0].apply(lambda x: " ".join(jb.lcut(x)))

# 2. 特征提取,简单理解 这是个词频特征提取器，输入分好词的数据 ，fit到他 就能拿到对应的词频向量之类的东西
vector =CountVectorizer()
# 2.1 拿到词频特征，这个input_feature 应该
# 就是样本数量（行数） x 词汇表大小（列数），然后存储每行对应的词语出现的个数，即词频
# 拿到的这个词频特征其实就是训练内容，可以每行分割出一行词表维度的向量 代表词表的每个词语出现次数
# 然后每行已经有一个分类，即为输出，训练完之后 ，模型就可以预测，
# 而预测的原理就是 将输入文本分词，按照之前的词表得到词表向量 直接预测即可
input_feature = vector.fit_transform(input_sententce.values)
# 3. 模型训练
knn_model = KNeighborsClassifier()
#将词频特征，记录了每行的词表向量 ，再加上 每行的结果类型 用于训练
knn_model.fit(input_feature, dataset[1])

#朴素贝叶斯 模型
nb_model = MultinomialNB()
nb_model.fit(input_feature, dataset[1])

#大预言模型 llm
qw_client = OpenAI(
    #自己生气的api_key
    api_key="sk-7a14b640xxxxx97e229a",
    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

@app.get("/ai")
def llm_user_query(text :str):
    #调用大预言模型,询问内容并得到回复
    res = qw_client.chat.completions.create(
        model="qwen-flash",  # 模型的代号
        messages=[{"role": "user", "content": text}]
    )
    return res.choices[0].message.content

types = set(dataset[1].values)

@app.get("/text-cls/ml")
def ml_cls(text :str) ->str:
    test_sentence = " ".join(jb.lcut(text))
    test_feature = vector.transform([test_sentence])
    #返回机器学习的预测，可选knn和nb
    #knn_model.predict(test_feature)[0]
    return nb_model.predict(test_feature)[0]

@app.get("/text-cls/llm")
def llm_cls(text :str) ->str:
    llm_str = (f"请帮我把这段话进行文本分类 字符串:'{text}'，"
               f"类别只能从我指定的列表中选择:{types},请直接回复类别即可")
    return llm_user_query(llm_str)

@app.get("/text-cls/all")
def test(text :str) ->str:
    result = f"{text} -> {ml_cls(text)},{llm_cls(text)}"
    print(result)
    return result

if __name__ == '__main__':
    print(f"主要类别有:{types}")
    print("开始测试字符串分类任务 ...")
    test("我想看和平精英上战神必备技巧的游戏视频")
    test("播放钢琴曲命运交响曲")
    test("我怎么去大梅沙")
    test("你好，吃饭了吗？")
    test("我要怎么才能学好ai编程")


