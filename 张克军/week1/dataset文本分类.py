import time

import pandas as pd #用于数据处理和分析的库
import jieba as jb #用于分词的库
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

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



def test(text :str):
    test_sentence = " ".join(jb.lcut(text))
    test_feature = vector.transform([test_sentence])
    knn_res = knn_model.predict(test_feature)[0]
    nb_res = nb_model.predict(test_feature)[0]
    print(f"{text} -> {knn_res},{nb_res}")

test("我想看和平精英上战神必备技巧的游戏视频")
test("播放钢琴曲命运交响曲")
test("我怎么去大梅沙")
test("美女，约吗？")
test("我要怎么才能学好ai编程")


