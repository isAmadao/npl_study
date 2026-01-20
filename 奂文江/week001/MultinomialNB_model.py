import pandas as pds
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


dataset_str = pds.read_csv("dataset.csv", sep="\t", header=None)

input_sentence = dataset_str[0].apply(lambda x: " ".join(jieba.lcut(x)))

countVectorizer = CountVectorizer()
countVectorizer.fit(input_sentence)
input_feature = countVectorizer.transform(input_sentence.values)

#朴素贝叶斯模型
model = MultinomialNB(alpha=0.5)
model.fit(input_feature, dataset_str[1].values)

text = "我要开车去西藏"

if __name__ == "__main__":
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = countVectorizer.transform([text_sentence])
    result = model.predict(text_feature)[0]
    print(f"MultinomialNB模型预测结果为：{result}")
