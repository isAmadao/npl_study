import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=100)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# max_iter=1000 表示最大迭代次数，文本数据维度高，需要多跑几次才能收敛
model = LogisticRegression(max_iter=1000)
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("逻辑回归模型预测结果: ", model.predict(test_feature))
