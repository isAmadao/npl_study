import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# 1. 数据加载
df = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=100)
print(df.head(5))
print(f"加载数据成功，共 {len(df)} 条")
print(df.head(), "\n")

# 2. 定义中文分词函数
# sklearn 的 TfidfVectorizer 需要接收字符串，而不是列表
# 所以我们需要把 jieba 分好的词用空格拼回字符串
def chinese_word_cut(text):
    return " ".join(jieba.lcut(text))

# 3. 构建机器学习 Pipeline (管道)
# Pipeline 可以把“分词转向量”和“分类”串在一起，非常方便
# 参数说明：
# TfidfVectorizer: 将文本转换为 TF-IDF 特征矩阵
# MultinomialNB: 多项式朴素贝叶斯，适合文本分类（离散特征）
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# 4. 划分训练集和测试集
X = df[0]
y = df[1]

# 20% 的数据用来做测试，其余用来训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 统计真实标签分布
print("训练真实标签分布:", Counter(y_train))
print("测试真实标签分布:", Counter(y_test))

# ==== start: 如果不使用 pipeline ====
# vectorizer = TfidfVectorizer()
#
# # 重要：只在训练集上 fit，然后在训练集和测试集上 transform
# X_train_tfidf = vectorizer.fit_transform(X_train)   # 学习词表 + TF-IDF 权重，并转换
# X_test_tfidf  = vectorizer.transform(X_test)
#
# print(f"训练集 TF-IDF 矩阵形状: {X_train_tfidf.shape}")
# print(f"测试集 TF-IDF 矩阵形状: {X_test_tfidf.shape}\n")
# ==== end: 如果不使用 pipeline ====

# 5. 训练模型
print("开始训练模型...")
# 在 fit 的时候，Pipeline 会自动调用我们的分词逻辑（需要先处理一下数据）
# 这里我们直接对原始文本进行预处理并训练
text_clf.fit(X_train.apply(chinese_word_cut), y_train)
print("训练完成！\n")

# ==== start: 如果不使用 pipeline ====
# clf = MultinomialNB()
# clf.fit(X_train_tfidf, y_train)
# print("模型训练完成！\n")
# ==== end: 如果不使用 pipeline ====

# 6. 模型评估
print("在测试集上的表现：")
y_pred = text_clf.predict(X_test.apply(chinese_word_cut))
print(classification_report(y_test, y_pred, zero_division=0))

# ==== start: 如果不使用 pipeline ====
# y_pred = clf.predict(X_test_tfidf)
#
# print("在测试集上的表现：")
# print(classification_report(y_test, y_pred, zero_division=0))
# ==== end: 如果不使用 pipeline ====

# 7. 预测新文本
print("-" * 30)
print("开始预测新文本...")

new_texts = [
    "帮我播放一下郭德纲的小品",       # 你的例子
    "我要买去北京的火车票",           # Travel-Query
    "这歌真好听，再来一首",          # Music-Play
    "NBA总决赛什么时候开始？",        # Sports (假设你数据里有这个类)
]

# 对新文本进行分词
new_texts_cut = [chinese_word_cut(t) for t in new_texts]

# 预测
predicted = text_clf.predict(new_texts_cut)

for text, label in zip(new_texts, predicted):
    print(f"文本: {text}  --->  预测类别: {label}")
