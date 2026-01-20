import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据集
df = pd.read_csv('dataset.csv', sep='\t', header=None, names=['text', 'label'])

print('数据集基本信息：')
print(df.info())
print('\n分类标签分布：')
print(df['label'].value_counts())


# 文本预处理：使用jieba分词
def tokenize(text):
    return ' '.join(jieba.cut(text))


df['tokenized_text'] = df['text'].apply(tokenize)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['tokenized_text'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\n数据集划分完成：')
print(f'训练集大小：{X_train.shape[0]}')
print(f'测试集大小：{X_test.shape[0]}')

# 模型1：Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# 模型2：KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print('\n=== 模型训练完成 ===')
print(f'Logistic Regression 准确率：{lr_accuracy:.4f}')
print(f'KNN 准确率：{knn_accuracy:.4f}')

user_input = input('请输入要分类的文本（输入"exit"退出）：')
user_tokenized = tokenize(user_input)
user_features = vectorizer.transform([user_tokenized])
lr_pred = lr_model.predict(user_features)[0]
knn_pred = knn_model.predict(user_features)[0]
print(f'\nLogistic Regression 预测结果：{lr_pred}')
print(f'KNN 预测结果：{knn_pred}')
print(f'\n模型精度：')
print(f'Logistic Regression：{lr_accuracy:.4f}')
print(f'KNN：{knn_accuracy:.4f}')
print('-' * 50)

