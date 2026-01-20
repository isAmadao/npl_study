import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# 加载数据
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "dataset.csv")
dataset = pd.read_csv(file_path, sep="\t", header=None, nrows=100)
print(dataset.head(5))
print(f"数据集形状: {dataset.shape}")

# 数据预处理 - 中文分词
def chinese_tokenize(text):
    return " ".join(jieba.lcut(text))

# 对文本进行分词
input_sentences = dataset[0].apply(chinese_tokenize)
labels = dataset[1].values

# 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(input_sentences.values)
y = labels

# 划分训练集和测试集（可选，用于评估模型）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 定义模型
models = {
    'Naive Bayes (MultinomialNB)': MultinomialNB(),
    'SVM (Support Vector Classifier)': SVC(kernel='linear')
}

# 训练和评估每个模型
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"训练 {model_name} 模型...")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} 模型评估结果:")
    print(classification_report(y_test, y_pred))
    
    # 预测用户输入的文本
    test_query = "帮我播放一下郭德纲的小品"
    test_sentence = chinese_tokenize(test_query)
    test_feature = vectorizer.transform([test_sentence])
    prediction = model.predict(test_feature)
    
    print(f"待预测的文本: {test_query}")
    print(f"{model_name} 预测结果: {prediction[0]}")

# 如果你想要同时使用所有数据训练并在新数据上测试，可以使用以下代码：
print(f"\n{'='*50}")
print("使用全部数据训练模型并进行预测")

# 使用全部数据重新训练
for model_name, model in models.items():
    # 使用全部数据训练
    model.fit(X, y)
    
    # 预测用户输入的文本
    test_query = "帮我播放一下郭德纲的小品"
    test_sentence = chinese_tokenize(test_query)
    test_feature = vectorizer.transform([test_sentence])
    prediction = model.predict(test_feature)
    
    print(f"\n{model_name} (使用全部数据训练):")
    print(f"待预测的文本: {test_query}")
    print(f"预测结果: {prediction[0]}")

# 可选：使用TF-IDF特征（通常效果更好）
print(f"\n{'='*50}")
print("尝试使用TF-IDF特征提取...")

from sklearn.feature_extraction.text import TfidfVectorizer

# 使用TF-IDF特征提取
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(input_sentences.values)

# 划分训练测试集
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 使用TF-IDF特征训练模型
for model_name, model in models.items():
    print(f"\n使用TF-IDF特征的 {model_name}:")
    model.fit(X_train_tfidf, y_train_tfidf)
    
    # 评估
    y_pred_tfidf = model.predict(X_test_tfidf)
    print(classification_report(y_test_tfidf, y_pred_tfidf))
    
    # 预测
    test_query = "帮我播放一下郭德纲的小品"
    test_sentence = chinese_tokenize(test_query)
    test_feature_tfidf = tfidf_vectorizer.transform([test_sentence])
    prediction = model.predict(test_feature_tfidf)
    print(f"预测结果: {prediction[0]}")
