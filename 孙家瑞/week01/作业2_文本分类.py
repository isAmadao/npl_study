import os

import pandas as pd
import jieba
import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("dataset.csv", sep="\t", header=None)
classList = str(data[1].unique().tolist())
vector = CountVectorizer()


def build_model() -> KNeighborsClassifier:
    """
    使用sk-learn 的组件训练一个kNN模型
    :return: kNN模型
    """
    print(data.head(5))
    # 进行分词，然后将每个词用空格重新连接
    input_sentence = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
    # 生成词频库
    vector.fit(input_sentence.values)
    # 转换为特征向量
    feature = vector.transform(input_sentence.values)
    input_feature = feature

    # 声明一个模型并进行训练
    KNN = KNeighborsClassifier()
    KNN.fit(input_feature, data[1].values)
    return KNN


KNN = build_model()


def test_classify(test_query: str) -> str:
    """
    根据生成的模型，对句子的分类进行预测
    :param test_query:
    :return:
    """
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print("test_feature：", test_feature)
    result = KNN.predict(test_feature)
    print("KNN 预测结果：", result)
    return result[0]


def test_LLM_classify(test_query: str) -> str:
    """
    使用通义千问大模型，对句子进行分类
    :param test_query: 输入的句子
    :return: 分类结果
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key=os.environ.get("DASHSCOPE_API_KEY", "该环境变量不存在"),  # 账号绑定的

        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",  # 模型的代号
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 给大模型的命令，角色的定义
            {"role": "user", "content": "帮我做文本分类，分类任务结果必须是以下结果"},
            # 分类词库
            {"role": "user", "content": classList},
            # 用户的提问
            {"role": "user", "content": test_query},  # 用户的提问
        ]
    )
    content = completion.choices[0].message.content
    print("大模型 预测结果：",content)
    return content




app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/classify/knn")
async def classify_1(q: str):
    cls = test_classify(q)
    result_ = {"test_query": q, "result": cls}
    print(result_)
    return result_


@app.get("/classify/llm")
async def classify(q: str):
    cls = test_LLM_classify(q)
    result_ = {"test_query": q, "result": cls}
    print(result_)
    return result_


# 执行方法 fastapi run xxx.py


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)