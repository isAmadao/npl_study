# 使用机器学习和调用qwen两种方式，实现dataset.csv中的中文文本分类任务，最后封装为两个函数
# 完成后，使用fastapi将其部署为服务

import jieba  # 中文分词
import pandas as pd  # 表格数据读取
from sklearn.feature_extraction.text import CountVectorizer  # 向量词典
from sklearn.neighbors import KNeighborsClassifier  # knn
from openai import OpenAI  #
from fastapi import FastAPI, HTTPException


# 字符串转特征向量
def str_to_feature(text: str, vector: CountVectorizer) -> list[int]:
    segmented_text = " ".join(jieba.lcut(text))
    return vector.transform([segmented_text])


# 输入为训练集路径，输出为1个KNN的model
# 数据处理思路：先用jieba对所有句子进行分词，之后使用CountVectorizer生成一个词典，再统计每个句子中单词的频数，生成特征向量
def lm_text_classification_fit(datapath: str) -> tuple[CountVectorizer, KNeighborsClassifier]:
    dataset = pd.read_csv(datapath, sep="\t", names=["words", "label"], nrows=10000)
    input_sentence = dataset["words"].apply(lambda x: " ".join(jieba.lcut(x)))  # 对words列用jieba进行分词
    vector = CountVectorizer()  # 创建词频字典
    vector.fit(input_sentence.values)  # 生成词频字典
    input_feature = vector.transform(input_sentence.values)  # 利用词频字典，统计每个词的频数，生成特征向量
    model = KNeighborsClassifier()
    model.fit(input_feature, dataset["label"].values)
    return vector, model


# 使用knn对文本进行分类，输入X为词频特征向量，输出为预测的标签
def lm_text_classification(text: str, model: KNeighborsClassifier, vector: CountVectorizer) -> str:
    test_feature = str_to_feature(text, vector)
    return model.predict(test_feature)[0]


# 登录百炼
def login_bailian(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# 使用qwen对文本进行分类
# 输入为client，text为待处理的文本，prompt为qwen要做的事情和所有的文本类别
def llm_text_classification(client: OpenAI, text: str, prompt: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型的代号
        messages=[
            {"role": "system", "content": prompt},  # 给大模型的命令，角色的定义
            {"role": "user", "content": text}  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


# 接下来是实现fastapi，有两个Get接口，第一个接口是训练模型，并且对入参使用knn进行分类；第二个接口是使用llm进行分类、
app = FastAPI(
    title="文本分类",
    description="一个简单的 FastAPI 应用，用于文本分类。",
    version="1.0.0"
)


@app.get("/")
def welcome():
    return  """你好，欢迎使用文本分类，路径如下:
                /lm_text_classification/text?={query}
                /llm_text_classification/text?={query}
             """


@app.get("/lm_text_classification/")
def lm_text_classification_api(text: str):
    vector, model = lm_text_classification_fit("dataset.csv")
    return lm_text_classification(text, model, vector)


@app.get("/llm_text_classification/")
def llm_text_classification_api(text: str):
    client = login_bailian("sk-4426782b77554b91a2ece1a23fd007ed")
    prompt = """请将输入的本文进行分类，你可以从以下类别中进行选择：
    FilmTele-Play
    Video-Play
    Music-Play
    Radio-Listen
    Alarm-Update
    Travel-Query
    HomeAppliance-Control
    Weather-Query
    Calendar-Query
    TVProgram-Play
    Audio-Play
    Other"""
    return llm_text_classification(client, text, prompt)


# if __name__ == "__main__":
    # 在终端运行 uvicorn 课上练习:app --reload

