from typing import Union
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from fastapi import FastAPI

app = FastAPI()



from openai import OpenAI

client = OpenAI(
    api_key="sk-1f8f970c557xxxxxdc981366f9",  # 账号绑定的
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=None)
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理

vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)


# 机器学习选取模型 更换使用   朴素贝叶斯  适用: 文本、大样本    适用: 向量机 高维、小样本
model = MultinomialNB()
# model = SVC(kernel='linear')  # 线性核适合文本
model.fit(input_feature, dataset[1].values)




@app.get("/ml")
def text_classify_using_ml(text: str) -> str:
    """
    文本分类,输入文本完成划分  机器学习
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


@app.get("/llm")
def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号
        messages=[

            {"role": "user", "content": f"""
            帮我进行文本分类:{text}

输出的类型只能从如下中进行选择
FilmTele-Play            
Video-Play               
Music-Play               
Radio-Listen             
Alarm-Update             
Weather-Query            
Travel-Query             
HomeAppliance-Control    
Calendar-Query           
TVProgram-Play           
Audio-Play               
Other                    
"""},
        ]
    )
    return completion.choices[0].message.content
