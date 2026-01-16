import jieba
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 数据加载与模型训练 ---
data = pd.read_csv("../dataset.csv", sep="\t", header=None, nrows=10000)
target = data[1].values  # 目标标签

raws = data[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 对文本列进行分词和连接
vector = CountVectorizer()
vector.fit(raws.values)
input_features = vector.transform(raws.values)  # 将所有训练文本转换为特征矩阵

model = KNeighborsClassifier()
model.fit(input_features, target)  # 训练模型

# --- LLM 客户端配置 ---
client = OpenAI(
    api_key="sk-dfc733xxxxxx0d94005c07c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# --- API 路由 ---

@app.get("/predict1")
def get_ml_prediction(sentence: str) -> str:
    """
    使用机器学习模型进行预测
    """
    processed_input = " ".join(jieba.lcut(sentence))  # 预处理输入文本
    input_dim = vector.transform([processed_input])  # 将文本转换为特征向量,接收为列表
    prediction = model.predict(input_dim)  # 预测
    return prediction[0]  # 返回预测结果字符串


@app.get("/predict2")
def get_llm_prediction(sentence: str) -> str:
    """
    使用大语言模型进行预测
    """
    # 构造提示词
    prompt = f"""你是一个智能文本分类助手。请完成以下任务：

1. 首先，请简短地自我介绍，说明你使用的是什么模型。
2. 然后，分析以下用户输入的文本，将其归类到指定的类别列表中。

可选类别列表：
FilmTele-Play, Video-Play, Music-Play, Radio-Listen, Alarm-Update, Travel-Query, HomeAppliance-Control, Weather-Query, Calendar-Query, TVProgram-Play, Audio-Play, Other

用户输入文本：
“{sentence}”

请直接输出介绍和分类结果。"""

    result = client.chat.completions.create(
        model="qwen-long-latest",  # 指定百炼模型
        messages=[{"role": "user", "content": prompt}]
    )
    return result.choices[0].message.content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
