"""
1.机器学习方法：使用TF-IDF方法进行特征提取，使用朴素贝叶斯进行文本分类
2.大语言模型方法：仿照老师代码写作思路，使用自己的百炼api_key，调用大模型进行文本分类
"""
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer #使用TF-IDF词袋模型
from sklearn.naive_bayes import MultinomialNB
from openai import OpenAI
from typing import Union
from fastapi import FastAPI

app = FastAPI()
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

# 使用jieba库对中文进行处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 特征工程，使用TF-IDF模型对文本进行特征提取
vector = TfidfVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 使用朴素贝叶斯模型对文本进行分类
model = MultinomialNB()
model.fit(input_feature, dataset[1].values)

# 初始化OpenAI客户端，用于调用通义千问API
client = OpenAI(
    # 如果没有配置环境变量，可以用自己的百炼API Key替换下行内容：api_key="sk-xxx"
    api_key="sk-b7c19xxxxxxd8d2fcdf3a7",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云DashScope API兼容模式的端点
)

# 配置fastapi服务
@app.get("/text-cls/nb")
def text_classify_using_nb(text:str) -> str:
    """
    文本分类（朴素贝叶斯），输入文本完成类别划分
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    result = model.predict(text_feature)
    return result[0]

@app.get("/text-cls/llm")
def text_classify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",

        messages=[
            {
                "role":"user",
                "content":f"""帮我进行文本分类：{text}
输出的类别只能从如下中进行选择， 只需要输出类别名称，不需要给出类别的详细解释，
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
Other 
"""
            }
        ]
    )
    return completion.choices[0].message.content
