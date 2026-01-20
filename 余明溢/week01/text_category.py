import jieba
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from fastapi import FastAPI, Request, Response

app = FastAPI()

csv_dataset = pd.read_csv('dataset.csv', sep='\t', encoding='utf-8', header=None, nrows=13000)
# print(csv_dataset[0])
# print(csv_dataset[1])

# for d in csv_dataset[0]:
# print(jieba.lcut(d))

# 中文分割
cn_text_input = csv_dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 中文词频统计
count_vector = CountVectorizer()
count_vector.fit(cn_text_input.values)
input_feat = count_vector.transform(cn_text_input.values)

model = KNeighborsClassifier()
model.fit(input_feat, csv_dataset[1])

open_client = OpenAI(
    api_key="sk-8fb3abb20xxxxced430028",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


@app.get("/text-category/ml")
def ml(text: str) -> str:
    input_text_feat = count_vector.transform([" ".join(jieba.lcut(text))])
    return model.predict(input_text_feat)[0]


@app.get("/text-category/llm")
def llm(text: str) -> str:
    completion = open_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

            输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
            """},
        ]
    )
    return completion.choices[0].message.content
