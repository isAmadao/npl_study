import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import jieba
from openai import  OpenAI
client = OpenAI(
    api_key='sk-564e35xxxxx807394428',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def test_classify_using_ml(text:str)->str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)

def test_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f'''帮我进行文本分类：{text},分类的类型只能从如下类型中进行选择：
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
            输出只显示最后的单词
            '''

            },
        ]
    )
    return completion.choices[0].message.content
if __name__ == '__main__':
    print(test_classify_using_ml("帮我播放一首2010年的流行歌曲")[0])
    print(test_classify_using_llm("帮我播放一首2010年的流行歌曲"))
