import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from openai import OpenAI

#机器学习
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=100000)
print(dataset[1].value_counts())

# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

#大模型学习
client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



def text_classify_using_ml(text: str) -> str:
    """
    文本分类(机器学习)，输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_classify_using_llm(text: str) -> str:
    """
    文本分类(机器学习)，输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 给大模型的命令，角色的定义
            {"role": "user", "content": f"""
            帮我进行文本分类：{str}
            输出的类别只能从如下中进行选择：
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
            """
            },  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    print("机器学习",text_classify_using_ml("故宫怎么走"))
    print("大语言模型",text_classify_using_llm("故宫怎么走"))
    print("机器学习",text_classify_using_ml("帮我导航到天安门"))
    print("大语言模型",text_classify_using_llm("帮我导航到天安门"))