import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN  这是K近邻(K-Nearest Neighbors)分类算法的具体实现类

"""
KNeighborsClassifier是一个监督学习分类器，其工作原理是：
将训练数据存储起来
对于新的样本点，计算它与所有训练样本的距离
找到距离最近的K个训练样本(邻居)
将这K个邻居中最常见的类别作为预测结果
主要用途：
分类任务(如手写数字识别、图像分类等)
简单易懂的非参数算法
适合小到中等规模的数据集
对异常值相对不敏感
这种算法特别适用于边界不规则的分类问题，在文本分类、推荐系统等领域有广泛应用。
"""
from openai import OpenAI

"""
openai - 这是OpenAI公司提供的官方Python库，用于访问他们的各种AI服务
OpenAI - 这是该库中的主客户端类，提供了访问OpenAI API的接口
作用：
OpenAI类是一个客户端，用于连接和调用OpenAI的各种AI模型和服务，包括：
ChatGPT系列语言模型(如GPT-3.5, GPT-4等)
文本生成、对话理解、翻译等功能
嵌入向量(embeddings)服务
图像生成(DALL-E)等服务
使用场景：
构建基于大语言模型的应用程序
自然语言处理任务
智能对话系统
文本分析和内容生成
需要注意的是，使用这个库需要有效的OpenAI API密钥才能调用服务。这是访问OpenAI强大AI能力的标准方式，让开发者能够在自己的应用程序中集成先进的AI功能。
"""

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset[1].value_counts())

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

client = OpenAI(
    api_key="sk-b220c23xxxxf000e48920",
    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_classify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


def text_classify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型代号

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
"""}
        ]
    )
    return completion.choices[0].message.content


def text_classify_using_kimi(text: str) -> str:
    """
    文本分类（kimi大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="kimi-k2-thinking",

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
    """}
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print("机器学习: ", text_classify_using_ml("帮我打开情感电台"))
    print("大语言模型: ", text_classify_using_llm("帮我打开情感电台"))
    print("kimi大语言模型：", text_classify_using_kimi("帮我打开情感电台"))  # kimi这个最慢

