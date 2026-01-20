import jieba
import pandas as pd
from sklearn import linear_model
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI

dataSrc=pd.read_csv("dataset.csv",sep="\t",header=None,nrows=10000)
#print("src:",dataSrc[1].value_counts())

input_token=dataSrc[0].apply(lambda x:" ".join(jieba.lcut(x)))
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_token.values) # 统计词表
input_feature = vector.transform(input_token.values)
#knn
#model = neighbors.KNeighborsClassifier()
#model.fit(input_feature, dataSrc[1].values)

#线性回归
model=linear_model.LogisticRegression()
model.fit(input_feature, dataSrc[1].values)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-aaxxxxxf1954bc797753a3", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_using_ml(text:str)->str:
    """
        文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

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
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("result:")
    print("机器学习: ", text_using_ml("帮我导航到天安门"))
    print("大语言模型: ", text_using_llm("帮我导航到天安门"))

