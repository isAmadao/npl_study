import pandas as pd
import jieba
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

vector = None
model = None
client = None
conversation_history = []

def _init_ml():
    """机器学习初始化"""
    global vector, model
    # 读取csv为DataFrame对象
    dataset = pd.read_csv("../dataset.csv", sep="\t", header=None, nrows=10)
    print(dataset)
    # 获取样本文本为Series对象
    sampledata = dataset[0]
    print(sampledata)
    # 使用匿名函数对每个Series对象样本文本进行分词
    input_sentence = sampledata.apply(lambda x: " ".join(jieba.lcut(x)))
    print(input_sentence)
    print(input_sentence.values)

    # 创建词频统计对象
    vector = CountVectorizer()
    # 输入文本
    vector.fit(input_sentence.values)
    # 使用transform将文本转换为特征矩阵
    input_feature = vector.transform(input_sentence.values)
    # print(input_feature)

    # 创建KNN模型
    model = KNeighborsClassifier()
    # 训练模型
    model.fit(input_feature, dataset[1].values)


def _init_llm():
    """大模型初始化"""
    global client, conversation_history
    # 创建客户端连接
    client = OpenAI(
        # 百炼API Key
        api_key="sk-9dac9bc999f246eca490105ea0xxxxxx",
        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 初始化提示词
    conversation_history = [
        # 给大模型的命令，角色的定义
        {"role": "system", "content": "You are a helpful assistant."},
        # 用户的提问
        {"role": "user", "content": f"""
                在之后的对话中，我每发送给你一句话，则请你帮我进行文本分类，输出的类别只能从如下中进行选择，且只输出类别，不要输出其他内容： 
                Video-Play 
                FilmTele-Play 
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
         },
    ]


_init_ml()
_init_llm()


def text_classify_by_ml(text: str) -> str:
    """文本分类（机器学习）"""
    # 输入数据分词
    test_sentence = " ".join(jieba.lcut(text))
    print(test_sentence)
    # 转换为特征矩阵
    test_future = vector.transform([test_sentence])
    # 输出第一个预测值
    return model.predict(test_future)[0]


def text_classify_by_llm(text: str) -> str:
    """文本分类（大语言模型）"""
    global client, conversation_history
    # 添加新的用户输入
    conversation_history.append({"role": "user", "content": text})
    # 向大语言模型发送提问
    completion = client.chat.completions.create(
        # 模型的代号
        model="qwen-plus",
        messages = conversation_history
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    text = input("请输入文本：")
    while text != "退出":
        print("机器学习输出结果", text_classify_by_ml(text))
        print("大语言模型输出结果", text_classify_by_llm(text))
        text = input("请输入文本：")
