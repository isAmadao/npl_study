import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI


data = pd.read_csv('dataset.csv', sep='\t', header=None)
def transform_using_knn(text:str)->str:
    input = data[0].apply(lambda x: " ".join(jieba.lcut(x)))

    vector = CountVectorizer()
    vector.fit(input.values)
    input_feature = vector.transform(input.values)

    knn_model = KNeighborsClassifier()
    knn_model.fit(input_feature, data[1].values)

    processed_text = " ".join(jieba.lcut(text))
    text_feature = vector.transform([processed_text])
    return knn_model.predict(text_feature)


def transform_using_llm(text:str)->str:
    client = OpenAI(
        api_key = "gemini_api_key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-3-flash-preview",
        messages = [
            {
                "role": "user",
                "content": f"""
                帮我进行文本分类， 一下是一些训练示例：
                {data}
                现在请{text}进行分类， 输出的类别只能是{data[1].values}, 请给出
                最合适和类别
            
"""
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_text = "帮我导航到天安门"
    print("机器学习: ", transform_using_knn(test_text))
    print("大语言模型 (无训练示例): ", transform_using_llm(test_text))
















