
from openai import OpenAI

from typing import Union
from fastapi import FastAPI

app = FastAPI()


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # 申请地址：https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-557a9a5ddxxxxx2a11342", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@app.get("/text-cls/llm")
def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
            
输出的类别只能从下列类别中进行选择，不用多余话语。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control        
Weather-Query               
TVProgram-Play      
Audio-Play       
Other             
"""},  # 提问内容
        ]
    )
    return completion.choices[0].message.content

# http://127.0.0.1:8000/text-cls/llm?text=

test_query = "帮我创建一个周一上午9点的会议提醒"
print("待预测的文本:", test_query)
print("Qwen大语言模型预测结果: ", text_calssify_using_llm(test_query))

