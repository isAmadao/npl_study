from openai import OpenAI

client = OpenAI(
    api_key="sk-db6f1e3xxxxxf948e292d076",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（利用大语言模型），输入文本完成类别划分
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
            """},  # 用户输入的内容
        ]
    )
    return completion.choices[0].message.content


#if __name__ == "__name__":
#    print("大模型语言：", text_calssify_using_llm("帮我导航到武汉"))  # 这个好像不会执行一样的，不会输出结果
if __name__ == "__main__":
    print("机器学习: ", text_calssify_using_llm("帮我导航到天安门"))
