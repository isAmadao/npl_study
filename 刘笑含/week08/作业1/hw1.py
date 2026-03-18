class ExtractionAgent:
  def __init__(self, model_name: str):
    self.model = model_name


  def call(self,user_prompt, response_model):
    messages = [
      {
        "role":"user",
        "content":user_prompt
      }
    ]

    tools = [
      {
        "type":"function",
        "function": {
          "name": response_model.model_json_schema()["title"],
          "description":response_model.model_json_schema()['description'],
          "parameters": {
            "type":"object",
            "properties": response_model.model_json_schema()['properties'],
            "required":response_model.model_json_schema()['required']
          }
        }
      }
    ]

    response = client.chat.completions.create(
      model= self.model,
      messages = messages,
      tools = tools,
      tool_choice="auto"
    )

    try:
      arguments = response.choices[0].message.tool_calls[0].function.arguments
      return response_model.model_validate_json(arguments)
    except:
      print('ERROR', response.choices[0].message)
      return None



class Translation(BaseModel):
  """文本翻译小助手"""
  origin: str=Field(description="原始语种")
  target: str=Field(description="目标语种")
  text: str=Field(description="待翻译的文本")

result1 = ExtractionAgent(model_name="qwen-plus").call('请把nice weather翻译成中文', Translation)
result2 = ExtractionAgent(model_name="qwen-plus").call('请把nice weather翻译成德文', Translation)
print(result1, result2)
