import requests
import json


def test_api(url, payload, api_name):
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(f"{api_name} - Status Code:", response.status_code)
        print(f"{api_name} - Response Body:", response.text)
        print("-" * 50)
    except Exception as e:
        print(f"{api_name} - 请求失败: {e}")
        print("-" * 50)


# 测试所有API端点
base_url = "http://localhost:8000"
payload = {
    "request_id": "test123",
    "request_text": "帮我播放周杰伦的歌曲"
}

# 测试TF-IDF分类接口
test_api(f"{base_url}/v1/text-cls/tfidf", payload, "TF-IDF API")

# 测试正则表达式分类接口
test_api(f"{base_url}/v1/text-cls/regex", payload, "Regex API")

# 测试BERT分类接口（可能会失败，因为缺少模型文件）
test_api(f"{base_url}/v1/text-cls/bert", payload, "BERT API")

# 测试GPT分类接口（可能会失败，因为缺少OpenAI配置）
test_api(f"{base_url}/v1/text-cls/gpt", payload, "GPT API")