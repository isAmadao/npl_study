from PIL import Image
import requests
import torch

from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("./model/AI-ModelScope/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./model/AI-ModelScope/chinese-clip-vit-base-patch16")


candidate_labels = [
    "小狗",          # 狗
    "小猫",          # 猫
    "小鸟",         # 鸟
    "汽车",          # 汽车
    "房子",        # 房子
    "鲜花",       # 花
    "食物"          # 食物
]

image_path = "./image/dog.png"   # 请替换为你的小狗图片实际路径
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"错误：未找到图片 {image_path}，请检查路径")
    exit()

# 为了使分类更准确，可以使用带描述的短语（CLIP 对自然语言友好）
# 例如：["a photo of a dog", "a photo of a cat", ...]
text_inputs = [f"这个图片是 {label}" for label in candidate_labels]


#使用 CLIP 处理器将图片和文本转换为模型输入
inputs = processor(
    text=text_inputs,
    images=image,
    return_tensors="pt",
    padding=True
)

# ------------------------------
# 前向传播，计算相似度分数
# ------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像与文本的相似度分数
    probs = logits_per_image.softmax(dim=1)      # 转换为概率分布

# ------------------------------
# 输出分类结果
# ------------------------------
for i, label in enumerate(candidate_labels):
    print(f"{label:10s}: {probs[0][i].item():.4f}")

# 预测最可能的类别
predicted_idx = probs.argmax(dim=1).item()
predicted_label = candidate_labels[predicted_idx]
print(f"\n预测结果：这张图片最可能是【{predicted_label}】")