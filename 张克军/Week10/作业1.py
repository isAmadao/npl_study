#作业1: 本地使用一张图（小狗图片），尝试进行一下clip的zero shot classification 图像分类；
import os
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
model_base_path = os.environ.get("MODEL_BASE_PATH")
chinese_clip_path = model_base_path +'/AI-ModelScope/chinese-clip-vit-base-patch16'
print(model_base_path)
print(chinese_clip_path)

model = ChineseCLIPModel.from_pretrained(chinese_clip_path)
processor = ChineseCLIPProcessor.from_pretrained(chinese_clip_path)

#加载本地图片
image = Image.open("dog.jpg")
print(f"已加载图片：{image.size}")

labels = [
    "猫",
    "鸟",
    "汽车",
    "花",
    "狗",      # 期望的分类
    "树",
    "人",
    "食物"
]
print("候选分类标签：", labels)

# 处理图片
# 分开处理
image_inputs = processor(images=image, return_tensors="pt")  # 只处理图片
text_inputs = processor(text=labels, return_tensors="pt", padding=True)  # 只处理文本


#方法1
# 提取特征向量
with torch.no_grad():
    # 图片特征
    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)
    similarities = (image_features @ text_features.T).squeeze()
#拿掉 图片的向量特征 和文本分类的向量特征，下面是输出维度，维度图片简单为一维 ，文本类比为分类维
#图片特征向量： torch.Size([1, 512])
#文本特征向量： torch.Size([8, 512])
print("图片特征向量：", image_features.shape)
print("文本特征向量：", text_features.shape)
#输出的一维结果 tensor([153.9512, 151.8122, 127.6717, 157.5312, 180.7639, 140.4515, 150.7501,149.1056])
print("similarities：", similarities)


# 找出相似度最高的类别
probabilities = similarities.softmax(dim=-1)
print("概率分布：", probabilities)
predicted_index = similarities.argmax().item()
predicted_label = labels[predicted_index]
print(f"预测类别：{predicted_label}")
print(f"\n概率分布：")
for i, label in enumerate(labels):
    prob = probabilities[i].item()
    bar = "█" * int(prob * 50)
    print(f"  {label:6s}: {prob:.4f} ({prob*100:.2f}%) {bar}")

print("\n" + "=" * 50)
if predicted_label == "狗":
    print("✅ 成功识别出小狗！")
else:
    print(f"⚠️  预测结果是'{predicted_label}'，可能需要更换图片或调整标签")
print("=" * 50)

#最后的结果分别使用人和狗的图片进行测试 准确率 接近100% 证明 模型的原始效果就很好