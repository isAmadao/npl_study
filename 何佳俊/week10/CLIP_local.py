"""
CLIP Zero-Shot Image Classification
使用已下载到本地的 CLIP 模型进行图像分类
"""
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# ========== 配置区域 ==========
# 设置本地模型路径
MODEL_PATH = r"C:\Users\Administrator.DESKTOP-K90E5CL\Desktop\AiLearn\aiLearnWork\models\clip\AI-ModelScope\clip-vit-large-patch14"

# 图片路径
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(script_dir, "dog.png")

# 候选类别（可以使用中文描述）
LABELS = [
    "一张狗的照片",
    "一张猫的照片",
    "一张汽车的照片",
    "一张房子的照片",
    "一张动物的照片"
]
# ===========================

def main():
    print("=" * 50)
    print("CLIP Zero-Shot Image Classification")
    print("=" * 50)
    
    # 1. 加载本地模型
    print(f"\n正在加载本地模型：{MODEL_PATH}")
    model = CLIPModel.from_pretrained(MODEL_PATH)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    print("✓ 模型加载完成")
    
    # 2. 读取图片
    print(f"\n正在读取图片：{IMAGE_PATH}")
    image = Image.open(IMAGE_PATH)
    print(f"✓ 图片尺寸：{image.size}")
    
    # 3. 显示候选类别
    print(f"\n候选类别 ({len(LABELS)}个):")
    for i, label in enumerate(LABELS, 1):
        print(f"  {i}. {label}")
    
    # 4. 进行分类
    print("\n正在进行分类...")
    inputs = processor(
        text=LABELS,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 5. 计算概率
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # 6. 显示结果
    print("\n" + "=" * 50)
    print("分类结果")
    print("=" * 50)
    
    predicted_idx = probs.argmax(dim=1).item()
    confidence = probs[0][predicted_idx].item() * 100
    
    print(f"\n🎯 预测类别：{LABELS[predicted_idx]}")
    print(f"📊 置信度：{confidence:.2f}%")
    
    print("\n所有类别得分（按置信度排序）:")
    sorted_probs, sorted_indices = probs.sort(descending=True)
    for i, (prob, idx) in enumerate(zip(sorted_probs[0], sorted_indices[0]), 1):
        label = LABELS[idx.item()]
        score = prob.item() * 100
        print(f"  {i}. {label:20s} {score:6.2f}%")
    
    print("\n" + "=" * 50)
    print("✅ 完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()

