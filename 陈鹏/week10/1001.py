from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import warnings
warnings.filterwarnings("ignore")  # 屏蔽无用警告

"""
本地使用一张图（小狗图片），尝试进行一下clip的zero shot classification 图像分类；
"""

# ----------------------
# 1. 加载本地中文 CLIP 模型
# ----------------------
local_model_path = "D:/AI/modelscope/AI-ModelScope/chinese-clip-vit-base-patch16"

model = ChineseCLIPModel.from_pretrained(local_model_path)
processor = ChineseCLIPProcessor.from_pretrained(local_model_path)

# ----------------------
# 2. 加载小狗图片
# ----------------------
image = Image.open("./dog.jpeg")


# ----------------------
# 3. 分类类别
# ----------------------
classes = ["小狗", "小猫", "猴子", "鸟", "花"]

# ----------------------
# 4. 预处理
# ----------------------
inputs = processor(
    text=classes,
    images=image,
    return_tensors="pt",  # 返回 PyTorch 格式的数据
    padding=True  # 文字长度不一样，自动补空格对齐
)

# ----------------------
# 5. 推理
# ----------------------
with torch.no_grad():
    outputs = model(**inputs)

# ----------------------
# 6. 计算概率
# ----------------------
logits_per_image = outputs.logits_per_image  # # 拿出原始分数
probs = logits_per_image.softmax(dim=1)  # 把分数 → 转成百分比概率

# ----------------------
# 7. 输出结果
# ----------------------
print("🔍 分类概率：")
for i, class_name in enumerate(classes):
    print(f"{class_name}：{probs[0][i].item():.2%}")

max_index = torch.argmax(probs)
print("\n✅ 最终分类结果：", classes[max_index])
