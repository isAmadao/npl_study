from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载 CLIP 模型
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# 3. 读取本地图片
image_path = "dog.jpg"
image = Image.open(image_path).convert("RGB")

# 4. 候选标签
# CLIP 官方 zero-shot 示例常用类似 "a photo of a ..." 的自然语言提示词
candidate_labels = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a wolf",
    "a photo of a fox",
    "a photo of a teddy bear"
]

# 5. 预处理
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 把输入放到同一设备
inputs = {k: v.to(device) for k, v in inputs.items()}

# 6. 推理
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image   # [1, num_labels]
    probs = logits_per_image.softmax(dim=1)[0]

# 7. 输出结果
print("分类结果：")
for label, prob in sorted(zip(candidate_labels, probs.tolist()), key=lambda x: x[1], reverse=True):
    print(f"{label:25s} -> {prob:.4f}")

best_idx = probs.argmax().item()
print("\n最终预测：", candidate_labels[best_idx])
