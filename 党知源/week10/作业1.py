import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "AI-ModelScope/chinese-clip-vit-base-patch16"
MODEL_DIR = os.path.join(_ROOT, "model", "chinese-clip-vit-base-patch16")


def ensure_chinese_clip_model() -> str:
    cfg = os.path.join(MODEL_DIR, "config.json")
    if os.path.isfile(cfg):
        return MODEL_DIR
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError(
            "未检测到本地模型"
        ) from e
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    try:
        snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    except TypeError:
        p = snapshot_download(MODEL_ID, cache_dir=os.path.join(_ROOT, "model"))
        if not os.path.isfile(os.path.join(p, "config.json")):
            raise RuntimeError("请检查 cache 路径: " + str(p))
        return p
    if not os.path.isfile(cfg):
        raise RuntimeError(f"缺少 config.json，请检查目录: {MODEL_DIR}")
    return MODEL_DIR


# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model_path = ensure_chinese_clip_model()
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=False)
model.eval()
_device = next(model.parameters()).device
print("模型加载完成，路径:", model_path)


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def clip_text_features(model, processor, texts, device) -> np.ndarray:
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = _batch_to_device(inputs, device)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            return_dict=True,
        )
        cls = text_out.last_hidden_state[:, 0, :]
        proj = model.text_projection(cls)
    return proj.cpu().numpy()


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


IMAGE_PATH = os.path.join(_ROOT, "data", "dog.jpg")
CLASS_LABELS = [
    "一只小狗",
    "一只小猫",
    "一只鸟",
    "一辆汽车",
    "一片风景",
    "一份食物",
]

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"未找到图片: {IMAGE_PATH}\n请将图片放到该路径，或修改 IMAGE_PATH。"
    )

pil_img = Image.open(IMAGE_PATH).convert("RGB")
plt.figure(figsize=(5, 5))
plt.imshow(pil_img)
plt.axis("off")
plt.title("输入图像")
plt.show()

with torch.no_grad():
    img_inputs = _batch_to_device(
        processor(images=pil_img, return_tensors="pt"), _device
    )
    image_feat = model.get_image_features(**img_inputs).detach().cpu().numpy()
    image_feat = normalize(image_feat)

    text_feat = clip_text_features(model, processor, CLASS_LABELS, _device)
    text_feat = normalize(text_feat)

logits = np.dot(image_feat, text_feat.T).ravel()

pred_idx = int(np.argmax(logits))
pred_label = CLASS_LABELS[pred_idx]

print("Zero-shot 预测类别:", pred_label)
print("\n各类别相似度 (越大越匹配):")
for i, name in enumerate(CLASS_LABELS):
    print(f"  {logits[i]:8.4f}  {name}")

fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(CLASS_LABELS))
ax.barh(y_pos, logits, color="steelblue")
ax.set_yticks(y_pos)
ax.set_yticklabels(CLASS_LABELS)
ax.invert_yaxis()
ax.set_xlabel("图文相似度 (归一化特征点积)")
ax.set_title(f"预测: {pred_label}")
plt.tight_layout()
plt.show()
