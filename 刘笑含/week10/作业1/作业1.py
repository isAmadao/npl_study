from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor =  ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

image = Image.open("./data/dog.jpg")
labels = ["小狗", "边牧", "猫", "狐狸", "鱼"]


image_inputs = processor(images = image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)
image_features = image_features.last_hidden_state[:, 0, :]
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)


text_inputs = processor(text=labels, padding=True, return_tensors="pt")
text_features = model.get_text_features(**text_inputs)
text_features = text_features.last_hidden_state[:,0,:]
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)


inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)


top2_idx = probs.argsort(dim=1, descending=True)[0][:2]
for idx in top2_idx:
    print(f"{labels[idx]}: {probs[0][idx].item():.4f}")