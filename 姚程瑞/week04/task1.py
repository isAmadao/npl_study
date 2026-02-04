import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 加载本地20 Newsgroups数据集
def load_local_20newsgroups(data_path):
    categories = []
    texts = []
    labels = []
    
    for category in sorted(os.listdir(data_path)):
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            categories.append(category)
            category_idx = len(categories) - 1
            
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append(category_idx)
    
    return texts, np.array(labels), categories

# 加载数据
data_path = r"d:\GZU\badou\作业\姚程瑞\week04\20news-18828"
texts, targets, target_names = load_local_20newsgroups(data_path)

# 划分训练集和测试集
train_texts, test_texts, train_targets, test_targets = train_test_split(
    texts, targets, test_size=0.2, random_state=42, stratify=targets
)

# 进一步划分训练集和验证集
train_texts, val_texts, train_targets, val_targets = train_test_split(
    train_texts, train_targets, test_size=0.2, random_state=42, stratify=train_targets
)

print(f"训练样本数: {len(train_texts)}")
print(f"验证样本数: {len(val_texts)}")
print(f"测试样本数: {len(test_texts)}")
print(f"类别数: {len(target_names)}")

# 定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 检查本地是否存在BERT模型
local_model_path = r"D:\GZU\badou\models\google-bert\bert-base-uncased"

if os.path.exists(local_model_path):
    # 从本地加载模型
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertForSequenceClassification.from_pretrained(
        local_model_path,
        num_labels=len(target_names)
    )
    print("从本地加载BERT模型成功")
else:
    # 从网络加载模型（仅在有网络连接时）
    print("本地未找到BERT模型，尝试从网络加载...")
    print("注意：如遇网络问题，请先在网络环境良好的情况下下载模型到本地")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(target_names)
        )
        print("成功加载BERT模型")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("使用ModelScope下载：")
        print("modelscope download --model google-bert/bert-base-uncased")
        print("将模型保存到：D:\\GZU\\badou\\models\\google-bert\\bert-base-uncased")
        exit()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建数据集和加载器
train_dataset = NewsDataset(train_texts, train_targets, tokenizer)
val_dataset = NewsDataset(val_texts, val_targets, tokenizer)
test_dataset = NewsDataset(test_texts, test_targets, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 训练函数
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# 评估函数
def eval_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(true_labels, predictions)

# 微调模型
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3个epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

for epoch in range(3):
    print(f"Epoch {epoch+1}/3")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_acc = eval_model(model, val_loader)
    print(f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

# 测试模型
test_acc = eval_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

# 预测新样本
def predict_new_text(text, model, tokenizer, target_names, device):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1).max().item()
    
    return target_names[prediction], confidence

# 测试新样本
new_sample = """
I am having trouble with my computer. It keeps freezing and crashing.
The screen goes black randomly and I lose all my work. Can anyone help me?
"""
predicted_category, confidence = predict_new_text(new_sample, model, tokenizer, target_names, device)
print(f"新样本预测: {predicted_category}, 置信度: {confidence:.4f}")

# 测试另一个新样本
new_sample2 = """
The basketball team won the championship game last night.
They played an amazing game and the crowd went wild with excitement.
"""
predicted_category2, confidence2 = predict_new_text(new_sample2, model, tokenizer, target_names, device)
print(f"新样本预测: {predicted_category2}, 置信度: {confidence2:.4f}")
