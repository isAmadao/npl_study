import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil

# ==========================================
# 1. 数据集处理
# ==========================================
dataset_dir = "./assets/dataset/clue-tnews"
print(f"检查本地数据集目录: {dataset_dir}")

if os.path.exists(dataset_dir):
    print("发现本地数据集，直接加载...")
    dataset = load_from_disk(dataset_dir)
else:
    print("本地数据集不存在，开始从Hugging Face下载...")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset = load_dataset("clue", "tnews")
    dataset.save_to_disk(dataset_dir)
    print(f"数据集已保存到本地: {dataset_dir}")

# 获取标签信息
num_labels = dataset['train'].features['label'].num_classes
label_list = dataset['train'].features['label'].names

print(f"数据集类别数量: {num_labels}")

# ==========================================
# 2. 数据预处理
# ==========================================
model_dir = "./assets/models/bert-base-chinese/"
print("正在初始化分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese",
    cache_dir=model_dir
)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("正在进行数据分词...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# ==========================================
# 3. 加载本地BERT模型（15 分类）
# ==========================================
print(f"正在加载模型... (如果本地不完整，将自动从 Hub 下载完整版)")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels,
    cache_dir=model_dir,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f"当前使用设备: {device}")

# ==========================================
# 4. 定义评估指标
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc}

# ==========================================
# 5. 训练参数（唯一修改点：evaluation_strategy -> eval_strategy）
# ==========================================
training_args = TrainingArguments(
    output_dir='./assets/results/tnews',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./assets/logs',
    logging_steps=50,
    eval_strategy="epoch",          # 修改为 eval_strategy（兼容新版 transformers）
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=3e-5,
    report_to="none"
)

# ==========================================
# 6. 训练模型
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("开始训练模型...")
trainer.train()

print("\n训练结束，正在在验证集上进行评估...")
eval_results = trainer.evaluate()
print(f"验证集评估结果: {eval_results}")

# ==========================================
# 7. 保存模型与预测
# ==========================================
print("\n正在保存微调后的模型...")
model_save_path = "./assets/models/tnews-bert-classifier"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return label_list[predicted_class_id]

print("\n" + "="*30)
print("预测测试：")
print("="*30)
sample_1 = "中国女排在奥运会决赛中以3比0战胜对手，夺得金牌。"
print(f"文本: {sample_1} -> 分类: {predict_text(sample_1)}")

sample_2 = "华为发布了最新款的人工智能芯片，算力提升了数倍。"
print(f"文本: {sample_2} -> 分类: {predict_text(sample_2)}")

sample_3 = "央行决定下调存款准备金率0.5个百分点，释放长期资金约1万亿元。"
print(f"文本: {sample_3} -> 分类: {predict_text(sample_3)}")
