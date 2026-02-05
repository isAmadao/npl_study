import pandas as pd  # 用于数据读取和处理
import torch  # PyTorch核心库，用于张量计度学算和深习
from sklearn.model_selection import train_test_split  # 把数据集按比例拆分成训练集和测试集（/ 验证集）
#  BertTokenizer: BERT 专属的分词器
#  BertForSequenceClassification: HuggingFace 封装好的、用于序列分类的 BERT 模型
#  Trainer: HuggingFace 封装的标准化训练器 用于简化模型训练代码
#  TrainingArguments: Trainer 的参数配置类，用于定义所有训练相关的超参数和行为
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder  #  标签编码器
from datasets import Dataset  #  HuggingFace 的数据集类
import numpy as np

# -------------------------- 1. 加载并预处理新数据集（THUCNews精简版） --------------------------
# 加载数据集（格式：每行是“类别\t文本”）
dataset_df = pd.read_csv("../Week04/THUCNews.csv", sep="\t", header=None)

# 仅保留4个高频类别：体育、娱乐、科技、教育（解决小样本+类别过多问题）
dataset_df = dataset_df[dataset_df[0].isin(["体育", "娱乐", "科技", "教育"])].reset_index(drop=True)

# 初始化LabelEncoder
print("数据集类别分布：")
print(dataset_df[0].value_counts())

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
dataset_df["label_id"] = lbl.fit_transform(dataset_df[0])

# 可选：仅使用前 N 条数据（例如 500 或 5000）
N_SAMPLES = 500  # 改为 5000 若资源充足
texts = dataset_df[1].iloc[:N_SAMPLES].tolist()
labels = dataset_df["label_id"].iloc[:N_SAMPLES].values.astype(int)  # 确保标签为 int

# 分割训练集/测试集（8:2），保持类别分布
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,       # 文本数据
    labels,             # 对应的数字标签
    test_size=0.2,      # 测试集比例为20%
    stratify=labels     # 确保训练集和测试集的标签分布一致
)

# -------------------------- 2. 加载BERT分词器和模型 --------------------------
# 使用本地模型路径
model_path = 'D:/AI/model/bert-base-chinese'
# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(lbl.classes_)
)

# 编码文本
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 转换为 HuggingFace Dataset（确保 labels 是 list of int）
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],              # 文本的token ID
    'attention_mask': train_encodings['attention_mask'],    # 注意力掩码
    'labels': train_labels.tolist()                         # 转为 list 避免 numpy 类型问题
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels.tolist()
})


# -------------------------- 3. 定义评估指标 --------------------------
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
     # 计算预测准确率并返回一个字典
    return {'accuracy': float((predictions == labels).mean())}


# -------------------------- 4. 配置训练参数 --------------------------
training_args = TrainingArguments(
    output_dir='./thucnews_results',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=9,               # 训练的总轮数
    per_device_train_batch_size=4,   # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=4,    # 评估时每个设备的批次大小
    warmup_steps=10,    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.001,   # 权重衰减，用于防止过拟合
    logging_dir='./thucnews_logs',
    logging_steps=30,       # 每隔100步记录一次日志
    eval_strategy="epoch",   # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",   # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True   # 训练结束后加载效果最好的模型
)

# -------------------------- 5. 训练模型 --------------------------
trainer = Trainer(
    model=model,                       # 要训练的模型
    args=training_args,                # 训练参数
    train_dataset=train_dataset,       # 训练数据集
    eval_dataset=test_dataset,         # 评估数据集
    compute_metrics=compute_metrics,   # 用于计算评估指标的函数
)

print("开始训练...")
trainer.train() # 开始训练模型

print("最终评估...")
eval_result = trainer.evaluate() # 在测试集上进行最终评估
print("评估结果：", eval_result)

# -------------------------- 6. 新样本测试（核心验证步骤） --------------------------
# 将模型移至设备（自动检测 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式


def predict_new_sample(text):
    """预测单个新样本的类别"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )
    # 移动到对应设备
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    pred_label_id = torch.argmax(logits, dim=1).item()
    pred_label = lbl.inverse_transform([pred_label_id])[0]
    return pred_label


# 测试示例
test_samples = [
    "湖人击败凯尔特人，詹姆斯砍下30分8篮板5助攻",
    "2024年国考报名人数突破300万，热门岗位竞争比超5000:1",
    "iPhone 16发布：搭载A18芯片，支持卫星通信",
    "周杰伦新专辑《最伟大的作品》销量破亿",
]

print("\n===== 新样本测试结果 =====")
for sample in test_samples:
    try:
        pred = predict_new_sample(sample)
        print(f"输入文本：{sample}")
        print(f"预测类别：{pred}\n")
    except Exception as e:
        print(f"预测出错：{e}\n")
