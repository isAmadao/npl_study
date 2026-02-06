from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification

from config import BERT_MODEL_PERTRAINED_PATH, BERT_MODEL_PKL_PATH, CATEGORY_NAME

# 设备选择，有gpu-》cuda，无gpu-》cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
# 加载模型结构
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=6)
# 加载训练后的参数权重
model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH))
# 模型迁移到设备
model.to(device)

# 数据集封装
class NewsDataset(Dataset):

    def __init__(self, encodings, labels):
        # tokenizer输出
        self.encodings = encodings
        # 标签
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    # 长度
    def __len__(self):
        return len(self.labels)

# 服务接入函数
def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    # 同意输入格式为List[str]
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")

    # import pdb; pdb.set_trace()
    # 文本编码
    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=30)
    # 构建数据集
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    # DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 推理模式
    model.eval()
    # 批量推理
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # 模型输出解析，等价于logits=outputs.logits
        logits = outputs[1]
        # 概率分布空间
        logits = logits.detach().cpu().numpy()
        # 最大概率类别索引
        pred += list(np.argmax(logits, axis=1).flatten())
    # 标签映射，数字标签-》语义标签
    classify_result = [CATEGORY_NAME[x] for x in pred]
    return classify_result
