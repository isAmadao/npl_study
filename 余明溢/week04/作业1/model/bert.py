from typing import Union, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,BertForSequenceClassification
import pickle
from config import label_encoder_path,BERT_BASE_CN_DIR,BERT_MODEL_PKL_PATH


# 加载标签编码器
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

CATEGORY_NAME = label_encoder.classes_.tolist()
print(f"类别标签: {CATEGORY_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_CN_DIR)
model = BertForSequenceClassification.from_pretrained(BERT_BASE_CN_DIR, num_labels=7)

model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH))
model.to(device)



class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")



    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=32)
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())

    classify_result = [CATEGORY_NAME[x] for x in pred]
    return classify_result

if __name__ == '__main__':
    print(model_for_bert([
        "你好，支付失败了怎么办",
        "我的订单怎么还没发货",
        "这个商品可以退货吗",
        "有没有优惠券可以用",
        "账号密码忘记了怎么找回",
        "我要投诉客服态度不好",
        "这个手机多少钱"
    ]))