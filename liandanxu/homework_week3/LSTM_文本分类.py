import pandas as pd
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ==================== 1-7. 预处理部分（完全相同）====================
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()


def cut_text(text_list):
    return [" ".join(jieba.cut(text)) for text in text_list]


tokenized_texts = cut_text(texts)
split_texts = [text.split() for text in tokenized_texts]

word2idx = {"<PAD>": 0, "<UNK>": 1}
for sentence in split_texts:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
print(f"词汇表大小 (vocab_size): {vocab_size}")

seq_len = max(len(sentence) for sentence in split_texts)
print(f"最大句子长度 (seq_len): {seq_len}")


def sentence_to_indices(sentence, word2idx, max_len):
    indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sentence]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


all_indices = [sentence_to_indices(sentence, word2idx, seq_len) for sentence in split_texts]
all_indices = torch.LongTensor(all_indices)
print(f"文本索引张量形状：{all_indices.shape}")

unique_labels = list(set(string_labels))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}
output_size = len(label2idx)

print(f"\n标签映射关系：{label2idx}")
print(f"分类类别数 (output_size): {output_size}")

all_labels = [label2idx[label] for label in string_labels]
all_labels = torch.LongTensor(all_labels)
print(f"标签索引张量形状：{all_labels.shape}")


# ==================== Dataset ====================
class TextDataset(Dataset):
    def __init__(self, all_indices, all_labels):
        self.all_indices = all_indices
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        return self.all_indices[idx], self.all_labels[idx]


rnn_dataset = TextDataset(all_indices, all_labels)
dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)


# ==================== 8. LSTM 模型（关键修改）====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # LSTM 有 cell 状态，能更好地捕捉长依赖
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM 返回 (output, (hidden, cell))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden 形状：[num_layers, batch_size, hidden_dim]

        # 取最后一层的隐藏状态
        hidden = hidden[-1, :, :]

        output = self.fc(hidden)
        return output


# ==================== 9. 超参数 ====================
embedding_dim = 128
hidden_dim = 256
num_layers = 2

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==================== 10. 训练循环 ====================
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch {idx}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, word2idx, idx2label, seq_len):
    """
    单条文本分类预测

    参数:
        text: 输入文本字符串
        model: 训练好的模型 (LSTM/GRU/RNN)
        word2idx: 词到索引的映射字典 (训练时的)
        idx2label: 索引到标签的映射字典 (训练时的)
        seq_len: 最大句子长度 (训练时的)

    返回:
        predicted_label: 预测的标签字符串
    """
    # 1. Jieba 分词
    words = list(jieba.cut(text))

    # 2. 转索引 (未知词用 <UNK>=1)
    indices = [word2idx.get(word, 1) for word in words]

    # 3. Padding 或截断
    if len(indices) < seq_len:
        indices += [0] * (seq_len - len(indices))
    else:
        indices = indices[:seq_len]

    # 4. 转为 Tensor [1, seq_len]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 5. 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # 6. 取概率最大的类别
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = idx2label[predicted_index]

    return predicted_label


new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, word2idx, idx2label, seq_len)
print(f"输入 '{new_text}' LSTM预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, word2idx, idx2label, seq_len)
print(f"输入 '{new_text_2}' LSTM预测为: '{predicted_class_2}'")