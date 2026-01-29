import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out
# --- NEW RNN Model Class ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- NEW GRU Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
# lstm
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
# rnn
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
# gru
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)



num_epochs = 10
# --- 定义模型训练和评估函数 ---
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_and_evaluate(model, dataloader, num_epochs, criterion, optimizer):
    loss_history = []
    print(f"--- 训练模型，参数量: {get_parameter_count(model)} ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"{model.__class__.__name__} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return loss_history
# for epoch in range(num_epochs):
#     lstm_model.train()
#     running_loss = 0.0
#     for idx, (inputs, labels) in enumerate(dataloader):
#         lstm_optimizer.zero_grad()
#         outputs = lstm_model(inputs)
#         loss = lstm_criterion(outputs, labels)
#         loss.backward()
#         lstm_optimizer.step()
#         running_loss += loss.item()
#         lstm_loss_dist[idx] = loss.item()
        # if idx % 50 == 0:
        #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
# print(lstm_loss_dist)
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
#     indices = [char_to_index.get(char, 0) for char in text[:max_len]]
#     indices += [0] * (max_len - len(indices))
#     input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
#
#     model.eval()
#     with torch.no_grad():
#         output = model(input_tensor)
#
#     _, predicted_index = torch.max(output, 1)
#     predicted_index = predicted_index.item()
#     predicted_label = index_to_label[predicted_index]
#
#     return predicted_label
#
# index_to_label = {i: label for label, i in label_to_index.items()}

# 训练并记录损失
loss_lstm = train_and_evaluate(lstm_model, dataloader, num_epochs, lstm_criterion, lstm_optimizer)
loss_rnn = train_and_evaluate(rnn_model, dataloader, num_epochs, rnn_criterion, rnn_optimizer)
loss_gru = train_and_evaluate(gru_model, dataloader, num_epochs, gru_criterion, gru_optimizer)

# --- 绘制损失曲线 ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_lstm, label=f'lstm(Parameters: {get_parameter_count(lstm_model)})')
plt.plot(range(1, num_epochs + 1), loss_rnn, label=f'rnn(Parameters: {get_parameter_count(rnn_model)})')
plt.plot(range(1, num_epochs + 1), loss_gru, label=f'gru(Parameters: {get_parameter_count(gru_model)})')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# new_text = "帮我导航到北京"
# predicted_class = classify_text_lstm(new_text, lstm_model, char_to_index, max_len, index_to_label)
# print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
#
# new_text_2 = "查询明天北京的天气"
# predicted_class_2 = classify_text_lstm(new_text_2, lstm_model, char_to_index, max_len, index_to_label)
# print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")