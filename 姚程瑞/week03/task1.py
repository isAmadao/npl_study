import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# 1. 数据准备
file_path = 'jaychou_lyrics.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = set(f.readlines())
    text = " ".join(lines)

# 找出所有的独立字符并创建映射
vocab = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
vocab_size = len(vocab)

# 转换为整数序列
text_as_int = np.array([char_to_idx[c] for c in text])


# 2. 定义数据集和数据加载器
class LyricsDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.data_size = len(text) - seq_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # 输入序列和目标序列
        # 输入X1 - X10    从反方向开始移
        # 输出X2 - X11    反方向开始移动
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))


# 创建数据集和数据加载器
seq_length = 100
dataset = LyricsDataset(text_as_int, seq_length)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 3. 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = output.reshape(-1, self.hidden_dim)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)


# 4. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = output.reshape(-1, self.hidden_dim)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))


# 5. 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = output.reshape(-1, self.hidden_dim)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)


# 6. 训练和评估函数
def train_model(model, model_name, model_path, device, epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    if os.path.exists(model_path):
        print(f"{model_name}: 载入已有的模型权重...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        training_time = 0
    else:
        print(f"{model_name}: 开始训练...")
        for epoch in range(epochs):
            total_loss = 0
            for i, (inputs, targets) in enumerate(dataloader):
                batch_size = inputs.size(0)
                
                if model_name == "LSTM":
                    hidden = model.init_hidden(batch_size)
                    hidden = tuple([h.to(device) for h in hidden])
                    hidden = tuple([h.data for h in hidden])
                else:  # RNN or GRU
                    hidden = model.init_hidden(batch_size)
                    hidden = hidden.to(device)
                    hidden = hidden.data
                
                inputs, targets = inputs.to(device), targets.to(device)

                model.zero_grad()
                logits, hidden = model(inputs, hidden)
                loss = criterion(logits, targets.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    print(f'{model_name} - Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        # 保存模型
        torch.save(model.state_dict(), model_path)
        print(f"{model_name}模型已保存至 {model_path}")
    
    training_time = time.time() - start_time
    
    return model, training_time


def evaluate_model(model, model_name, start_string="枫叶", num_generate=100):
    model.eval()
    device = next(model.parameters()).device
    
    # 将起始字符串转换为张量
    input_eval = torch.tensor([char_to_idx[s] for s in start_string], dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_string
    
    with torch.no_grad():
        if model_name == "LSTM":
            hidden = model.init_hidden(1)
        else:  # RNN or GRU
            hidden = model.init_hidden(1)
            hidden = hidden.to(device)

        start_gen_time = time.time()
        for _ in range(num_generate):
            input_eval = input_eval.view(1, -1)
            logits, hidden = model(input_eval, hidden)
            logits = logits.squeeze(1)
            predicted_id = torch.argmax(logits[0], dim=-1).item()
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)
            generated_text += idx_to_char[predicted_id]
        generation_time = time.time() - start_gen_time

    return generated_text, generation_time


# 7. 运行实验
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 32
hidden_dim = 128

print("开始对比实验...")

# 实验结果存储
results = {}

# RNN 模型实验
print("\n" + "="*50)
print("开始 RNN 模型实验...")
rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)
rnn_model, rnn_train_time = train_model(rnn_model, "RNN", 'rnn_experiment_model.pt', device)
rnn_generated, rnn_gen_time = evaluate_model(rnn_model, "RNN")
results['RNN'] = {'train_time': rnn_train_time, 'gen_time': rnn_gen_time, 'generated': rnn_generated}
print(f"RNN 训练时间: {rnn_train_time:.2f}s")
print(f"RNN 生成时间: {rnn_gen_time:.2f}s")

# LSTM 模型实验
print("\n" + "="*50)
print("开始 LSTM 模型实验...")
lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
lstm_model, lstm_train_time = train_model(lstm_model, "LSTM", 'lstm_experiment_model.pt', device)
lstm_generated, lstm_gen_time = evaluate_model(lstm_model, "LSTM")
results['LSTM'] = {'train_time': lstm_train_time, 'gen_time': lstm_gen_time, 'generated': lstm_generated}
print(f"LSTM 训练时间: {lstm_train_time:.2f}s")
print(f"LSTM 生成时间: {lstm_gen_time:.2f}s")

# GRU 模型实验
print("\n" + "="*50)
print("开始 GRU 模型实验...")
gru_model = GRUModel(vocab_size, embedding_dim, hidden_dim).to(device)
gru_model, gru_train_time = train_model(gru_model, "GRU", 'gru_experiment_model.pt', device)
gru_generated, gru_gen_time = evaluate_model(gru_model, "GRU")
results['GRU'] = {'train_time': gru_train_time, 'gen_time': gru_gen_time, 'generated': gru_generated}
print(f"GRU 训练时间: {gru_train_time:.2f}s")
print(f"GRU 生成时间: {gru_gen_time:.2f}s")

# 8. 显示最终比较结果
print("\n" + "="*60)
print("模型性能对比总结")
print("="*60)
print(f"{'模型':<10} {'训练时间(s)':<12} {'生成时间(s)':<12}")
print("-"*40)
for model_name, result in results.items():
    print(f"{model_name:<10} {result['train_time']:<12.2f} {result['gen_time']:<12.2f}")

print("\n" + "="*60)
print("生成的歌词示例")
print("="*60)
for model_name, result in results.items():
    print(f"\n--- {model_name} 生成的歌词 ---")
    print(result['generated'][:200] + "..." if len(result['generated']) > 200 else result['generated'])
