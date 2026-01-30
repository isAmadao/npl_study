import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 数据加载和预处理
dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
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

# 2. 自定义数据集类
class CharRNNDataset(Dataset):
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

# 3. 数据划分
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

# 4. 模型定义 - RNN
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

# 5. 模型定义 - LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# 6. 模型定义 - GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

# 7. 训练函数
def train_and_evaluate(model_class, model_name, num_epochs=4):
    print(f"\n{'='*60}")
    print(f"训练 {model_name} 模型")
    print(f"{'='*60}")
    
    # 创建数据集
    train_dataset = CharRNNDataset(train_texts, train_labels, char_to_index, max_len)
    test_dataset = CharRNNDataset(test_texts, test_labels, char_to_index, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    
    # 创建模型
    model = model_class(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return accuracy

# 8. 预测函数
def classify_text(model, text, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    
    return predicted_label

# 9. 对比实验
index_to_label = {i: label for label, i in label_to_index.items()}

results = {}

# 训练RNN
rnn_acc = train_and_evaluate(RNNClassifier, "RNN", num_epochs=4)
results['RNN'] = rnn_acc

# 训练LSTM
lstm_acc = train_and_evaluate(LSTMClassifier, "LSTM", num_epochs=4)
results['LSTM'] = lstm_acc

# 训练GRU
gru_acc = train_and_evaluate(GRUClassifier, "GRU", num_epochs=4)
results['GRU'] = gru_acc

# 10. 结果对比
print(f"\n{'='*60}")
print("模型精度对比结果")
print(f"{'='*60}")
for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:10s}: {acc:.4f} ({acc*100:.2f}%)")

best_model = max(results, key=results.get)
print(f"\n最佳模型: {best_model} - {results[best_model]:.4f}")

# 11. 测试示例
print(f"\n{'='*60}")
print("测试示例")
print(f"{'='*60}")

# 重新训练最佳模型用于测试
if best_model == 'RNN':
    final_model = RNNClassifier(vocab_size, 64, 128, len(label_to_index))
elif best_model == 'LSTM':
    final_model = LSTMClassifier(vocab_size, 64, 128, len(label_to_index))
else:
    final_model = GRUClassifier(vocab_size, 64, 128, len(label_to_index))

optimizer = optim.Adam(final_model.parameters(), lr=0.001)
train_dataset = CharRNNDataset(train_texts, train_labels, char_to_index, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    final_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试
test_texts_examples = ["帮我导航到北京", "查询明天北京的天气"]
for text in test_texts_examples:
    predicted = classify_text(final_model, text, char_to_index, max_len, index_to_label)
    print(f"输入 '{text}' 预测为: '{predicted}'")
