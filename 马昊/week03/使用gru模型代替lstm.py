import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===================== 1. 数据加载与预处理 =====================
# 读取数据集（TSV格式，无表头，第0列文本，第1列标签）
dataset = pd.read_csv("../Week03/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 文本列表
string_labels = dataset[1].tolist()  # 标签列表（字符串形式）

# 标签映射：字符串标签 → 数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符映射：字符 → 数字索引（字符级编码，<pad>填充符默认索引0）
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 词表大小（所有唯一字符的数量）
vocab_size = len(char_to_index)
# 文本最大长度：超过截断，不足填充
max_len = 40

# ===================== 2. 自定义数据集类 =====================
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 标签转为长整型张量
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        # 返回数据集总长度
        return len(self.texts)

    def __getitem__(self, idx):
        # 按索引获取单条数据：文本→数字张量 + 标签
        text = self.texts[idx]
        # 字符转索引，截断超长文本
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充不足长度（用<pad>的索引0填充）
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# ===================== 3. GRU分类模型定义 =====================
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        # 嵌入层：字符索引→高维稠密向量（可训练）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU层：提取序列特征（batch_first=True → 输入形状[batch_size, seq_len, embedding_dim]）
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层：GRU输出→分类标签
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播：嵌入→GRU→全连接
        embedded = self.embedding(x)  # [batch_size, max_len, embedding_dim]
        # GRU输出：(输出序列, 最后时刻隐藏状态) → 仅用最后时刻隐藏状态做分类
        gru_out, hidden_state = self.gru(embedded)
        # 挤压维度：hidden_state形状[1, batch_size, hidden_dim] → [batch_size, hidden_dim]
        out = self.fc(hidden_state.squeeze(0))
        return out

# ===================== 4. 模型训练 =====================
# 实例化数据集
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
# 数据加载器（批量加载、打乱数据）
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 超参数设置
embedding_dim = 64    # 嵌入维度
hidden_dim = 128      # GRU隐藏层维度
output_dim = len(label_to_index)  # 输出维度=标签类别数

# 实例化GRU模型
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 损失函数（多分类用交叉熵损失）
criterion = nn.CrossEntropyLoss()
# 优化器（Adam优化器）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练轮数
num_epochs = 4
for epoch in range(num_epochs):
    model.train()  # 切换训练模式
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空梯度（避免累积）
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()
        
        # 打印批次损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item():.4f}")
    
    # 打印轮次平均损失
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# ===================== 5. 预测函数 =====================
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    # 文本预处理：字符转索引、截断/填充
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    # 增加batch维度（模型要求输入带batch维度）
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    model.eval()  # 切换评估模式
    with torch.no_grad():  # 禁用梯度计算（节省内存、加快速度）
        output = model(input_tensor)
    
    # 获取预测标签索引
    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.item()]
    return predicted_label

# 标签反向映射：数字索引→字符串标签
index_to_label = {i: label for label, i in label_to_index.items()}

# ===================== 6. 测试预测 =====================
# 测试示例1
new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 测试示例2
new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
