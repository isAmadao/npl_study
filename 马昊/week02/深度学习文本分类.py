import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 用于可视化loss

# 数据加载和预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签转数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符转索引（构建字符表）
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


# 构建BoW数据集类
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            # 截断/补齐到max_len
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 构建BoW向量（词袋：统计字符出现次数）
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:  # 跳过pad字符
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 重构后的灵活分类器（支持自定义层数和各层节点数）
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        灵活的全连接分类器
        :param input_dim: 输入维度（这里是字符表大小）
        :param hidden_dims: 隐藏层节点数列表，比如[128, 64]表示2个隐藏层，分别128和64个节点
        :param output_dim: 输出维度（类别数）
        """
        super(FlexibleClassifier, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 每个隐藏层后加ReLU激活
            prev_dim = hidden_dim
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        # 把所有层组合成Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 初始化数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# --------------------------
# 重点：调整这里的参数来改变模型结构
# --------------------------
# 示例1：1层隐藏层，128个节点（和你原模型一致）
# hidden_dims = [128]
# 示例2：2层隐藏层，第一层128节点，第二层64节点
hidden_dims = [256,128,64]
# 示例3：3层隐藏层，节点数依次递减
# hidden_dims = [256, 128, 64]

output_dim = len(label_to_index)
# 初始化模型（现在用重构后的FlexibleClassifier）
model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务常用，内部会处理softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练参数
num_epochs = 10
# 用于记录每个epoch的loss（可视化用）
epoch_losses = []

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播+优化
        loss.backward()
        optimizer.step()
        # 累计损失
        running_loss += loss.item()

        if idx % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx}], Loss: {loss.item():.4f}")

    # 计算当前epoch的平均loss并记录
    avg_epoch_loss = running_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    print(f"===== Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f} =====")

# --------------------------
# Loss可视化
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title(f'Model Loss Over Epochs (Hidden Layers: {len(hidden_dims)}, Nodes: {hidden_dims})')
plt.grid(True)
plt.legend()
plt.savefig('loss_curve.png')  # 保存图片到本地
plt.show()  # 显示图片


# 预测函数（和原代码一致，仅调整模型调用）
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 预处理输入文本
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 构建BoW向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 增加batch维度（模型需要batch输入）
    bow_vector = bow_vector.unsqueeze(0)

    # 预测（关闭梯度计算）
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.item()]

    return predicted_label


# 标签反向映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试预测
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
