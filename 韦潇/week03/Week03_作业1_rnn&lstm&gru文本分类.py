import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# ===================== 1. 配置全局参数 =====================
# 设备配置（优先GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# 训练参数（统一配置，保证对比公平）
BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MAX_LEN = 40  # 文本最大长度


# ===================== 2. 数据加载与预处理 =====================
# 加载数据集
def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    """加载数据集，返回文本列表和标签列表"""
    dataset = pd.read_csv(file_path, sep="\t", header=None)
    texts = dataset[0].tolist()
    labels = dataset[1].tolist()

    # 数据清洗：处理空值/非字符串
    texts = [text if isinstance(text, str) else "" for text in texts]
    labels = [label if isinstance(label, str) else "" for label in labels]
    return texts, labels


# 编码工具函数
def build_vocab(texts: List[str]) -> Tuple[Dict[str, int], int]:
    """构建字符到索引的映射，返回char_to_index和词汇表大小"""
    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)
    vocab_size = len(char_to_index)
    return char_to_index, vocab_size


def encode_labels(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """编码标签，返回label_to_index、index_to_label、数值标签列表"""
    unique_labels = list(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    numerical_labels = [label_to_index[label] for label in labels]
    return label_to_index, index_to_label, numerical_labels


# 执行数据加载和编码
texts, string_labels = load_data("../Week01/dataset.csv")
char_to_index, vocab_size = build_vocab(texts)
label_to_index, index_to_label, numerical_labels = encode_labels(string_labels)
output_dim = len(label_to_index)

print(f"数据加载完成：")
print(f"- 样本数：{len(texts)}")
print(f"- 字符表大小：{vocab_size}")
print(f"- 标签类别数：{output_dim} ({list(label_to_index.keys())})")


# ===================== 3. 自定义Dataset =====================
class CharRNNDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], char_to_index: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        # 字符转索引 + 截断
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充到固定长度
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 创建数据集和数据加载器
dataset = CharRNNDataset(texts, numerical_labels, char_to_index, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ===================== 4. 定义三种循环神经网络模型 =====================
class RNNClassifier(nn.Module):
    """基础RNN分类器"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 基础RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        rnn_out, hidden = self.rnn(embedded)  # hidden: [1, batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))  # [batch, output_dim]
        return out


class LSTMClassifier(nn.Module):
    """LSTM分类器（原模型）"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)  # hidden: [1, batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))  # [batch, output_dim]
        return out


class GRUClassifier(nn.Module):
    """GRU分类器"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU层
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        gru_out, hidden = self.gru(embedded)  # hidden: [1, batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))  # [batch, output_dim]
        return out


# ===================== 5. 训练与评估函数 =====================
def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, num_epochs: int) -> Tuple[List[float], List[float]]:
    """
    训练模型并返回损失和准确率历史
    :return: (loss_history, acc_history)
    """
    model = model.to(DEVICE)
    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            # 数据移到指定设备
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算本轮平均损失和准确率
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        # 打印训练信息
        model_name = model.__class__.__name__
        print(f"[{model_name}] Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return loss_history, acc_history


# ===================== 6. 执行对比实验 =====================
# 初始化模型、损失函数、优化器
models = {
    "RNN": RNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim),
    "LSTM": LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim),
    "GRU": GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim)
}

# 存储实验结果
results = {
    "loss": {},
    "acc": {}
}

print("\n========== 开始对比实验 ==========")
for model_name, model in models.items():
    print(f"\n训练 {model_name} 模型：")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_hist, acc_hist = train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS)
    results["loss"][model_name] = loss_hist
    results["acc"][model_name] = acc_hist

# ===================== 7. 结果可视化 =====================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 颜色配置
colors = {"RNN": "red", "LSTM": "blue", "GRU": "green"}
markers = {"RNN": "o", "LSTM": "s", "GRU": "^"}

# 子图1：损失曲线对比
ax1.set_title("三种模型训练损失对比", fontsize=14)
ax1.set_xlabel("训练轮数 (Epoch)", fontsize=12)
ax1.set_ylabel("平均损失 (Loss)", fontsize=12)
for model_name, loss_hist in results["loss"].items():
    ax1.plot(range(1, NUM_EPOCHS + 1), loss_hist,
             color=colors[model_name], marker=markers[model_name],
             label=model_name, linewidth=2)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 子图2：准确率曲线对比
ax2.set_title("三种模型分类准确率对比", fontsize=14)
ax2.set_xlabel("训练轮数 (Epoch)", fontsize=12)
ax2.set_ylabel("分类准确率 (Accuracy)", fontsize=12)
for model_name, acc_hist in results["acc"].items():
    ax2.plot(range(1, NUM_EPOCHS + 1), acc_hist,
             color=colors[model_name], marker=markers[model_name],
             label=model_name, linewidth=2)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 1.05)  # 准确率范围0-1

# 保存图片
plt.tight_layout()
plt.savefig("rnn_lstm_gru_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 8. 最终结果汇总 =====================
print("\n========== 实验结果汇总 ==========")
final_results = {}
for model_name in models.keys():
    final_loss = results["loss"][model_name][-1]
    final_acc = results["acc"][model_name][-1]
    final_results[model_name] = {"loss": final_loss, "acc": final_acc}
    print(f"{model_name} - 最终损失：{final_loss:.4f}，最终准确率：{final_acc:.4f}")

# 找出最优模型
best_model = max(final_results.items(), key=lambda x: x[1]["acc"])
print(f"\n最优模型：{best_model[0]}（准确率：{best_model[1]['acc']:.4f}）")


# ===================== 9. 预测函数（可选） =====================
def predict_text(model: nn.Module, text: str, char_to_index: Dict[str, int],
                 max_len: int, index_to_label: Dict[int, str]) -> str:
    """使用训练好的模型预测文本类别"""
    model.eval()
    # 文本预处理
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = index_to_label[predicted_idx.item()]

    return predicted_label


# 测试最优模型
print("\n========== 最优模型预测测试 ==========")
best_model_instance = models[best_model[0]]
test_texts = ["帮我导航到北京", "查询明天北京的天气"]
for text in test_texts:
    pred_label = predict_text(best_model_instance, text, char_to_index, MAX_LEN, index_to_label)
    print(f"输入：'{text}' 预测为：'{pred_label}'")
