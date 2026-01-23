import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 改进的模型类，支持可配置的层数和节点数
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表，例如 [128, 64, 32] 表示三层隐藏层
        output_dim: 输出维度
        """
        super(FlexibleClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建多层网络
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model_config, dataloader, num_epochs=10):
    """
    训练模型并返回loss历史记录
    """
    input_dim, hidden_dims, learning_rate = model_config

    model = FlexibleClassifier(input_dim, hidden_dims, output_dim=len(label_to_index))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Model {hidden_dims} - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history, model


# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义不同的模型配置进行对比
model_configs = [
    (vocab_size, [64], 0.001),  # 单隐藏层，64节点
    (vocab_size, [128], 0.001),  # 单隐藏层，128节点
    (vocab_size, [256], 0.001),  # 单隐藏层，256节点
    (vocab_size, [128, 64], 0.001),  # 双隐藏层，128->64
    (vocab_size, [256, 128], 0.001),  # 双隐藏层，256->128
    (vocab_size, [128, 64, 32], 0.001),  # 三隐藏层，128->64->32
]

# 存储每个模型的loss历史
all_loss_histories = {}
trained_models = {}

print("开始训练不同配置的模型...")
for i, config in enumerate(model_configs):
    print(f"\n训练模型配置 {i + 1}: {config[1]}")
    loss_history, trained_model = train_model(config, dataloader, num_epochs=10)
    all_loss_histories[f"Model_{config[1]}"] = loss_history
    trained_models[f"Model_{config[1]}"] = trained_model

# 可视化loss变化
plt.figure(figsize=(12, 8))
for model_name, loss_history in all_loss_histories.items():
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label=model_name)

plt.title('不同模型配置的训练Loss对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 打印每个模型的最终loss
print("\n各模型最终Loss对比:")
for model_name, loss_history in all_loss_histories.items():
    final_loss = loss_history[-1]
    print(f"{model_name}: {final_loss:.4f}")

# 选择最佳模型进行预测演示
best_model_key = min(all_loss_histories.keys(), key=lambda k: all_loss_histories[k][-1])
best_model = trained_models[best_model_key]
print(f"\n最佳模型: {best_model_key}, 最终Loss: {all_loss_histories[best_model_key][-1]:.4f}")


# 预测函数
def classify_text_with_trained_model(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

# 测试最佳模型
new_text = "帮我导航到北京"
predicted_class = classify_text_with_trained_model(new_text, best_model, char_to_index, vocab_size, max_len,
                                                   index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_with_trained_model(new_text_2, best_model, char_to_index, vocab_size, max_len,
                                                     index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
