import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据加载和预处理 (保持不变)
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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
        """属性说明"""
        # 文本数据集
        self.texts = texts
        # 类别标签
        self.labels = torch.tensor(labels, dtype=torch.long)
        # 字符索引字典
        self.char_to_index = char_to_index
        # 最大长度
        self.max_len = max_len
        # 词典大小
        self.vocab_size = vocab_size
        # 词向量
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleClassifier, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 添加指定数量的隐藏层
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model_config, model_name):
    """训练模型并返回损失历史"""
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
    
    input_dim = vocab_size
    hidden_dim = model_config['hidden_dim']
    output_dim = len(label_to_index)
    num_layers = model_config['num_layers']
    
    model = SimpleClassifier(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=model_config['lr'])
    
    num_epochs = model_config['epochs']
    loss_history = []
    
    print(f"\n开始训练模型: {model_name}")
    print(f"配置: 隐藏层数={num_layers}, 隐藏节点数={hidden_dim}, 学习率={model_config['lr']}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history, model


# 定义不同的模型配置进行对比
model_configs = {
    "2层_64节点": {
        "hidden_dim": 64,
        "num_layers": 2,
        "lr": 0.01,
        "epochs": 10
    },
    "3层_64节点": {
        "hidden_dim": 64,
        "num_layers": 3,
        "lr": 0.01,
        "epochs": 10
    },
    "4层_64节点": {
        "hidden_dim": 64,
        "num_layers": 4,
        "lr": 0.01,
        "epochs": 10
    },
    "2层_128节点": {
        "hidden_dim": 128,
        "num_layers": 2,
        "lr": 0.01,
        "epochs": 10
    },
    "3层_128节点": {
        "hidden_dim": 128,
        "num_layers": 3,
        "lr": 0.01,
        "epochs": 10
    },
    "4层_128节点": {
        "hidden_dim": 128,
        "num_layers": 4,
        "lr": 0.01,
        "epochs": 10
    }
}

# 训练不同配置的模型并记录损失历史
results = {}
for model_name, config in model_configs.items():
    loss_history, trained_model = train_model(config, model_name)
    results[model_name] = {
        'loss_history': loss_history,
        'model': trained_model
    }

# 绘制对比图
plt.figure(figsize=(15, 10))

# 绘制所有模型的损失曲线
for model_name, result in results.items():
    plt.plot(result['loss_history'], label=model_name, marker='o')

plt.title('不同模型结构的Loss对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 打印各模型最终的loss值
print("\n各模型最终Loss对比:")
for model_name, result in results.items():
    final_loss = result['loss_history'][-1]
    config = model_configs[model_name]
    print(f"{model_name}: 最终Loss = {final_loss:.4f}, 隐藏层数={config['num_layers']}, 隐藏节点数={config['hidden_dim']}")

# 测试最佳模型
best_model_name = min(results.keys(), key=lambda x: results[x]['loss_history'][-1])
best_model = results[best_model_name]['model']

print(f"\n最佳模型: {best_model_name}")

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
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

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
