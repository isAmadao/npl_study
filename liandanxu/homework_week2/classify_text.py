import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)

# 反向字典，数字 -》 字
index_to_char = {i: char for char, i in char_to_index.items()}

# 建立一个字典，输出类别 -》 数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 建立一个字典，数字 -》 输出类别
index_to_label = {i: label for label, i in label_to_index.items()}
# 将训练集对应的类别转换成数字
numerical_labels = [label_to_index[label] for label in string_labels]


def create_bow_vector(text, vocab_size, char_to_index):
    bow_vector = torch.zeros(vocab_size)
    for char in text:
        bow_vector[char_to_index[char]] += 1
    return bow_vector


def create_bow_vectors(texts, vocab_size, char_to_index):
    bow_vectors = []
    for text in texts:
        bow_vector = create_bow_vector(text, vocab_size, char_to_index)
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors)


# 继承nn.Module类，实现本处的网络
class SimpleClassifier(nn.Module):
    # 构造函数中声明可能用到的组件
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()  #调用父类的构造函数，python子类构造时不自动调用父类构造函数
        self.fc1 = nn.Linear(input_dim, hidden_dim)  #线性变换：W₁x + b₁
        self.relu = nn.ReLU()  # ax(0, out)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    # 定义前向传播过程，即网络结构
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = vocab_size
hidden_dim = 256
output_dim = len(label_to_index)

model = SimpleClassifier(input_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # 分类损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Mini-batch SGD 小批量样本随机梯度下降


class CharBoWDataset(Dataset):
    def __init__(self, bow_vectors, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.bow_vectors = bow_vectors

    def __len__(self):
        return len(self.bow_vectors)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

bow_vectors = create_bow_vectors(texts, vocab_size, char_to_index)
char_dataset = CharBoWDataset(bow_vectors, numerical_labels)


num_epochs = 10
batch = 16
dataloader = DataLoader(char_dataset, batch_size=batch, shuffle=True)

for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式，开启Dropout功能
    running_loss = 0.0 # 统计每个epoch的损失
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad() #PyTorch 默认会累加梯度，用此接口清零梯度
        outputs = model(inputs) # 前向计算
        loss = criterion(outputs, labels) # 计算损失值
        loss.backward() # 计算梯度
        optimizer.step() # 更新参数
        running_loss += loss.item() # 累加本batch的loss值
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, index_to_label):
    bow_vector = create_bow_vector(text, vocab_size, char_to_index)
    model.eval() # 将模型转为推理，禁用Dropout
    with torch.no_grad(): # 不构建计算图，节省内存
        output = model(bow_vector)
    _, predicted_index = torch.max(output, dim=0)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")