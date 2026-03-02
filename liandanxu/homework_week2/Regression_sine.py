import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ========== 添加中文字体支持 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成更多数据点，特别是在极值点附近
x = np.linspace(0, 4 * np.pi, 2000)  # 增加数据点数量
y = np.sin(x)

class sineDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 继承nn.Module类，实现本处的网络
class ImprovedSineNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[256, 128, 64], output_dim=1):
        super(ImprovedSineNet, self).__init__()

        # 创建网络层列表
        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))  # 添加dropout防止过拟合
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        # 将所有层组合成一个sequential模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


input_size = 1
hidden_dim = [256, 128, 64]
output_dim = 1

model = ImprovedSineNet(input_size, hidden_dim, output_dim)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # Mini-batch SGD 小批量样本随机梯度下降
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 100
batch = 4

dataset = sineDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

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

# ========== 新增的对比验证代码 ==========
print("\n=== 模型验证和对比 ===")

# 生成测试数据点用于可视化
test_x = np.linspace(0, 4 * np.pi, 200)
test_x_tensor = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)

# 模型预测
model.eval()
with torch.no_grad():
    predictions = model(test_x_tensor).numpy().flatten()

# 计算真实值
true_values = np.sin(test_x)

# 计算评估指标
mse = np.mean((predictions - true_values) ** 2)
mae = np.mean(np.abs(predictions - true_values))

print(f"均方误差 (MSE): {mse:.6f}")
print(f"平均绝对误差 (MAE): {mae:.6f}")

# 可视化对比
plt.figure(figsize=(12, 8))

# 子图1: 完整对比图
plt.subplot(2, 1, 1)
plt.scatter(x, y, alpha=0.6, s=1, color='lightblue', label='训练数据')
plt.plot(test_x, predictions, 'r-', linewidth=2, label='神经网络预测')
plt.plot(test_x, true_values, 'g--', linewidth=2, label='真实sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦函数回归结果对比')
plt.legend()
plt.grid(True)

# 子图2: 误差分析
plt.subplot(2, 1, 2)
errors = predictions - true_values
plt.plot(test_x, errors, 'purple', linewidth=1, label='预测误差')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('x')
plt.ylabel('误差')
plt.title('预测误差分布')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
