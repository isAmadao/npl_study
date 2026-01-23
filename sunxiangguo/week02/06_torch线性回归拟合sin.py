import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 - sin函数
X_numpy = np.linspace(0, 8 * np.pi, 1000).reshape(-1, 1)  # 在[0, 8π]范围内生成1000个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # 添加少量噪声使数据更真实

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print(f"输入数据形状: {X.shape}, 输出数据形状: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.hidden1 = nn.Linear(1, 64)  # 输入层到第一个隐藏层 (1 -> 64)
        self.hidden2 = nn.Linear(64, 32)  # 第一个隐藏层到第二个隐藏层 (64 -> 32)
        self.hidden3 = nn.Linear(32, 16)  # 第二个隐藏层到第三个隐藏层 (32 -> 16)
        self.output = nn.Linear(16, 1)  # 第三个隐藏层到输出层 (16 -> 1)
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        return x


# 创建网络实例
model = SinNet()
print("多层神经网络结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，通常比SGD更适合复杂网络

# 4. 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 5. 测试网络性能
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_predicted = model(X).numpy()

# 计算R²分数来评估拟合效果
ss_res = np.sum((y_numpy - y_predicted) ** 2)  # 残差平方和
ss_tot = np.sum((y_numpy - np.mean(y_numpy)) ** 2)  # 总平方和
r2_score = 1 - (ss_res / ss_tot)
print(f"R² 分数: {r2_score:.4f}")

# 6. 绘制结果
plt.figure(figsize=(12, 8))
plt.scatter(X_numpy, y_numpy, label='原始数据 (含噪声)', color='blue', alpha=0.6, s=20)
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linewidth=2, linestyle='--')
plt.plot(X_numpy, y_predicted, label='神经网络拟合结果', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层神经网络拟合sin函数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
