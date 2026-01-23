import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.linspace(-np.pi, np.pi, 100)[:, np.newaxis]
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [-pi, pi) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.05 # 噪音*0.05尽量不要影响主体曲线
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层线性模型
class MultilayerClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 层初始化
        super(MultilayerClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

hidden_dim = 128
multilayerModel = MultilayerClassifier(1, hidden_dim, 1)


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD(multilayerModel.parameters(), lr=0.0005) # 优化器，基于 model中所有的参数 自动更新

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    multilayerModel.train()
    # 前向传播
    y_pred = multilayerModel.forward(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("模型参数:")
for name, param in multilayerModel.named_parameters():
    print(f"{name}: shape={param.shape}")
    print(param.data)

# 6. 绘制结果
# 使用最终学到的模型来计算拟合直线的 y 值
multilayerModel.eval()
with torch.no_grad():
    y_predicted = multilayerModel(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Model Prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
