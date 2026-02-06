#2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

### 原始数据-》torch支持的格式tensor
# 1. 生成sin非线性模拟数据（核心改动：替换原线性数据，加少量噪声更贴合真实场景）
# 生成0~2π均匀点（sin一个完整周期），200个样本保证曲线覆盖完整，形状(200,1)
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
# 生成y=sin(x)+0.1倍高斯噪声，模拟真实带噪声数据，考验非线性拟合能力
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)

#转换成torch支持的设备更多一些，可以使用gpu/cpu计算
#torch的输入只能是tensor格式
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("Sin函数模拟数据生成完成，形状：X={}, y={}".format(X.shape, y.shape))
print("---" * 10)

### 搭建多层全连接网络（核心改动：替换原a/b单一线性模型，用nn.Sequential搭多层非线性网络）
#### 模型参数：网络各层的权重/偏置（由PyTorch自动创建，无需手动定义）
#### 模型的超参数：网络结构(1→32→16→1)、激活函数、学习率、损失函数、epoch
#### 调参-》人工调整模型的超参数，进行实验，对比挑选
# 用nn.Sequential按顺序搭建多层网络，贴合原代码高层API风格，无需自定义类
# 网络结构：输入层(1维) → 隐藏层1(32维)+tanh激活 → 隐藏层2(16维)+tanh激活 → 输出层(1维)
# 激活函数：非线性，打破线性映射，让网络能拟合sin曲线（线性网络无法拟合非线性函数）
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=32),  # 输入1维→隐藏层1 32维
    torch.nn.Tanh(),                                   # 非线性激活函数，可选ReLU/LeakyReLU
    torch.nn.Linear(in_features=32, out_features=16), # 隐藏层1 32维→隐藏层2 16维
    torch.nn.Tanh(),                                   # 继续非线性激活，强化拟合能力
    torch.nn.Linear(in_features=16, out_features=1)   # 输出层16维→1维（回归任务，输出层无激活）
)
print("多层全连接网络搭建完成，网络结构：")
print(model)
print("---" * 10)

### 训练损失函数，优化器（复用原代码逻辑，仅微调优化器参数传入方式）
# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)，回归任务通用，无需修改
#pytorch内置类定义，想换损失只需要修改loss_fn的定义，无需修动训练逻辑
#内置损失函数做了数值稳定性优化（比如避免除零、梯度爆炸），比手写的更健壮
loss_fn = torch.nn.MSELoss() # 回归任务

# 优化器传入模型所有可训练参数（model.parameters()自动收集网络所有层的权重/偏置）
# PyTorch 会自动根据这些参数的梯度来更新它们。
# 优化器：记录对哪些参数进行更新？如何更新（会记录梯度，参数，更新参数）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 微调学习率，适配多层网络

### 训练过程：正向传播、计算损失 损失计算参数梯度，更新参数（几乎完全复用原代码）
# 4. 训练模型(实现多层网络的梯度下降训练)
num_epochs = 5000 #整体训练轮数：非线性拟合需要更多轮数收敛，从1000调至5000
loss_history = [] # 记录损失变化，用于后续绘制损失曲线
for epoch in range(num_epochs):
    # 前向传播：输入X通过多层网络得到预测值y_pred（替换原a*X+b）
    # 正向传播 输入 -》 模型 -》输出
    y_pred = model(X)

    # 计算损失
    # 模型输出VS真实差异（mse损失函数）
    loss = loss_fn(y_pred, y)
    loss_history.append(loss.item()) # 记录每轮损失

    # 反向传播和优化
    #🌂：optimizer初始化已经绑定了所有需要更新的参数，调用zero_grad会自动遍历绑定的所有参数，统一清空梯度，无需手动判断，逐个处理
    #无论模型有多少参数，这行代码都能搞定
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加 #梯度累积：默认不会清空历史梯度
    loss.backward()        # 计算梯度 # loss对网络所有参数的梯度
    #🐟 优化器内部已经实现了参数更新的所有逻辑（包括梯度的使用、学习率的乘法、关闭梯度追踪），无需手动写with torch.no_grad()和更新公式
    #初始化优化器时选择不同的算法（SGD/Adam/Adagrad），optimizer.step()会自动执行对应算法的更新逻辑，训练代码完全不用改
    optimizer.step()       # 更新参数

    # 每500个 epoch 打印一次损失，避免输出过多（原1轮打印一次，修改后更简洁）
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 模型预测（替换原参数计算方式，用训练好的网络直接预测）
# with torch.no_grad()：关闭梯度计算，节省算力，仅用于推理/预测
with torch.no_grad():
    y_predicted = model(X).numpy() # 网络预测结果转numpy，方便可视化

print("\n训练完成！多层网络拟合sin函数结束")
print("---" * 10)

# 6. 绘制结果（优化可视化，对比原始数据、模型拟合、纯sin曲线）
plt.figure(figsize=(12, 5)) # 1行2列子图，同时展示拟合效果+损失曲线

# 子图1：原始sin数据 + 模型拟合曲线 + 纯sin基准曲线
plt.subplot(1, 2, 1)
plt.scatter(X_numpy, y_numpy, label='Raw Data (sin(x)+noise)', color='blue', alpha=0.6, s=10)
plt.plot(X_numpy, y_predicted, label='Model Fitting', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='Pure sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('X (0 ~ 2π)')
plt.ylabel('y = sin(x)')
plt.title('Sin Function Fitting by Multi-Layer Network')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：训练损失变化曲线，查看模型收敛情况
plt.subplot(1, 2, 2)
plt.plot(loss_history, color='orange', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.ylim(0, 0.2) # 限定y轴范围，更清晰看损失下降趋势

plt.tight_layout() # 自动调整子图间距，避免标签重叠
plt.show()

# 可选：单独绘制纯sin曲线和模型预测曲线，更清晰看拟合细节
plt.figure(figsize=(10, 6))
plt.plot(X_numpy, np.sin(X_numpy), label='Pure sin(x)', color='green', linewidth=2)
plt.plot(X_numpy, y_predicted, label='Multi-Layer Network', color='red', linewidth=2, linestyle='-.')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Model Prediction vs Pure Sin(x)')
plt.legend()
plt.grid(True)
plt.show()

