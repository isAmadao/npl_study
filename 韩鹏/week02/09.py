import os

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import warnings

# 设置中文字体（如果系统中有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("使用默认字体")

warnings.filterwarnings('ignore')

print("=" * 60)
print("开始深度学习文本分类实验 - 模型结构对Loss的影响")
print("=" * 60)

# 数据加载和预处理
try:
    dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
    print("数据集加载成功，样本数量:", len(dataset))
except FileNotFoundError:
    print("错误: 找不到数据集文件")
    print("请确保路径 '../Week01/dataset.csv' 正确")
    # 创建模拟数据用于演示
    print("创建模拟数据用于演示...")
    texts = [
        "帮我导航到北京", "导航到上海", "我要去广州",
        "查询明天天气", "北京天气怎么样", "今天会下雨吗",
        "播放音乐", "我想听周杰伦的歌", "播放流行音乐"
    ]
    labels = ["导航", "导航", "导航", "天气", "天气", "天气", "音乐", "音乐", "音乐"]
    dataset = pd.DataFrame({"text": texts, "label": labels})
else:
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

print(f"标签数量: {len(label_to_index)}")
print(f"标签映射: {label_to_index}")

# 构建字符词典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40

print(f"词汇表大小: {vocab_size}")
print(f"最大序列长度: {max_len}")


# 自定义数据集
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


# 灵活的模型类
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(FlexibleClassifier, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加批标准化
            layers.append(nn.ReLU()) # 添加激活函数
            layers.append(nn.Dropout(dropout_rate)) # 随机删除一部分神经元，防止过拟合
            prev_dim = hidden_dim
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # 解包layers,并使用Sequential容器包装
        self.network = nn.Sequential(*layers)

    # 模型计算入口，定义了向前计算过程
    def forward(self, x):
        return self.network(x)


# 创建数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 分割训练集和验证集
train_size = int(0.8 * len(char_dataset))
val_size = len(char_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(char_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

output_dim = len(label_to_index)
print(f"输出维度: {output_dim}")

# 定义不同的模型结构配置
model_configs = [
    {
        "name": "1层128节点",
        "hidden_dims": [128],
        "color": "blue",
        "lr": 0.01
    },
    {
        "name": "2层128节点",
        "hidden_dims": [128, 128],
        "color": "green",
        "lr": 0.01
    },
    {
        "name": "3层128节点",
        "hidden_dims": [128, 128, 128],
        "color": "red",
        "lr": 0.01
    },
    {
        "name": "3层递减(256-128-64)",
        "hidden_dims": [256, 128, 64],
        "color": "purple",
        "lr": 0.01
    },
    {
        "name": "3层递增(64-128-256)",
        "hidden_dims": [64, 128, 256],
        "color": "brown",
        "lr": 0.01
    },
    {
        "name": "5层递减(512-256-128-64-32)",
        "hidden_dims": [512, 256, 128, 64, 32],
        "color": "gray",
        "lr": 0.005  # 更复杂模型使用更小的学习率
    },
]


# 评估函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    return avg_loss, accuracy


# 训练和评估每个模型
def train_model(model_config, num_epochs=10):
    """训练单个模型并返回训练历史"""
    print(f"\n{'=' * 60}")
    print(f"训练模型: {model_config['name']}")
    print(f"隐藏层结构: {model_config['hidden_dims']}")
    print(f"层数: {len(model_config['hidden_dims'])}")

    # 创建模型
    model = FlexibleClassifier(vocab_size, model_config["hidden_dims"], output_dim, dropout_rate=0.3)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config.get("lr", 0.01))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 记录训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 验证阶段
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1:2d}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {
        "name": model_config["name"],
        "hidden_dims": model_config["hidden_dims"],
        "num_layers": len(model_config["hidden_dims"]),
        "total_params": total_params,
        "color": model_config["color"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accuracies[-1],
        "model": model  # 保存模型
    }


# 训练所有模型
print("\n" + "=" * 60)
print("开始比较不同模型结构对Loss的影响")
print("=" * 60)

results = []
for config in model_configs:
    try:
        result = train_model(config, num_epochs=10)
        results.append(result)
    except Exception as e:
        print(f"训练模型 {config['name']} 时出错: {e}")
        continue


# 可视化分析
def visualize_results(results):
    """可视化不同模型的训练结果"""

    if not results:
        print("没有训练结果可可视化")
        return

    plt.figure(figsize=(15, 10))

    # 1. 绘制训练和验证Loss曲线
    plt.subplot(2, 2, 1)
    for result in results:
        epochs = range(1, len(result["train_losses"]) + 1)
        plt.plot(epochs, result["train_losses"],
                 label=f"{result['name']}-Train",
                 color=result["color"],
                 linewidth=2,
                 linestyle='-')
        plt.plot(epochs, result["val_losses"],
                 label=f"{result['name']}-Val",
                 color=result["color"],
                 linewidth=2,
                 linestyle='--')

    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.title('训练和验证Loss变化曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)

    # 2. 绘制最终验证Loss对比图
    plt.subplot(2, 2, 2)
    model_names = [r["name"] for r in results]
    final_val_losses = [r["final_val_loss"] for r in results]

    # 按最终验证loss排序
    sorted_indices = np.argsort(final_val_losses)
    model_names_sorted = [model_names[i] for i in sorted_indices]
    final_val_losses_sorted = [final_val_losses[i] for i in sorted_indices]

    colors = [results[i]["color"] for i in sorted_indices]
    bars = plt.bar(range(len(model_names_sorted)), final_val_losses_sorted, color=colors)

    plt.xlabel('模型结构', fontsize=12)
    plt.ylabel('最终验证Loss', fontsize=12)
    plt.title('不同模型的最终验证Loss对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names_sorted)), model_names_sorted, rotation=45, ha='right')

    # 添加数值标签
    for i, (bar, loss) in enumerate(zip(bars, final_val_losses_sorted)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(final_val_losses_sorted) * 0.01,
                 f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

    plt.grid(True, alpha=0.3, axis='y')

    # 3. 绘制验证准确率曲线
    plt.subplot(2, 2, 3)
    for result in results:
        epochs = range(1, len(result["val_accuracies"]) + 1)
        plt.plot(epochs, result["val_accuracies"],
                 label=result["name"],
                 color=result["color"],
                 linewidth=2,
                 marker='o')

    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('验证准确率', fontsize=12)
    plt.title('验证准确率变化曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 4. 绘制层数与性能的关系
    plt.subplot(2, 2, 4)

    # 按层数分组
    layer_groups = {}
    for result in results:
        layer_num = result["num_layers"]
        if layer_num not in layer_groups:
            layer_groups[layer_num] = {"losses": [], "accuracies": []}
        layer_groups[layer_num]["losses"].append(result["final_val_loss"])
        layer_groups[layer_num]["accuracies"].append(result["final_val_accuracy"])

    # 计算每层的平均Loss和准确率
    layer_nums = sorted(layer_groups.keys())
    avg_losses = [np.mean(layer_groups[l]["losses"]) for l in layer_nums]
    avg_accuracies = [np.mean(layer_groups[l]["accuracies"]) for l in layer_nums]

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('隐藏层层数', fontsize=12)
    ax1.set_ylabel('平均验证Loss', color=color1, fontsize=12)
    ax1.plot(layer_nums, avg_losses, color=color1, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('平均验证准确率', color=color2, fontsize=12)
    ax2.plot(layer_nums, avg_accuracies, color=color2, marker='s', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('层数对性能的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. 参数量与性能的关系
    plt.figure(figsize=(10, 6))

    param_counts = [r["total_params"] for r in results]
    final_val_losses = [r["final_val_loss"] for r in results]
    final_val_accuracies = [r["final_val_accuracy"] for r in results]

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('模型参数量', fontsize=12)
    ax1.set_ylabel('最终验证Loss', color=color1, fontsize=12)
    scatter1 = ax1.scatter(param_counts, final_val_losses,
                           c=[r["num_layers"] for r in results],
                           cmap='viridis', s=100, alpha=0.8)
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 添加模型名称标签
    for i, (params, loss, result) in enumerate(zip(param_counts, final_val_losses, results)):
        ax1.annotate(result["name"],
                     xy=(params, loss),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     alpha=0.7)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('最终验证准确率', color=color2, fontsize=12)
    ax2.scatter(param_counts, final_val_accuracies,
                c=[r["num_layers"] for r in results],
                cmap='viridis', s=100, alpha=0.8, marker='s')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('参数量与性能的关系', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('隐藏层层数', fontsize=12)

    plt.tight_layout()
    plt.show()


# 生成详细分析报告
def generate_analysis_report(results):
    """生成详细的分析报告"""
    if not results:
        print("没有训练结果可分析")
        return

    print("\n" + "=" * 80)
    print("模型性能详细分析报告")
    print("=" * 80)

    # 按最终验证Loss排序
    sorted_by_val_loss = sorted(results, key=lambda x: x["final_val_loss"])

    # 按最终验证准确率排序
    sorted_by_val_acc = sorted(results, key=lambda x: x["final_val_accuracy"], reverse=True)

    # Loss排名
    print(f"\n🏆 验证Loss排名 (从低到高):")
    print(f"{'排名':<5} {'模型名称':<20} {'层数':<6} {'验证Loss':<10} {'验证准确率':<12} {'参数量':<15}")
    print("-" * 80)

    for i, result in enumerate(sorted_by_val_loss, 1):
        print(f"{i:<5} {result['name']:<20} {result['num_layers']:<6} "
              f"{result['final_val_loss']:<10.4f} {result['final_val_accuracy']:<12.2%} "
              f"{result['total_params']:<15,}")

    # 准确率排名
    print(f"\n🎯 验证准确率排名 (从高到低):")
    print(f"{'排名':<5} {'模型名称':<20} {'层数':<6} {'验证准确率':<12} {'验证Loss':<10}")
    print("-" * 80)

    for i, result in enumerate(sorted_by_val_acc, 1):
        print(f"{i:<5} {result['name']:<20} {result['num_layers']:<6} "
              f"{result['final_val_accuracy']:<12.2%} {result['final_val_loss']:<10.4f}")

    # 最佳模型分析
    best_by_loss = sorted_by_val_loss[0]
    best_by_acc = sorted_by_val_acc[0]

    print(f"\n⭐ 最佳模型分析:")
    print(f"   最低验证Loss模型: {best_by_loss['name']}")
    print(f"     隐藏层结构: {best_by_loss['hidden_dims']}")
    print(f"     层数: {best_by_loss['num_layers']}")
    print(f"     最终验证Loss: {best_by_loss['final_val_loss']:.4f}")
    print(f"     验证准确率: {best_by_loss['final_val_accuracy']:.2%}")

    print(f"\n   最高验证准确率模型: {best_by_acc['name']}")
    print(f"     隐藏层结构: {best_by_acc['hidden_dims']}")
    print(f"     层数: {best_by_acc['num_layers']}")
    print(f"     最终验证Loss: {best_by_acc['final_val_loss']:.4f}")
    print(f"     验证准确率: {best_by_acc['final_val_accuracy']:.2%}")

    # 过拟合分析
    print(f"\n🔍 过拟合分析 (Train Loss vs Val Loss):")
    print(f"{'模型名称':<20} {'最终Train Loss':<15} {'最终Val Loss':<15} {'差值':<10} {'过拟合程度':<15}")
    print("-" * 80)

    for result in results:
        train_loss = result["final_train_loss"]
        val_loss = result["final_val_loss"]
        diff = val_loss - train_loss
        overfit_level = "轻微" if diff < 0.1 else ("中等" if diff < 0.3 else "严重")

        print(f"{result['name']:<20} {train_loss:<15.4f} {val_loss:<15.4f} {diff:<10.4f} {overfit_level:<15}")

    # 训练稳定性分析
    print(f"\n📊 训练稳定性分析 (Loss标准差):")
    print(f"{'模型名称':<20} {'Train Loss标准差':<15} {'Val Loss标准差':<15}")
    print("-" * 80)

    for result in results:
        train_std = np.std(result["train_losses"][3:]) if len(result["train_losses"]) > 3 else 0
        val_std = np.std(result["val_losses"][3:]) if len(result["val_losses"]) > 3 else 0

        print(f"{result['name']:<20} {train_std:<15.4f} {val_std:<15.4f}")

    print(f"\n💡 实验结论和建议:")
    print(f"   1. 最佳平衡模型: {sorted_by_val_loss[0]['name']} (验证Loss最低)")
    print(f"   2. 最准确模型: {sorted_by_val_acc[0]['name']} (验证准确率最高)")
    print(f"   3. 参数量与性能的关系: 更多参数并不总是意味着更好性能")
    print(f"   4. 层数建议: {sorted_by_val_loss[0]['num_layers']}-{sorted_by_val_acc[0]['num_layers']} 层之间")
    print(f"   5. 注意过拟合: 查看过拟合分析，考虑添加更多正则化")


# 测试函数
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


# 主程序
if __name__ == "__main__":
    # 运行分析和可视化
    if results:
        visualize_results(results)
        generate_analysis_report(results)

        # 使用最佳模型进行预测
        index_to_label = {i: label for label, i in label_to_index.items()}

        # 选择最佳模型（按验证Loss）
        best_result = min(results, key=lambda x: x["final_val_loss"])

        print(f"\n🚀 使用最佳模型 '{best_result['name']}' 进行预测:")
        new_text = "帮我导航到北京"
        predicted_class = classify_text(new_text, best_result["model"], char_to_index, vocab_size, max_len,
                                        index_to_label)
        print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

        new_text_2 = "查询明天北京的天气"
        predicted_class_2 = classify_text(new_text_2, best_result["model"], char_to_index, vocab_size, max_len,
                                          index_to_label)
        print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

        # 测试一些边缘案例
        test_cases = [
            "导航去上海",
            "今天天气如何",
            "播放一首歌",
            "明天会下雨吗",
            "我要去南京"
        ]

        print(f"\n📋 额外测试案例:")
        for test_text in test_cases:
            predicted = classify_text(test_text, best_result["model"], char_to_index, vocab_size, max_len,
                                      index_to_label)
            print(f"输入 '{test_text}' 预测为: '{predicted}'")

        print(f"\n🎉 分析完成！已成功比较{len(results)}种不同模型结构。")
    else:
        print("没有成功训练任何模型，请检查错误信息。")