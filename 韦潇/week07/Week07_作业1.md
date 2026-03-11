
# 阅读 02-joint-bert-training-only 代码，并回答以下问题：

## 1.bert 文本分类和 实体识别有什么关系，分别使用什么loss？

```
关系：
- 共享底层BERT编码器 ：两个任务共用同一个BERT提取语义特征
- 输出头独立 ：各自有独立的分类头
- 互补增强 ：意图识别提供全局语义，槽位填充提供细粒度实体信息
```

```
使用的Loss (main.py:45-47)：

# 两个任务都使用 CrossEntropyLoss
self.criterion = nn.CrossEntropyLoss()

# 意图识别loss - 对整个句子的分类
seq_loss = self.criterion(seq_output, seq_label_ids)

# 槽位填充loss - 对每个有效token的分类（忽略padding）
active_loss = attention_mask.view(-1) == 1
active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
active_labels = token_label_ids.view(-1)[active_loss]
token_loss = self.criterion(active_logits, active_labels)

# 直接相加
loss = seq_loss + token_loss
```

## 2.多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？

```
坏处：
① Loss量级不平衡
- token_loss 涉及多个token，量级比 seq_loss 大
- 导致模型 偏向槽位填充任务 ，意图识别学习不充分
② 任务难度不一致
意图识别：简单的多分类任务
槽位填充：序列标注任务，更复杂
收敛速度不同，可能一个任务过拟合，另一个欠拟合
③ 梯度竞争
共享的BERT参数同时接收两个任务的梯度
梯度方向可能冲突，导致优化困难
```

```
解决方案：

方案1：加权损失
loss = α * seq_loss + β * token_loss
# α, β 通过超参数搜索确定，如 α=1.0, β=0.5

方案2：不确定性加权（自适应学习权重）
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))  # 学习两个任务的权重
    
    def forward(self, seq_loss, token_loss):
        precision_seq = torch.exp(-self.log_vars[0])
        precision_token = torch.exp(-self.log_vars[1])
        
        loss = precision_seq * seq_loss + self.log_vars[0] + \
               precision_token * token_loss + self.log_vars[1]
        return loss
        
方案3：梯度归一化（GradNorm）
让不同任务的梯度量级接近，动态调整权重。
```
