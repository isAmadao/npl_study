1. BERT 文本分类和实体识别的关系及 Loss
   - 关系：两者是多任务学习（Multi-task Learning）关系。共享同一个 BERT 编码器提取特征，上层分别接两个任务头：
     - 文本分类：取 [CLS] 位或全局池化特征，做句子级分类。
     - 实体识别：取每个 Token 的输出特征，做Token 级序列标注。
   - Loss：两者均使用 交叉熵损失 (nn.CrossEntropyLoss)。
     - seq_loss：计算句子意图标签的交叉熵。
     - token_loss：计算有效 Token（mask=1）实体标签的交叉熵。

2. `loss = seq_loss + token_loss` 的坏处及处理方法
   - 坏处：通常 Token 数量远大于句子数量，导致 token_loss 数值远大于 seq_loss，梯度更新主要由实体识别任务主导，意图识别任务可能训练不充分。 
   - 处理方法：对Loss 值进行归一化，使两者处于同一数量级。