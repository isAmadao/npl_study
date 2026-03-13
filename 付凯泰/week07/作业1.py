阅读 02-joint-bert-training-only 代码，并回答以下问题：
    ◦ bert 文本分类和 实体识别有什么关系，分别使用什么loss？


      文本分类是句子级任务（intent）
      实体识别是token级任务（NER/slot）
     共享 BERT encoder，都属于 Joint NLU 模型

      文本分类loss :	CrossEntropyLoss   seq_loss
      实体识别loss:	CrossEntropyLoss   token_loss 


    ◦ 多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？

直接相加会导致任务不平衡，token_loss主导训练，文本分类基本不学，因为token_loss >> seq_loss 
处理方法：
1.loss加权  loss = α seq_loss + β token_loss
2.采用动态权重
3.使用梯度平衡
4.分成两阶段训练
