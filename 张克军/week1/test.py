
import jieba #用于分词的库
import sklearn #用于机器学习的库
import matplotlib #用于画图的库
import pandas as pd #用于数据处理和分析的库
import numpy as np #用于科学计算的库
import torch #用于深度学习的库

print("jieba:",jieba.__version__)
print("sklearn:",sklearn.__version__)
print("matplotlib:",matplotlib.__version__)
print("pandas:",pd.__version__)
print("numpy:",np.__version__)
print("torch:",torch.__version__)
print("torch.cuda.is_available():",torch.cuda.is_available())
