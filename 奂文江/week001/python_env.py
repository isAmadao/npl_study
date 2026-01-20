import jieba
import sklearn
import numpy as np
import torch
import pandas as pd
import fastapi as ft
text = '这是一个中文分词器'
jl = " ".join(jieba.lcut(text))
print(jl)
print('jieba版本号：', jieba.__version__)
print('sklearn版本号：', sklearn.__version__)
print('pytorch版本号：', torch.__version__)
print('numpy版本号：', np.__version__)
print('pandas版本号：', pd.__version__)
print('fastapi版本号：', ft.__version__)
