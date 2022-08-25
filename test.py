import os
import pandas as pd
import torch
from utils import get_df
import torch.nn as nn
path = 'nlp_data'
# df_train, df_test, df_pcl, df_cat = get_df(path)
# print(df_test.groupby('keyword').label.sum())
# print(df_train.groupby('keyword').label.sum())
# ext_col_names = ['paragraph_id', 'article_id', 'keyword', 'country_code', 'paragraph', 'label']
# df_train = pd.read_csv(os.path.join(path, 'df_test_expansion.csv'), names=ext_col_names, index_col='paragraph_id')
# # print(df_train.label.value_counts())
# print(df_train)
# df = pd.read_csv('all_hard_examples.csv')
# print(df.labels.value_counts())
# print(df.input_ids.apply(len))

# result_path = 'result'
# res_1 = []
# res_2 = []
# with open(os.path.join(result_path, 'result_1.txt')) as f:
#     lines = f.readlines()
#     for line in lines:
#         res_1.append(line.strip('\n'))
# with open(os.path.join(result_path, 'task1.txt')) as f:
#     lines = f.readlines()
#     for line in lines:
#         res_2.append(line.strip('\n'))

# diff = 0
# same = 0
# for i in range(len(res_1)):
#     if res_1[i] == res_2[i]:
#         same+=1
#     else:
#         diff+=1
# print(same)
# print(diff)
loss = nn.MSELoss()
input = torch.randn(2, requires_grad=True)
target = torch.FloatTensor([2,3])
output = loss(input, target)
output.backward()
print(type(target))


# def f1_loss(y_pred, y_true, is_training=False):
#     """
#     Calculate F1-score
#     """
#     assert y_true.ndim == 1
#     assert y_pred.ndim == 1 or y_pred.ndim == 2
#     if y_pred.ndim == 2:
#         y_pred = y_pred.argmax(dim=1)
#     tp = (y_true * y_pred).sum().to(torch.float32)
#     tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
#     fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
#     fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
#     epsilon = 1e-7
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)
    
#     f1 = 2* (precision*recall) / (precision + recall + epsilon)
#     f1.requires_grad = is_training
#     return f1
# y_pred = torch.FloatTensor([1.,1.,0.])
# y_true = torch.FloatTensor([1.,1.,1.])
# print(f1_loss(y_pred, y_true))
# def idx2onehot(idx, n):

#     assert torch.max(idx).item() < n

#     if idx.dim() == 1:
#         idx = idx.unsqueeze(1)
#     onehot = torch.zeros(idx.size(0), n).to(idx.device)
#     onehot.scatter_(1, idx, 1)
    
#     return onehot
# c = torch.LongTensor([1,2,3,2,1])
# print(idx2onehot(c,4))

from modulefinder import ModuleFinder
f = ModuleFinder()
# Run the main script
f.run_script('main.py')
# Get names of all the imported modules
names = list(f.modules.keys())
# Get a sorted list of the root modules imported
basemods = sorted(set([name.split('.')[0] for name in names]))
# Print it nicely
print("\n".join(basemods))