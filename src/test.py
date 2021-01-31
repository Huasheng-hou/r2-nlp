import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import torch

newsgroup_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')

train = newsgroup_train.data
test = newsgroup_test.data

train_labels = newsgroup_train.target
test_labels = newsgroup_test.target

f = np.concatenate([train, test])

input_ids = []  # input char ids
input_segs = []  # segment ids
input_masks = []  # attention mask
label = []  # 标签

for row in tqdm(f):
    print(row)
# f = pd.read_csv('../data/AGNews/test.csv', sep='\t', quoting=3)
f = pd.read_csv('../data/AGNews/train.csv')

for index, row in tqdm(f.iterrows()):
    print(row)
    # print(row['#1 String'])
    # print(row['entailment_label'])

# a = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=False)
#
# a.data.fill_(0.1)
#
# for i in range(1, 5):
#     print(i)
