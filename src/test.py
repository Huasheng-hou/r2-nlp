import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import torch

f = pd.read_csv('../data/SNLI/snli_1.0_train.txt', sep='\t')

datas = f.drop(labels=['sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse',
                       'sentence2_parse', 'captionID', 'pairID', 'label1', 'label2', 'label3',
                       'label4', 'label5'], axis=1, inplace=False)

datas.to_csv('../data/SNLI/dev.csv')



# for index, row in tqdm(f.iterrows()):
#     print(row)
    # print(row['#1 String'])
    # print(row['entailment_label'])

# a = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=False)
#
# a.data.fill_(0.1)
#
# for i in range(1, 5):
#     print(i)
