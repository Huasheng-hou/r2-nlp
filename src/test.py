import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import torch
from datasets import load_dataset

train_loader, test_loader = load_dataset('20NG')

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
