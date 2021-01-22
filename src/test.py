import pandas as pd
from tqdm import tqdm
import torch

# f = pd.read_csv('../data/SICK/SICK.txt', sep='\t', quoting=3)

# for index, row in tqdm(f.iterrows()):
    # print(row['#1 String'])
    # print(row['entailment_label'])

a = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=False)

a.data.fill_(0.1)

for i in range(1, 5):
    print(i)
