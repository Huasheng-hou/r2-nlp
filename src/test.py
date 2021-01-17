import pandas as pd
from tqdm import tqdm

f = pd.read_csv('../data/SICK/SICK.txt', sep='\t', quoting=3)

for index, row in tqdm(f.iterrows()):
    # print(row['#1 String'])
    print(row['entailment_label'])
