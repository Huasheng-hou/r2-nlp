import pandas as pd
from tqdm import tqdm

f = pd.read_csv('../data/MSRP/msr_paraphrase_test.txt', sep='\t', quoting=3)

for index, row in tqdm(f.iterrows()):
    print(row['#1 String'])
