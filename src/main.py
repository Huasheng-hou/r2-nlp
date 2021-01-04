import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from model import Model
from utils import train, test

path = "../data/"
bert_path = "../bert/"
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []  # 标签
pad_size = 32  # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)

with open(path + "train.txt", encoding='utf-8') as f:
    for i, l in tqdm(enumerate(f)):
        x1, y = l.strip().split('\t')
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # 得到input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)
        # 短则补齐，长则切断
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        #         print(len(ids), len(masks), len(types))
        assert len(ids) == len(masks) == len(types) == pad_size
        label.append([int(y)])

# 随机打乱索引
random_order = list(range(len(input_ids)))
np.random.seed(2020)   # 固定种子
np.random.shuffle(random_order)
print(random_order[:10])

# 4:1 划分训练集和测试集
input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids)*0.8)]])
input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids)*0.8)]])
input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids)*0.8)]])
y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids)*0.8):]])
input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids)*0.8):]])
input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids)*0.8):]])
y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

BATCH_SIZE = 16
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_types_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)
# print(model)

param_optimizer = list(model.named_parameters())  # 模型参数名字列表
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

NUM_EPOCHS = 3
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_loader) * NUM_EPOCHS
                     )

best_acc = 0.0
PATH = 'roberta_model.pth'  # 定义模型保存路径
for epoch in range(1, NUM_EPOCHS+1):  # 3个epoch
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)  # 保存最优模型
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
