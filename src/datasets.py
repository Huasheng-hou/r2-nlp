from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch


''' 1. TOKENIZE '''


def process_data(input_ids, input_segs, input_masks, label):

    # 随机打乱索引
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)  # 固定种子
    np.random.shuffle(random_order)
    print(random_order[:10])

    # 4:1 划分训练集和测试集
    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_types_train = np.array([input_segs[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

    input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_types_test = np.array([input_segs[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
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

    return train_loader, test_loader


def embeddings(tokenizer, tokens, pad_size=48):
    # 得到input_id, seg_id, att_mask
    ids = tokenizer.convert_tokens_to_ids(tokens)
    segs = [0] * len(ids)
    masks = [1] * len(ids)
    # 短则补齐，长则切断
    if len(ids) < pad_size:
        segs = segs + [1] * (pad_size - len(ids))  # mask部分 segment置为1
        masks = masks + [0] * (pad_size - len(ids))
        ids = ids + [0] * (pad_size - len(ids))
    else:
        segs = segs[:pad_size]
        masks = masks[:pad_size]
        ids = ids[:pad_size]
    return ids, segs, masks


def MSRP(pad_size=72):

    train = pd.read_csv('../data/MSRP/msr_paraphrase_train.txt', sep='\t', quoting=3)
    test = pd.read_csv('../data/MSRP/msr_paraphrase_test.txt', sep='\t', quoting=3)

    f = pd.concat([train, test])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    for index, row in tqdm(f.iterrows()):
        s1, s2 = row['#1 String'], row['#2 String']
        if not isinstance(s1, str):
            continue
        if not isinstance(s2, str):
            continue
        x1 = tokenizer.tokenize(s1)
        x2 = tokenizer.tokenize(s2)
        tokens = ["[CLS]"] + x1 + ["[SEP]"] + x2 + ["[SEP]"]
        # tokens = ["[CLS]"] + x1 + x2 + ["[SEP]"]

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, tokens, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        label.append([int(row['Quality'])])

    train_len = len(train)

    input_ids_train = np.array(input_ids[:train_len])
    input_types_train = np.array(input_segs[:train_len])
    input_masks_train = np.array(input_masks[:train_len])
    y_train = np.array(label[:train_len])
    print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

    input_ids_test = np.array(input_ids[train_len:])
    input_types_test = np.array(input_segs[train_len:])
    input_masks_test = np.array(input_masks[train_len:])
    y_test = np.array(label[train_len:])
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

    return train_loader, test_loader


def Quora(pad_size=48):

    f = pd.read_csv('../data/Quora/quora_duplicate_questions.tsv', sep='\t')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    for index, row in tqdm(f.iterrows()):
        q1, q2 = row['question1'], row['question2']
        if not isinstance(q1, str):
            continue
        if not isinstance(q2, str):
            continue
        x1 = tokenizer.tokenize(q1)
        x2 = tokenizer.tokenize(q2)
        tokens = ["[CLS]"] + x1 + ["[SEP]"] + x2 + ["[SEP]"]

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, tokens, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        label.append([int(row['is_duplicate'])])

    return process_data(input_ids, input_segs, input_masks, label)
