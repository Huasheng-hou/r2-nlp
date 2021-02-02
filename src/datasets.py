from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.datasets import fetch_20newsgroups


def embeddings(tokenizer, s1, s2, pad_size=48):
    # 得到input_id, seg_id, att_mask
    x1 = tokenizer.tokenize(s1)
    x2 = tokenizer.tokenize(s2)
    len_1, len_2 = len(x1), len(x2)

    tokens = ["[CLS]"] + x1 + ["[SEP]"] + x2 + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    segs = [0] * (len_1 + 2) + [1] * (len_2 + 1)
    # segs = [0] * len(ids)
    masks = [1] * len(ids)
    # 短则补齐，长则切断
    if len(ids) < pad_size:
        segs = segs + [0] * (pad_size - len(ids))  # mask部分 segment置为1
        masks = masks + [0] * (pad_size - len(ids))
        ids = ids + [0] * (pad_size - len(ids))
    else:
        segs = segs[:pad_size]
        masks = masks[:pad_size]
        ids = ids[:pad_size]
    return ids, segs, masks


def MSRP(pad_size=128):
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

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, s1, s2, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        label.append([int(row['Quality'])])

    train_len = len(train)

    return [input_ids[:train_len], input_segs[:train_len], input_masks[:train_len], label[:train_len]], \
           [input_ids[train_len:], input_segs[train_len:], input_masks[train_len:], label[train_len:]]


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

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, q1, q2, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        label.append([int(row['is_duplicate'])])

    return [input_ids, input_segs, input_masks, label], []


def SICK(pad_size=48):
    f = pd.read_csv('../data/SICK/SICK.txt', sep='\t', quoting=3)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    for index, row in tqdm(f.iterrows()):
        s1, s2 = row['sentence_A'], row['sentence_B']
        if not isinstance(s1, str):
            continue
        if not isinstance(s2, str):
            continue

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, s1, s2, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        entail = row['entailment_label']
        if entail == 'NEUTRAL':
            label.append(0)
        elif entail == 'ENTAILMENT':
            label.append(1)
        else:
            label.append(2)

    return [input_ids, input_segs, input_masks, label], []


def AGNews(pad_size=60):

    if pad_size == 0:
        pad_size = 60

    train = pd.read_csv('../data/AGNews/train.csv')
    test = pd.read_csv('../data/AGNews/test.csv')

    f = pd.concat([train, test])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    for index, row in tqdm(f.iterrows()):
        s1, s2 = row['title'], row['article']
        if not isinstance(s1, str):
            continue
        if not isinstance(s2, str):
            continue

        # 得到input_id, seg_id, att_mask
        x = tokenizer.tokenize(s1+s2)

        tokens = ["[CLS]"] + x + ["[SEP]"]
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

        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        label.append([int(row['class']) - 1])

    train_len = len(train)

    return [input_ids[:train_len], input_segs[:train_len], input_masks[:train_len], label[:train_len]], \
           [input_ids[train_len:], input_segs[train_len:], input_masks[train_len:], label[train_len:]]


def NG(pad_size=300):

    if pad_size <= 0:
        pad_size = 300

    newsgroup_train = fetch_20newsgroups(subset='train')
    newsgroup_test = fetch_20newsgroups(subset='test')

    train = newsgroup_train.data
    test = newsgroup_test.data

    y_train = newsgroup_train.target.astype(int)
    y_test = newsgroup_test.target.astype(int)

    f = np.concatenate([train, test])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask

    for s in tqdm(f):
        if not isinstance(s, str):
            continue

        # 数据集文本太长，在这里截断
        if len(s) > 512:
            s = s[:512]

        # 得到input_id, seg_id, att_mask
        x = tokenizer.tokenize(s)

        tokens = ["[CLS]"] + x + ["[SEP]"]
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

        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size

    train_len = len(train)

    return [input_ids[:train_len], input_segs[:train_len], input_masks[:train_len], y_train], \
           [input_ids[train_len:], input_segs[train_len:], input_masks[train_len:], y_test]


def SNLI(pad_size=48):
    train = pd.read_csv('../data/SNLI/snli_1.0_train.txt', sep='\t')
    test = pd.read_csv('../data/SNLI/snli_1.0_test.txt', sep='\t')

    f = pd.concat([train, test])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []  # input char ids
    input_segs = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    for index, row in tqdm(f.iterrows()):
        s1, s2 = row['sentence1'], row['sentence2']
        if not isinstance(s1, str):
            continue
        if not isinstance(s2, str):
            continue

        if row['gold_label'] == '-':
            continue

        # 得到input_id, seg_id, att_mask
        ids, segs, masks = embeddings(tokenizer, s1, s2, pad_size=pad_size)
        input_ids.append(ids)
        input_segs.append(segs)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(segs) == pad_size
        entail = row['gold_label']
        if entail == 'neutral':
            label.append(0)
        elif entail == 'entailment':
            label.append(1)
        else:
            label.append(2)

    train_len = len(train)

    return [input_ids[:train_len], input_segs[:train_len], input_masks[:train_len], label[:train_len]], \
           [input_ids[train_len:], input_segs[train_len:], input_masks[train_len:], label[train_len:]]


def load_dataset(name, pad_size=0, sample_size=0, random=False, BATCH_SIZE=16, split_rate=0.8):
    train, test = None, None

    if name == 'AGNews':
        train, test = AGNews(pad_size=pad_size)
    elif name == '20NG':
        train, test = NG(pad_size=pad_size)
    elif name == 'DBPedia':
        train, test = AGNews(pad_size=pad_size)
    elif name == 'MSRP':
        train, test = MSRP(pad_size=pad_size)
    elif name == 'Quora':
        train, test = Quora(pad_size=pad_size)
    elif name == 'SICK':
        train, test = SICK()
    elif name == 'SNLI':
        train, test = SNLI()
    elif name == 'THUCNews':
        train, test = AGNews()

    ids, segs, masks, labels = train[0], train[1], train[2], train[3]

    ''' Sample data from Original Dataset '''
    if sample_size > 0:
        if len(ids) < sample_size:
            print("Sample size exceeds dataset capacity!")
            return
        if not random:
            ids, segs, masks, labels = ids[:sample_size], segs[:sample_size], masks[:sample_size], labels[:sample_size]
        else:
            ids = np.random.choice(ids, sample_size, replace=False)
            segs = np.random.choice(segs, sample_size, replace=False)
            masks = np.random.choice(masks, sample_size, replace=False)
            labels = np.random.choice(labels, sample_size, replace=False)

        test = []

    ''' Split The Dataset or use the original Split Set '''
    if len(test) <= 0:

        if split_rate >= 1:
            return

        # 随机打乱索引
        random_order = list(range(len(ids)))
        np.random.seed(2020)  # 固定种子
        np.random.shuffle(random_order)
        print(random_order[:10])

        # 4:1 划分训练集和测试集
        input_ids_train = np.array([ids[i] for i in random_order[:int(len(ids) * split_rate)]])
        input_types_train = np.array([segs[i] for i in random_order[:int(len(ids) * split_rate)]])
        input_masks_train = np.array([masks[i] for i in random_order[:int(len(ids) * split_rate)]])
        y_train = np.array([labels[i] for i in random_order[:int(len(ids) * split_rate)]])

        input_ids_test = np.array([ids[i] for i in random_order[int(len(ids) * split_rate):]])
        input_types_test = np.array([segs[i] for i in random_order[int(len(ids) * split_rate):]])
        input_masks_test = np.array([masks[i] for i in random_order[int(len(ids) * split_rate):]])
        y_test = np.array([labels[i] for i in random_order[int(len(ids) * split_rate):]])
    else:
        input_ids_train, input_types_train, input_masks_train, y_train = np.array(ids), np.array(segs), \
                                                                         np.array(masks), np.array(labels)
        input_ids_test, input_types_test, input_masks_test, y_test = np.asarray(test[0]), np.asarray(test[1]), \
                                                                     np.asarray(test[2]), np.asarray(test[3])

    ''' Print Input Tensor Sizes '''
    print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)
    print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

    ''' Put the Raw Data into Bert Dataloader '''
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
