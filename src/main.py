import torch
from pytorch_pretrained_bert import BertAdam, BertTokenizer

from model import Model, R2Net, LEM
from datasets import Quora, MSRP, SICK, AGNews
from utils import train, test, test_lem
from visualize import TSNE


def run(train_loader, test_loader, model):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    model.to(DEVICE)
    # print(model)

    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    NUM_EPOCHS = 1
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.05,
                         t_total=len(train_loader) * NUM_EPOCHS
                         )

    best_acc = 0.0
    PATH = 'sick_model.pth'  # 定义模型保存路径
    for epoch in range(1, NUM_EPOCHS + 1):  # 3个epoch
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc = test(model, DEVICE, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)  # 保存最优模型
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))


train_data, test_data = AGNews()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

times = 5

base, R2Net, LEM = Model(4), R2Net(4), LEM(4)

run(train_data, test_data, LEM)

_, V, C, Y = test_lem(LEM, DEVICE, test_data)

# 特征向量可视化
TSNE(torch.cat(V), C, torch.cat(Y), cls=4)

# for idx in range(times):
#     print("WITHOUT LABEL EMBEDDING:\n\n")
#     run(train_data, test_data, base)
#     print("WITH LABEL EMBEDDING:\n")
#     run(train_data, test_data, LEM)
