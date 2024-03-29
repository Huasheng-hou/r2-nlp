import torch
from pytorch_pretrained_bert import BertAdam, BertTokenizer

from model import Model, R2Net, LEM
from datasets import Quora, MSRP, SICK, AGNews, NG, SNLI
from utils import train, test, test_lem
from visualize import TSNE


def run(train_loader, test_loader, model):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    model.to(DEVICE)
    # print(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    NUM_EPOCHS = 6
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.05,
                         t_total=len(train_loader) * NUM_EPOCHS
                         )

    best_acc = 0.0
    PATH = 'sick_model.pth'
    for epoch in range(1, NUM_EPOCHS + 1):  # 3epoch
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc = test(model, DEVICE, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))


train_data, test_data = SNLI()

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(1)

times, cls = 5, 3

# base, R2Net, LEM = Model(cls), R2Net(cls), LEM(cls)

base = Model(cls)
# LEM = LEM(cls)

run(train_data, test_data, base)

# _, V, C, Y = test_lem(LEM, DEVICE, test_data)

# TSNE(torch.cat(V), C, torch.cat(Y), cls=4)

# for idx in range(times):
#     print("WITHOUT LABEL EMBEDDING:\n\n")
#     run(train_data, test_data, base)
#     print("WITH LABEL EMBEDDING:\n")
#     run(train_data, test_data, LEM)
