import torch
from pytorch_pretrained_bert import BertAdam, BertTokenizer

from model import Model, R2Net
from datasets import Quora, MSRP, SICK
from utils import train, test


def run(train_loader, test_loader):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    model = R2Net(3).to(DEVICE)

    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    NUM_EPOCHS = 100
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


train_data, test_data = SICK()

times = 20

for idx in range(times):
    run(train_data, test_data)
