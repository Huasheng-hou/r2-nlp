import torch
from pytorch_pretrained_bert import BertAdam

from model import Model
from preprocess import process_data
from utils import train, test

train_loader, test_loader = process_data()

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
