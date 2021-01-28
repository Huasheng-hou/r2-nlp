import torch
import time
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch):   # 训练模型
    model.train()
    best_acc = 0.0
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1, x2, x3])  # 得到预测结果
        model.zero_grad()             # 梯度清零
        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 100 == 0:    # 打印loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(x1),
                                                                           len(train_loader.dataset),
                                                                           100. * (batch_idx+1) / len(train_loader),
                                                                           loss.item()))  # 记得为loss.item()


def test(model, device, test_loader):    # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0
    acc = 0
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, acc, len(test_loader.dataset),
          100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)
