import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from network import LeNet
from torchvision import datasets, transforms
from torch.autograd import Variable

lr = 0.01  # 学习率
momentum = 0.5
log_interval = 10  # 跑多少次batch进行一次日志记录
epochs = 10
batch_size = 64
test_batch_size = 1000


def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        loss = F.cross_entropy(output, target)  # 交叉熵损失函数
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数
        if batch_idx % log_interval == 0:  # 准备打印相关信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU
    data_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/mnist'
    model_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/pytorch-mnist/model.pth"
    train_loader = torch.utils.data.DataLoader(  # 加载训练数据
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=test_batch_size, shuffle=True)

    model = LeNet()  # 实例化一个网络对象
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 初始化优化器

    for epoch in range(1, epochs + 1):  # 以epoch为单位进行循环
        train(epoch)
        test()

    torch.save(model, model_path)  # 保存模型