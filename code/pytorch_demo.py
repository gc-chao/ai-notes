# ========================================================================================
# pytorch的应用示例
# ========================================================================================
# 这里主要给出了使用PyTorch进行自动微分和神经网络训练的基本结构。
# 除此之外，还附加了两种典型的网络结构用法，分别是CNN和RNN。
# ========================================================================================
import torch
# ========================================================================================
# 计算图
# ========================================================================================
# 按照list建立一个张量(requires_grad=True表示需要存储梯度，否则无法反向传播)
X1 = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
], dtype=torch.float32, requires_grad=True)
print("X1:\n" + str(X1))
# 随机生成一个特定形状的张量
X2 = torch.rand((5, 4), dtype=torch.float32, requires_grad=True)
print("X2:\n" + str(X2))
# @表示矩阵乘法X3 = (X1 - 5) * X2^T
X3 = (X1 - 5) @ X2.T
print("X3=(X1-5) * X2^T:\n" + str(X3))
# 要求保留X3的梯度（否则PyTorch默认会将非叶子节点的梯度清除以减小开销）
X3.retain_grad()
# 计算求和
Y = torch.sum(X3)
print("Y:\n" + str(Y))
# 反向传播
Y.backward()
# 输出各个变量的梯度
print("grad(X3):\n" + str(X3.grad))
print("grad(X2):\n" + str(X2.grad))
print("grad(X1):\n" + str(X1.grad))
# ========================================================================================
# 以全连接神经网络为示例演示完整的训练过程
# ========================================================================================
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 继承nn.Module可以定义自己的模型
class FCNN(nn.Module):
    # 通常在初始化函数中定义网络的各层结构
    def __init__(self):
        # 包括nn.Linear和nn.ReLU在内的大部分模块都继承自nn.Module
        # 继承了nn.Module的类的对象，都可以被parameters()反射枚举到其内部参数
        super(FCNN, self).__init__()
        # 添加三个线性(即y=Wx形式)层
        self.layer1 = nn.Linear(in_features=784, out_features=50)
        self.layer2 = nn.Linear(in_features=50, out_features=20)
        self.layer3 = nn.Linear(in_features=20, out_features=10)
        # 定义激活函数
        self.relu = nn.ReLU()
    # 编写前向传播方法
    def forward(self, x):
        # 扁平化
        x = x.view(x.size(0), -1)
        # nn.Module对象的括号运算符已经被重载为默认做前向传播
        a = self.relu(self.layer1(x))
        b = self.relu(self.layer2(a))
        # 分类问题，最后一层不用激活函数了
        y = self.layer3(b)
        return y
# 定义测试函数
def test(model, device, test_loader):
    # 设置模型为评估模式
    model.eval()
    total, correct = 0, 0
    # 不计算梯度，节省内存和计算资源
    with torch.no_grad():
        # 从test_loader中读取一批数据
        for samples, targets in test_loader:
            # 送至指定的设备(典型的是从内存调入显存以进行并行计算)
            samples, targets = samples.to(device), targets.to(device)
            # 正向传播，为了计算准确率
            outputs = model(samples)
            # 输出的是概率，我们要找到概率最大的分类
            max_value, max_indexes = torch.max(outputs.data, 1)
            # 更新样本总数和预测正确的数量
            total += targets.size(0)
            correct += (max_indexes == targets).sum()
        # 输出准确率信息
        print('Accuracy %d %%' % (100 * correct / total))
# 定义训练函数
def train(model, device, epoch, train_loader, test_loader):
    # 使用交叉熵损失函数，适合多分类问题
    criterion = nn.CrossEntropyLoss()
    # 建立优化器，model.parameters()会枚举类中所有的Module类型变量作为参数
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    # 开启迭代
    for i in range(epoch):
        # 打开模型的训练模式
        model.train()
        for batch_idx, (samples, targets) in enumerate(train_loader):
            # 送至指定的设备(典型的是从内存调入显存以进行并行计算)
            samples, targets = samples.to(device), targets.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 先前向传播，再通过损失函数计算输出和目标间的损失，最后把损失反向传播到所有节点
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            # 通过优化器的算法更新参数
            optimizer.step()
            # 每训练一小批数据，打印一次信息，并计算一次准确率
            if batch_idx % 100 == 0:
                print(f'Epoch: {i} [{batch_idx * len(samples)}/{len(train_loader.dataset)}\tLoss: {loss.item():.6f}]')
                test(model, device, test_loader)
# 定义数据集加载函数，读取数据后以DataLoader的包装形式返回
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # download=True意味着如果本地没有数据集会默认下载一份
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    return train_loader, test_loader
if __name__ == '__main__':
    # 加载数据集
    train_loader, test_loader = load_mnist()
    # 查找一个设备（算力硬件），如果有支持的显卡可以传递gpu
    device = torch.device("cpu")
    # 建立模型，并将其参数等上下文送入指定设备
    model = FCNN()
    model.to(device)
    # 调用训练方法和测试方法
    train(model, device, 5, train_loader, test_loader)
    # 将模型保存到文件
    torch.save(model.state_dict(), 'model.pkl')
# ========================================================================================
# 卷积神经网络
# ========================================================================================
class CNN(nn.Module):
    def __init__(self, in_dim=1176, out_dim=10):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.relu(self.conv(x))
        b = self.pool(a)
        y = self.fc(b.view(len(b), -1))
        return y
# ========================================================================================
# 循环神经网络
# ========================================================================================
class RNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=50, out_dim=10):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # 需要确定输入层、隐藏层和输出层神经元数
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_dim)
    # 编写前向传播方法
    def forward(self, x, h=None):
        # 若首次调用时无隐藏状态则使用0作为初始隐藏状态
        if h is None:
            # h的形状是(num_layers, batch_size, hidden_dim)
            h = torch.zeros(1, x.size(1), self.hidden_dim).to(x.device)
        # 经过RNN得到输出状态和隐藏状态
        # o的形状是(seq_len, batch_size, hidden_dim)
        o, h = self.rnn(x, h)
        # 若只关心最后一个状态，将其过线性层
        o = self.fc(o[-1])
        return o, h
# ========================================================================================