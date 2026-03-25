
<style>.markdown-body { counter-increment: level1 21; }</style>

# 机器学习应用简介

从本章开始的内容是机器学习在各领域的应用，每个领域大致从研究和工程两方面进行梳理。由于我并未涉猎所有领域，因此部分内容仅基于教材和公开资料，以至有些内容过于陈旧或简略。

## 主要会议和期刊

下面的主要学术会议和期刊按照中国计算机学会推荐2026年版本更新。

- TPAMI：IEEE Transactions on Pattern Analysis and Machine Intelligence（IEEE模式分析和机器智能汇刊），CCF-A类期刊。
- AAAI：AAAI Conference on Artificial Intelligence（美国人工智能协会年会），CCF-A类会议。
- ICLR：International Conference on Learning Representations（国际表示学习会议），CCF-A类会议。
- ICML：International Conference on Machine Learning（国际机器学习会议），CCF-A类会议。
- NeurIPS：Conference on Neural Information Processing Systems（神经信息处理系统年会），CCF-A类会议。
- ACL：Annual Meeting of the Association for Computational Linguistics（计算语言学年会），CCF-A类会议。
- CVPR：IEEE/CVF Computer Vision and Pattern Recognition Conference（IEEE计算机视觉与模式识别大会），CCF-A类会议。
- ICCV：International Conference on Computer Vision（国际计算机视觉会议），CCF-A类会议。

## 经典文献

### 传统机器学习

这个列表汇总在传统机器学习领域中的一些经典文献。暂时只列出了一篇，待进一步完善。

- Cortes C, Vapnik V. Support-Vector Networks[J]. Machine Learning, 1995, 20(3): 273-297. DOI: 10.1023/A:1022627411411.

### 神经网络和深度学习

这个列表汇总在深度学习领域中的一些经典文献。暂时只列出了几篇，待进一步完善。

- LeCun Y, Bottou L, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998.
- Krizhevsky A, Sutskever I, Hinton G. ImageNet Classification with Deep Convolutional Neural Networks[C]. NIPS, 2012.
- Bengio Y, Courville A, Vincent P. Representation learning: A review and new perspectives[J]. TPAMI, 2013.
- Szegedy C, Liu W, Jia Y, et al. Going Deeper with Convolutions[C]. CVPR, 2014.
- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]. MICCAI, 2015.
- He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.
- Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]. NIPS, 2017.
- Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]. NAACL, 2019.

## 著名数据集

这个部分汇总在机器学习领域中的著名数据集。暂时只列出了几个，待进一步完善。

### 自然语言处理

- NLP：Penn Treebank (PTB) 以华尔街日报为主要来源的文本语料，最初于上世纪90年代建立。

### 机器视觉

- MNIST：手写数字0~9图像，28×28灰度图像，训练数据6w张，测试数据1w张。
- CIFAR-10/100：10和100分类图像，32x32彩色图像。
- PASCAL VOC：共包含20类目标，一万多张图像，典型的是VOC 2007和VOC 2012。
- ImageNet：包含完整版和ILSVRC2012，上千万张标注图像，上万种类别。
- MS COCO：微软公布的大型图像数据集，包含目标检测、语义分割、实例分割、图像描述等标注数据。

## 知名竞赛

这个部分汇总在机器学习领域中的著名竞赛。暂时只列出了几个，待进一步完善。

- ILSVRC：ImageNet大型视觉识别挑战，现已经停办。
- MS COCO：继ILSVRC之后计算机视觉领域最受关注的比赛之一。

# 工程

## 模型示例

### 使用PyTorch训练神经网络模型

**简介**

PyTorch最重要的三个库分别是torch、torchvision和torchaudio。torch是PyTorch的核心库，提供GPU加速的张量计算，还包含通用的神经网络模块，以支持深度学习模型的构建和训练。torchvision专注于视觉任务，提供了图像加载、转换等操作，以及计算机视觉常用架构、预训练权重和常用数据集的封装。torchaudio专注于音频任务，提供了音频加载、转换等操作，以及信号处理常用架构、预训练权重以及对常用数据集的封装。

**张量计算和自动梯度**

PyTorch的张量计算过程中会自动建立计算图，以便在最终标量上调用backward方法即可自动进行反向传播。

```python
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
# @表示矩阵乘法X3 = (X1 - 5) x X2^T
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
```

**以全连接神经网络为例的完整训练过程**

下面以全连接神经网络为例，展示一个完整的训练过程。典型的PyTorch应用只需以下几个步骤：

- 导入PyTorch相关的包。
- 编写神经网络类，在其init方法中建立各层结构，添加forward方法实现前向传播。
- 编写数据集加载代码，并以数据加载器DataLoader来包装数据集，以供训练和测试使用。
- 实例化神经网络类（通常称其对象为模型model），初始化损失函数（通常称为准则criterion）、优化器(optimizer)。
- 通过for循环不断迭代的从数据加载器取出一批数据，进行前向传播、反向传播和参数更新。
- 在合适的时机输出训练信息，例如训练损失、训练准确率、测试损失、测试准确率等。
- 直至模型收敛或达到预期目标时停止。

```python
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
```

除了例子中的SGD优化器以外，也可以使用其它的优化器，比如`optim.Adam(model.parameters(), lr=0.001)`。

nn.Linear类用于实现仿射变换$y = xW^T + b$。其参数包括：

- in_features：输入特征维度（对于张量来说，是最后一个维度的大小）。
- out_features：输出特征维度（对于张量来说，是最后一个维度的大小。
- bias：默认为True，表示是否使用偏置$b$。

若任务不变，只想改变网络结构，只需新定义一个模型或者在原有模型上改动即可，其它代码通常不用较大修改。

**卷积神经网络**


```python
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
```

池化层的预定义类主要有nn.AvgPool1d/AvgPool2d/AvgPool3d和nn.MaxPool1d/nn.MaxPool2d/nn.MaxPool3d等。

以nn.MaxPool2d为例：输入张量的形状通常为$(N, C, H, W)$，其中$N$是批大小、$C$是通道数、$H$和$W$是特征的高度和宽度；输出张量的形状通常为$(N, C, H', W')$。例如100张640*480的彩色图像，其特征形状为(100, 3, 640, 480)。池化层的主要参数包括：

- kernel_size：池化窗口的大小。既可以是单一的整数，也可以是一个二元组，分别指定两个方向。
- stride：池化窗口的步长，默认和kernel_size一致。既可以是单一的整数，也可以是一个二元组，分别指定两个方向。

卷积层的预定义类包括nn.Conv1d、nn.Conv2d和nn.Conv3d等。

以nn.Conv2d为例：输入张量形状通常为$(N, C, H, W)$；输出张量形状通常是$(N, C, H', W')$。卷积层的主要参数包括：

- in_channels：输入的通道数，例如灰度图像为1，彩色图像为3。
- out_channels：输出通道数，即卷积核的个数。
- kernel_size：卷积核的大小。既可以是单个整数，也可以是由两个整数组成的元组/列表，分别表示高度和宽度。
- stride：卷积核在输入数据上滑动的步长。既可以是单个整数，也可以是由两个整数组成的元组/列表。
- padding：填充大小，用于保持输入和输出的尺寸一致。
- dilation：卷积核间距。
- groups：卷积核分组的数量，用于实现深度可分离卷积。
- bias：是否使用偏置项，默认为True。

**循环神经网络**

```python
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
```

nn.RNN在batch_first=False时的输入张量形状为$(L, N, H_{in})$，其中$L$是序列长度、$N$是批数量、$H_{in}$是每个时间步的特征维度。其参数包括：

- input_size：输入特征维度。对于图像来说可以是像素数。
- hidden_size：隐藏状态特征维度。它也是输出的维度。
- num_layers：RNN的深度，即隐藏层的个数，默认是1。
- nonlinearity：非线性激活函数的类型，可使用tanh（默认）或relu。
- bias：是否使用偏置。
- dropout：除最后一层以外Dropout的比率，默认是0。
- bidirectional：是否是双向RNN。