# ========================================================================================
# 使用numpy编写前馈全连接神经网络的示例
# ========================================================================================
import pickle
import numpy as np
import matplotlib.pyplot as plt
# S形函数单元
class SigmoidUnit:
    def __init__(self):
        # 缓存前向传播时的输出，在反向传播时使用
        self.y = None
    # 前向传播，每一个元素的值计算1/(1+e^-x)
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    # 反向传播，Sigmoid的求导结果很特殊，刚好是dx = dy * y(1 - y)
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx
    # Sigmoid的更新，因为无参数，所以什么也不用做
    def update(self, lr):
        return
# 仿射变换单元
class AffineUnit:
    def __init__(self, input_dim: int, output_dim: int, init_std=0.01):
        # 使用正态分布创建权重矩阵，同时创建偏置
        self.weight = init_std * np.random.randn(input_dim, output_dim).astype(np.float32)
        self.bias = np.zeros(output_dim).astype(np.float32)
        # 缓存前向传播时的输入，给反向传播使用
        self.x = None
        # 缓存反向传播时的dw和db，用于参数更新
        self.dW = None
        self.db = None
        return
    # 前向传播，计算y=xw+b
    def forward(self, x):
        self.x = x
        y = np.dot(x, self.weight) + self.bias
        return y
    # 反向传播，dx = dy * w.T, dw = x.T * dy, db = dy * 1
    def backward(self, dy):
        dx = np.dot(dy, self.weight.T)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return dx
    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db
# Softmax和交叉熵损失，整合为一个类的原因是它们的求导可以用性能更高的形式完成
class SoftmaxCrossEntropyOutputUnit:
    def __init__(self):
        # 缓存前向传播的输出，供反向传播时使用
        self.y = None
        # 缓存损失计算时的目标，供反向传播时使用
        self.t = None
        return
    # 正向传播，计算Softmax并存储输出
    def forward(self, x):
        x = x.T
        m = np.max(x, axis=0)
        e = np.exp(x - m, dtype=np.float32)
        s = np.sum(e, axis=0)
        y = e / s
        self.y = y.T
        return self.y
    # 反向传播，样本数量是矩阵的行数，把Softmax和交叉熵放在一起的好处是求导可简化为dx=(y-t)/count
    def backward(self, dy):
        count = self.y.shape[0]
        dx = (self.y - self.t) / count
        return dx
    # 求损失，
    def loss(self, t):
        self.t = t
        delta = 1e-7
        # 样本数是矩阵的行数
        count = self.y.shape[0]
        # t是一个one-hot向量，只有值为1的索引形成索引向量组
        t = t.argmax(axis=1)
        # 在y中获取要计算的数据，两个索引范围都是向量
        matrix = self.y[np.arange(count), t]
        return -np.sum(np.log(matrix + delta)) / count
    # Softmax和交叉熵损失的更新什么也不用做
    def update(self, lr):
        return
# 神经网络层的抽象类，一个层可以包含多个单元
class Layers:
    def __init__(self):
        self.units = []
        return
    def add_unit(self, unit):
        self.units.append(unit)
        return
    # 正向传播，逐级调用forward
    def forward(self, x):
        y = x
        for unit in self.units:
            y = unit.forward(y)
        return y
    # 反向传播，逆向逐级调用backward
    def backward(self, dy):
        i = 0
        dx = dy
        size = len(self.units)
        while i < size:
            dx = self.units[size - i - 1].backward(dx)
            i += 1
        return dx
    # 参数更新
    def update(self, lr):
        for unit in self.units:
            unit.update(lr)
# 隐藏层，从Layers下继承，内含仿射变换和Sigmoid两个单元，刚好是y=sigmoid(wx+b)
class HiddenLayer(Layers):
    def __init__(self, input_dim: int, output_dim: int, init_std=0.01):
        super().__init__()
        self.add_unit(AffineUnit(input_dim, output_dim, init_std))
        self.add_unit(SigmoidUnit())
# Softmax输出层，从Layers下继承，内含仿射变换和Softmax交叉熵两个单元，刚好是y=softmax(wx+b)
class SoftmaxOutputLayer(Layers):
    def __init__(self, input_dim: int, output_dim: int, weight_init_std=0.01):
        super().__init__()
        # 创建仿射变换线性单元和Softmax交叉熵单元
        self.linear = AffineUnit(input_dim, output_dim, weight_init_std)
        self.nonlinear = SoftmaxCrossEntropyOutputUnit()
        # 添加单元
        self.add_unit(self.linear)
        self.add_unit(self.nonlinear)
    # 求损失，就是调用Softmax交叉熵单元的损失
    def loss(self, t):
        return self.nonlinear.loss(t)
# 定义全连接神经网络
class FCNNNetwork:
    # layers_size是一个数组，描述每个层的神经元数量
    def __init__(self, layers_size):
        self.layers = []
        # 添加层，结构就是若干隐藏层加最后的Softmax输出层
        for i in range(len(layers_size) - 2):
            self.layers.append(HiddenLayer(layers_size[i], layers_size[i + 1]))
        self.layers.append(SoftmaxOutputLayer(layers_size[i + 1], layers_size[i + 2]))
    # 前向传播，就是依次调用各个层的前向传播
    def forward(self, samples):
        x = samples
        for layer in self.layers:
            x = layer.forward(x)
        return x
    # 反向传播，就是逆向依次调用各个层的反向传播
    def backward(self):
        dy = None
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
    # 求损失，典型的是在前向传播后，调用最后一层的loss()，根据标记和预测的值计算损失
    def loss(self, samples, labels):
        self.forward(samples)
        return self.layers[len(self.layers) - 1].loss(labels)
    # 更新参数
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
    # 预测有监督的样本，根据标记计算分类的精确度
    def classify_accuracy(self, samples, labels, batch=500):
        count = 0
        for i in range(0, len(samples), batch):
            batch_input = samples[i: i + batch]
            batch_output = self.forward(batch_input)
            batch_argmax = np.argmax(batch_output, axis=1)
            labels_argmax = np.argmax(labels[i: i + batch], axis=1)
            count += np.sum(batch_argmax == labels_argmax)
        return float(count) / len(samples)
    # 训练网络，要给数据集
    def train(self, X_train, y_train, X_test, y_test, lr, batch, step_num):
        # 获取样本数量，并计算每个epoch需要迭代多少次
        sample_count = X_train.shape[0]
        iter_per_epoch = max(sample_count // batch, 1)
        # 缓存画图的点
        loss_points = []
        acc_points = []
        # 迭代进行随机梯度下降
        for i in range(step_num):
            # 随机选择一批样本和它们的标记
            batch_indexes = np.random.choice(sample_count, batch)
            batch_samples = X_train[batch_indexes]
            batch_labels = y_train[batch_indexes]
            # 先求损失（内部会调用前向传播），然后反向将损失传播到各个神经元，最后更新参数
            self.loss(batch_samples, batch_labels)
            self.backward()
            self.update(lr)
            # 每个epoch打印损失和精确度
            if i % iter_per_epoch == 0:
                loss_points.append(self.loss(X_train, y_train))
                acc_points.append(self.classify_accuracy(X_train, y_train))
                print(f"Epoch: {i // iter_per_epoch + 1} "
                      f"Loss of train/test: {self.loss(X_train, y_train):.3f}/{self.loss(X_test, y_test):.3f}, "
                      f"Accuracy of train/test: {self.classify_accuracy(X_train, y_train):.3f}/{self.classify_accuracy(X_test, y_test):.3f}")
        # 创建图形
        fig = plt.figure()
        plt.plot(loss_points)
        plt.plot(acc_points)
# 加载MNIST数据集
def load_mnist():
    with open("mnist.pkl", 'rb') as f:
        dataset = pickle.load(f)
    # 归一化
    for key in ('train_img', 'test_img'):
        dataset[key] = dataset[key].astype(np.float32)
        dataset[key] /= 255.0
    # 将标记转换为one-hot
    for key in ('train_label', 'test_label'):
        labels = dataset[key]
        targets = np.zeros((labels.size, 10))
        for idx, row in enumerate(targets):
            row[labels[idx]] = 1
        dataset[key] = targets
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
# 程序入口
if __name__ == '__main__':
    # 加载数据集
    (X_train, y_train), (X_test, y_test) = load_mnist()
    # 创建网络，包括784个输入神经元，10个输出神经元，然后进行训练
    network = FCNNNetwork([784, 50, 10])
    network.train(X_train, y_train, X_test, y_test, 0.2, 500, 5000)
# ========================================================================================