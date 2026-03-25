# ========================================================================================
# 标题：用于实验和教学的深度学习脚手架
# 版本：2.13
# 参考：
# 1、《深度学习入门1：基于Python的理论与实现》
# 2、《深度学习入门2：自制框架》
# 3、《深度学习进阶：自然语言处理》
# 说明：这部分代码参考了斋藤康毅所著的深度学习三部曲(中译本)，只不过一些封装形式有所变化。
#       若对神经网络框架的实现细节感兴趣，非常推荐阅读这三本书籍。
#       除了书中介绍的网络结构以外，我自己也尝试了增加诸如VGG16、Transformer等部分。
#       受限于个人理解水平、单元测试不全等问题，代码难免有误，但作为教学来说是合适的。
#       另外，随着后期的不断完善，如果能修正错误、再引入cupy加速等提升性能的手段，
#       也或许可以作为实验脚手架使用。
# 描述：
# 一：网络组件
# 1、定义了基本接口，包括单元、组件、网络、优化器。
# 2、实现了基本单元，包括Identity、ReLU、Sigmoid、Softmax、Reshape、Sum、Broadcast、
#    Concatenate、Multiply、MatMultiply、Linear、Affine、Convolution、MaxPool、Dropout、
#    Embedding、EmbeddingBag、RNN/LSTM、TimeRNN/TimeLSTM、
#    PositionEncoding、Attention、TimeAttention、KeyValueAttentionUnit、SelfAttention、
#    BatchNormalization、LayerNormalization、MeanSquaredError、SoftmaxCrossEntropy。
# 3、实现了基本层，包括全连接层、卷积层、池化层、Inception层、Residual层、ResidualConvolution层、
#    Transformer层、SoftmaxOutput层。
# 4、实现了基本网络，包括BP网络、CNN网络、RNN网络。
# 5、实现了经典网络，VGG16、Word2Vec-CBOW、RNNLM、Seq2Seq、AttentionSeq2Seq、Transformer。
# 6、实现了基本优化器，包括SGD、Momentum、Adam。
# 7、实现了基本数据集类、数据加载器类、训练器。
# 8、demo部分给出了八个示例。
# 二、其它说明
# 1、为了保持简洁，从2.3版开始删除了PythonUtil类。
# 2、为了便于打印浏览，从2.5版开始所有的空行均被删除。
# 3、为了便于教学，从2.5版开始注释将逐步使用汉语。
# ========================================================================================
import os
import sys
import uuid
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
# ========================================================================================
# Global Configuration Static Class
# ========================================================================================
class Global:
    forward_log = False     # 是否打印前向传播日志
    backward_log = False    # 是否打印反向传播日志
    @staticmethod
    def enable_warning():
        warnings.filterwarnings("always", category=RuntimeWarning)
# ========================================================================================
# Math Class
# ========================================================================================
# 这个类在较新的版本中引入是想在后续将张量抽离成Tensor类，以支持Tensor的计算，不过目前它就是对numpy的封装
class Math:
    @staticmethod
    def max(tensor, axis=None):
        return np.max(tensor, axis=axis)
    @staticmethod
    def sqrt(tensor):
        return np.sqrt(tensor)
    @staticmethod
    def exp(tensor, dtype=None):
        return np.exp(tensor, dtype=dtype)
    @staticmethod
    def log(tensor, dtype=None):
        return np.log(tensor, dtype=dtype)
    @staticmethod
    def sin(tensor, dtype=None):
        return np.sin(tensor, dtype=dtype)
    @staticmethod
    def cos(tensor, dtype=None):
        return np.cos(tensor, dtype=dtype)
    @staticmethod
    def tanh(tensor):
        return np.tanh(tensor)
    @staticmethod
    def argmax(tensor, axis=None):
        return np.argmax(tensor, axis=axis)
    @staticmethod
    def dot(tensor1, tensor2):
        return np.dot(tensor1, tensor2)
    @staticmethod
    def sum(tensor, axis=None, dtype=None, keepdims=np._NoValue):
        return np.sum(tensor, axis=axis, dtype=dtype, keepdims=keepdims)
    @staticmethod
    def add_at(tensor, indices, b=None):
        return np.add.at(tensor, indices, b)
    @staticmethod
    def eye(tensor, M=None, k=0, dtype=float):
        return np.eye(tensor, M=M, k=k, dtype=dtype)
    @staticmethod
    def pad(tensor, pad_width, mode='constant', **kwargs):
        return np.pad(tensor, pad_width, mode=mode, **kwargs)
    @staticmethod
    def maximum(tensor, arg1):
        return np.maximum(tensor, arg1)
    @staticmethod
    def broadcast_to(tensor, shape):
        return np.broadcast_to(tensor, shape)
    @staticmethod
    def reshape(tensor, newshape, order='C'):
        return np.reshape(tensor, newshape=newshape, order=order)
    @staticmethod
    def concatenate(tensor, axis):
        return np.concatenate(tensor, axis=axis)
    @staticmethod
    def empty(shape, dtype=None):
        return np.empty(shape, dtype=dtype)
    @staticmethod
    def rand(*shape):
        return np.random.rand(*shape)
    @staticmethod
    def randn(*shape):
        return np.random.randn(*shape)
    @staticmethod
    def permutation(x):
        return np.random.permutation(x)
    @staticmethod
    def choice(l, size=None, replace=True, p=None):
        return np.random.choice(l, size=size, replace=replace, p=p)
    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)
    @staticmethod
    def zeros_like(tensor, dtype=None):
        return np.zeros_like(tensor, dtype=dtype)
    @staticmethod
    def ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)
    @staticmethod
    def ones_like(a, dtype=None):
        return np.ones_like(a, dtype=dtype)
    @staticmethod
    def arange(start=None, *args, **kwargs):
        return np.arange(start, *args, **kwargs)
    @staticmethod
    def array(p_object, dtype=None, *args, **kwargs):
        return np.array(p_object, dtype, *args, **kwargs)
# ========================================================================================
# Loss Function Static Class
# ========================================================================================
class LossFunction:
    @staticmethod
    def mean_squared_error(y, t):
        d = y - t
        return Math.sum(d ** 2) / len(d)
    @staticmethod
    # calculate with sum(t * log(y))
    def cross_entropy_error(y, t):
        delta = 1e-7
        # expanding a vector into a matrix with only one row vector
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        # the sample count is the number of rows in the matrix
        count = y.shape[0]
        # if t is a one-hot vector(they size are equal) but not a tag(ex. 1~10), then
        # only the index with a value of 1 forms the index vector group
        if t.size == y.size:
            t = t.argmax(axis=1)
        # get the data to be calculated in y, both indexes ranges are vectors
        matrix = y[Math.arange(count), t]
        return -Math.sum(Math.log(matrix + delta)) / count
# ========================================================================================
# Gradient Calculation Function Static Class
# ========================================================================================
class Gradient:
    @staticmethod
    def numerical_vector_gradient(f, x):
        """
        Calculate the numerical symmetric vector gradient.
        :param f: the target function
        :param x: the value of the independent variable (in vector) at the specified point
        :return: the symmetric differential vector
        """
        delta = 1e-4
        result = Math.zeros_like(x)
        for idx in range(x.size):
            old = x[idx]
            # calculate the right side increment
            x[idx] = old + delta
            inc_r = f(x)
            # calculate the left side increment
            x[idx] = old - delta
            inc_l = f(x)
            # calculate the differential
            result[idx] = (inc_r - inc_l) / (2 * delta)
            # restore old value
            x[idx] = old
        return result
    @staticmethod
    def numerical_matrix_gradient(f, x):
        """
        Calculate the gradient of function 'f'.
        :param f: the target function
        :param x: the value of the independent variable (in vector of matrix)
        at the specified point
        :return: the symmetric differential vector or matrix
        """
        if x.ndim == 1:
            return Gradient.numerical_vector_gradient(f, x)
        else:
            result = Math.zeros_like(x)
            for idx, v in enumerate(x):
                result[idx] = Gradient.numerical_vector_gradient(f, v)
            return result
# ========================================================================================
# Convolutional Flattening Utils
# ========================================================================================
# 这两个函数参考自原书，逻辑较复杂，建议读者参考原书理解
def im2col(inputs, filter_width, filter_height, stride=1, pad=0):
    batch_size, channel, input_width, input_height = inputs.shape
    output_width = (input_width + 2 * pad - filter_width) // stride + 1
    output_height = (input_height + 2 * pad - filter_height) // stride + 1
    img = np.pad(inputs, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = Math.zeros((batch_size, channel, filter_width, filter_height, output_width, output_height), np.float32)
    for x in range(filter_width):
        x_max = x + stride * output_width
        for y in range(filter_height):
            y_max = y + stride * output_height
            col[:, :, x, y, :, :] = img[:, :, x:x_max:stride, y:y_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * output_height * output_width, -1)
    return col
# ========================================================================================
def col2im(inputs, inputs_shape, filter_width, filter_height, stride=1, pad=0):
    batch_size, channel, input_width, input_height = inputs_shape
    outputs_width = (input_width + 2 * pad - filter_width) // stride + 1
    outputs_height = (input_height + 2 * pad - filter_height) // stride + 1
    col = inputs.reshape(batch_size, outputs_width, outputs_height, channel, filter_width, filter_height).transpose(0, 3, 4, 5, 1, 2)
    img = Math.zeros((batch_size, channel, input_width + 2 * pad + stride - 1, input_height + 2 * pad + stride - 1), np.float32)
    for x in range(filter_width):
        x_max = x + stride * outputs_width
        for y in range(filter_height):
            y_max = y + stride * outputs_height
            img[:, :, x:x_max:stride, y:y_max:stride] += col[:, :, x, y, :, :]
    return img[:, :, pad:input_width + pad, pad:input_height + pad]
# ========================================================================================
# Neural Network Interfaces and Abstract Classes
# ========================================================================================
class Unit:
    # set work mode, 'train' or 'predict'
    def set_train(self, is_train: bool):
        pass
    # Returns unit parameters dictionary list.
    def parameters(self) -> list:
        pass
    # Returns unit grad dictionary list.
    def gradients(self) -> list:
        pass
    # Forward propagation, calculating forward output through input.
    def forward(self, x):
        pass
    # Back propagation, which propagates the differential result backwards.
    def backward(self, dy):
        pass
    # Calculate the gradient of function 'f'.
    def gradient(self, loss: Callable[[np.ndarray, np.ndarray], float]) -> None:
        pass
# ========================================================================================
class OutputUnit(Unit):
    # Calculate loss between output and target.
    def loss(self, t):
        pass
# ========================================================================================
# A component is a unit, and it contains multiple sub-units.
class Component(Unit):
    def __init__(self):
        # all units in current component, each element is an unit
        self.units = {}
        self.train = False
    # Add neural network unit.
    def add_unit(self, unit: Unit, name=None):
        if name is None:
            name = str(uuid.uuid4())
        self.units[name] = unit
    def set_train(self, is_train: bool):
        self.train = is_train
        for unit in self.units.values():
            unit.set_train(is_train)
    def parameters(self) -> list:
        params = []
        for unit in self.units.values():
            params = params + unit.parameters()
        return params
    def gradients(self) -> list:
        grads = []
        for unit in self.units.values():
            grads = grads + unit.gradients()
        return grads
    def forward(self, x):
        y = x
        for unit in self.units.values():
            if Global.forward_log is True and isinstance(x, np.ndarray):
                print("Unit '" + str(unit.__class__) + "' forward: min(" + str(round(y.min(), 6)) + ") max(" + str(round(y.max(), 6)) + ")")
            y = unit.forward(y)
        return y
    def backward(self, dy):
        dx = dy
        for unit in reversed(self.units.values()):
            if Global.backward_log is True and isinstance(dy, np.ndarray):
                print("Unit '" + str(unit.__class__) + "' backward: min(" + str(round(dy.min(), 6)) + ") max(" + str(round(dy.max(), 6)) + ")")
            dx = unit.backward(dx)
        return dx
    def gradient(self, loss: Callable[[np.array], float]) -> None:
        for unit in self.units.values():
            unit.gradient(loss)
# ========================================================================================
# A network is a component
class Network(Component):
    # Save network parameters to file
    def save(self, path: str = None) -> None:
        path = path if path is not None else self.__class__.__name__ + ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(self.units, f)
    # Load network parameters from file
    def load(self, path: str = None) -> None:
        path = path if path is not None else self.__class__.__name__ + ".pkl"
        if not os.path.exists(path):
            print('PKL File of "' + path + '" is not exist.')
        else:
            with open(path, 'rb') as f:
                self.units = pickle.load(f)
    # This method must call predict() automatically.
    def loss(self, samples, labels) -> float:
        # predict first
        self.forward(samples)
        # get the output layer index
        modules = list(self.units.values())
        return modules[len(self.units) - 1].loss(labels)
    def accuracy(self, samples, labels) -> float:
        pass
# ========================================================================================
class Optimizer:
    # Update parameters according backward calculation.
    def update(self, params: list, grads: list) -> None:
        pass
    @staticmethod
    def clip_grads(grads, max_norm):
        total_norm = 0
        for grad in grads:
            total_norm += Math.sum(grad ** 2)
        total_norm = Math.sqrt(total_norm)
        rate = max_norm / (total_norm + 1e-6)
        if rate < 1:
            for grad in grads:
                grad *= rate
# ========================================================================================
# Network Basic Unit
# ========================================================================================
class SimpleUnit(Unit):
    def set_train(self, is_train: bool):
        self.train = is_train
    def parameters(self) -> list:
        return []
    def gradients(self) -> list:
        return []
    def forward(self, x):
        return x
    def backward(self, dy):
        return dy * 1
    def gradient(self, loss: Callable[[np.ndarray, np.ndarray], float]) -> None:
        return
# ========================================================================================
class ParameterUnit(SimpleUnit):
    def __init__(self):
        self.params = {}
        self.grads = {}
    def parameters(self) -> list:
        return [self.params]
    def gradients(self) -> list:
        return [self.grads]
    def gradient(self, loss: Callable[[np.ndarray, np.ndarray], float]) -> None:
        for key in self.params:
            self.grads[key] = Gradient.numerical_matrix_gradient(loss, self.params[key])
# ========================================================================================
class IdentityUnit(SimpleUnit):
    def forward(self, x):
        return x
    def backward(self, dy):
        return dy * 1
# ========================================================================================
class ReluUnit(SimpleUnit):
    def __init__(self):
        # cache the latest forward output result (elements less than 0) for backward
        self.mask = None
    def forward(self, x):
        # value of each element calculate by max(0, x)
        self.mask = (x <= 0)
        y = Math.maximum(0, x)
        return y
    def backward(self, dy):
        dy[self.mask] = 0
        dx = dy
        return dx
# ========================================================================================
class SigmoidUnit(SimpleUnit):
    def __init__(self):
        # cache the latest forward output result for backward
        self.y = None
    def forward(self, x):
        # value of each element calculate by 1/(1+e^-x)
        self.y = 1 / (1 + Math.exp(-x, dtype=np.float32))
        return self.y
    def backward(self, dy):
        # dx = dy * y(1 - y)
        dx = dy * (1.0 - self.y) * self.y
        return dx
# ========================================================================================
class GeluUnit(SigmoidUnit):
    def forward(self, x):
        self.x = x
        self.h = super().forward(1.702 * x)
        return self.x * self.h
    def backward(self, dy):
        dh = dy * self.x
        dx1 = dy * self.h
        dx2 = super().backward(dh) * 1.702
        return dx1 + dx2
# ========================================================================================
class SoftmaxUnit(SimpleUnit):
    def forward(self, x):
        self.y = SoftmaxUnit.softmax(x)
        return self.y
    def backward(self, dy):
        dx = self.y * dy
        dx -= self.y * Math.sum(dx, axis=1, keepdims=True)
        return dx
    @staticmethod
    def softmax(x):
        """
        The softmax funciton.
        :param x: [V/M] vector group in a matrix or a single vector
        :return: [V/M] vector in group with each of value calculate by e^v[i] / e^sum(v[0] + ... + v[n])
        """
        # Considering the overflow situation in exponential operations, before the operation,
        # each element of each vector group is subtracted to the maximum value in that group.
        # It can be inferred that the final result is not affected by this.
        if x.ndim == 1:
            max = Math.max(x)
            ex = Math.exp(x - max, dtype=np.float32)
            ex_sum = Math.sum(ex)
            return ex / ex_sum
        else:
            x = x.T
            max = Math.max(x, axis=0)
            ex = Math.exp(x - max, dtype=np.float32)
            ex_sum = Math.sum(ex, axis=0)
            y = ex / ex_sum
            return y.T
# ========================================================================================
class ReshapeUnit(SimpleUnit):
    def forward(self, x):
        (x, shape) = x
        self.x_shape = x.shape
        return x.reshape(shape)
    def backward(self, dy):
        return dy if dy is None or np.isscalar(dy) else dy.reshape(self.x_shape)
# ========================================================================================
class ReshapeToTimeBatchUnit(ReshapeUnit):
    def forward(self, x):
        self.shape = (x.shape[0] * x.shape[1], -1)
        return super().forward((x, self.shape))
# ========================================================================================
class UnreshapeToTimeBatchUnit(ReshapeUnit):
    def __init__(self, reshape_unit: ReshapeToTimeBatchUnit):
        self.reshape_unit = reshape_unit
    def forward(self, x):
        self.shape = (self.reshape_unit.x_shape[0], self.reshape_unit.x_shape[1], -1)
        return super().forward((x, self.shape))
# ========================================================================================
class SumUnit(SimpleUnit):
    def __init__(self, axis=None):
        self.axis = axis
    def forward(self, x):
        self.shape = x.shape
        return x.sum(axis=self.axis)
    def backward(self, dy):
        def change_tuple(tpl, index, value):
            return tpl[:index] + (value,) + tpl[index + 1:]
        shape = change_tuple(self.shape, self.axis, 1)
        dy = dy.reshape(shape)
        dx = Math.broadcast_to(dy, self.shape)
        return dx
# ========================================================================================
class BroadcastUnit(SimpleUnit):
    def __init__(self, axis):
        self.axis = axis
    def forward(self, x):
        # 输入x需携带要广播到什么形状
        x, self.shape = x
        def change_tuple(tpl, index, value):
            return tpl[:index] + (value,) + tpl[index + 1:]
        # 先扩展出指定的轴(.., 1, ..)
        shape = change_tuple(self.shape, self.axis, 1)
        x = x.reshape(shape)
        # 然后在轴上广播
        return Math.broadcast_to(x, self.shape)
    def backward(self, dy):
        return Math.sum(dy, self.axis)
# ========================================================================================
class ConcatenateUnit(SimpleUnit):
    def __init__(self, axis):
        self.axis = axis
    def forward(self, x):
        x1, x2 = x
        # 记住两个变量在合并的轴上分别有多少个维度
        self.x1_dim = x1.shape[self.axis]
        self.x2_dim = x2.shape[self.axis]
        return Math.concatenate(x, axis=self.axis)
    def backward(self, dy):
        def slice_along_axis(arr, axis, start, end):
            # 创建全选切片的列表
            slices = [slice(None)] * arr.ndim
            # 替换目标轴的切片范围
            slices[axis] = slice(start, end)
            # 返回切片后的数组
            return arr[tuple(slices)]
        return slice_along_axis(dy, self.axis, 0, self.x1_dim), slice_along_axis(dy, self.axis, self.x1_dim, self.x1_dim + self.x2_dim)
# ========================================================================================
class MultiplyUnit(SimpleUnit):
    def forward(self, x):
        # 输入两个乘积
        self.x1, self.x2 = x
        return self.x1 * self.x2
    def backward(self, dy):
        dx1 = self.x2 * dy ; dx2 = self.x1 * dy
        return dx1, dx2
# ========================================================================================
class MatMultiplyUnit(ParameterUnit):
    def __init__(self, input_dimension: int, output_dimension: int, weight_init_std=0.01):
        super().__init__()
        # create weight matrix using 'Normal Distribution'
        weight = weight_init_std * Math.randn(input_dimension, output_dimension).astype(np.float32)
        # init params and grads
        self.params = {"weight": weight}
        self.grads = {"weight": None}
        # cache input
        self.x = None
    def forward(self, x):
        self.x = x
        y = Math.dot(x, self.params["weight"])
        return y
    def backward(self, dy):
        dx = Math.dot(dy, self.params["weight"].T)
        self.grads["weight"] = Math.dot(self.x.T, dy)
        return dx
# ========================================================================================
class LinearUnit(ParameterUnit):
    def __init__(self, input_dimension: int, output_dimension: int, weight_init_std=0.01):
        super().__init__()
        # create weight matrix
        weight = weight_init_std * Math.randn(input_dimension, output_dimension).astype(np.float32)
        # init params and grads
        self.params = {"weight": weight}
        self.grads = {"weight": None}
        # cache latest forward input
        self.x, self.x_shape = None, None
    def forward(self, x):
        # flattening tensor to vector
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        # get weight, bias and sigma
        x = self.x
        W = self.params["weight"]
        # forward value: y = x * w
        y = Math.dot(x, W)
        return y
    def backward(self, dy):
        # dx = dy * T(w), dw = T(x) * dy
        dx = Math.dot(dy, self.params["weight"].T)
        self.grads["weight"] = Math.dot(self.x.T, dy)
        # restore vector to tensor
        dx = dx.reshape(*self.x_shape)
        return dx
# ========================================================================================
class AffineUnit(LinearUnit):
    # Init affine unit by dimension.
    def __init__(self, input_dimension: int, output_dimension: int, weight_init_std=0.01):
        super().__init__(input_dimension, output_dimension, weight_init_std)
        bias = Math.zeros(output_dimension, np.float32)
        # init params and grads
        self.params["bias"] = bias
        self.grads["bias"] = None
    def forward(self, x):
        y = super().forward(x)
        b = self.params["bias"]
        return y + b
    def backward(self, dy):
        # db = dy * 1
        self.grads["bias"] = Math.sum(dy, axis=0)
        return super().backward(dy)
# ========================================================================================
class ConvolutionUnit(ParameterUnit):
    def __init__(self, filter_count, channel, filter_width, filter_height, stride=1, pad=0, weight_init_std=0.01):
        super().__init__()
        # fan_out将使用He权重按扇出数进行初始化（下一层神经元输入数量，即当前层神经元数量）
        if weight_init_std == 'fan_out':
            weight_init_std = Math.sqrt(2 / (filter_count * filter_width * filter_height))
        weight = weight_init_std * Math.randn(filter_count, channel, filter_width, filter_height).astype(np.float32)
        bias = Math.zeros(filter_count, dtype=np.float32)
        self.params = {"weight": weight, "bias": bias}
        self.grads = {"weight": None, "bias": None}
        self.stride, self.pad = stride, pad
        # 缓存反向传播需要的数据
        self.X, self.inputs, self.filters = None, None, None
    def forward(self, X):
        # 获得卷积核(滤波器)的各项尺寸
        filter_count, filter_channel, filter_width, filter_height = self.params["weight"].shape
        # 获得输入的各项尺寸
        batch_size, channel, input_width, input_height = X.shape
        # 计算输出大小
        output_width = 1 + int((input_width + self.pad * 2 - filter_width) / self.stride)
        output_height = 1 + int((input_height + self.pad * 2 - filter_height) / self.stride)
        # 将输入的数据组织成二维数组，列数为卷积核参数数=滤波器通道数x滤波器宽度x滤波器高度
        # 行数将输入数据是按卷积核参数长度，与输入数据位置一一对应(其大小已经和待卷积后输出的大小相同了)
        inputs = im2col(X, filter_width, filter_height, self.stride, self.pad)
        # 将卷积核也变成二维数组，行数为滤波器个数，列数自动由剩余形状计算(显然就是滤波器通道数x滤波器宽度x滤波器高度=卷积核参数数)，再转置
        filters = self.params["weight"].reshape(filter_count, -1)
        # 卷积的过程就是矩阵乘法，只不过通过上面的特殊处理可以一次批量完成
        # 显然输出的二维数组以卷积后的输出数据为行，以滤波器数量为列
        outputs = Math.dot(inputs, filters.T) + self.params["bias"]
        # 按照批大小、输出宽度、输出高度、滤波器数量重新组织数据
        outputs = outputs.reshape(batch_size, output_width, output_height, -1)
        # 按照批大小、滤波器数量、输出宽度、输出高度重新组织数据
        outputs = outputs.transpose(0, 3, 1, 2)
        self.X, self.inputs, self.filters = X, inputs, filters
        return outputs
    def backward(self, dY):
        # 获得卷积核(滤波器)的各项尺寸
        filter_count, filter_channel, filter_width, filter_height = self.params["weight"].shape
        # 反向传播的输入尺寸与正向的输出尺寸一样(批大小、滤波器数量、输出宽度、输出高度)
        # 需要通过两个步骤，参考正向传播进行一个逆变换，按批大小、输出宽度、输出高度、滤波器数量重新组织数据
        # 然后变成以上游梯度数据为行，滤波器数量为列的二维数组，便于做矩阵乘法
        dY = dY.transpose(0, 2, 3, 1)
        dY = dY.reshape(-1, filter_count)
        # 偏置的梯度是对所有滤波器的梯度数据求和
        db = Math.sum(dY, axis=0)
        # 通过输入数据和上游梯度计算卷积核的梯度
        dW = Math.dot(self.inputs.T, dY)
        # 恢复滤波器的构型
        dW = dW.T
        dW = dW.reshape(filter_count, filter_channel, filter_width, filter_height)
        self.grads["weight"] = dW
        self.grads["bias"] = db
        # 向上游传递梯度
        dX = Math.dot(dY, self.filters)
        dX = col2im(dX, self.X.shape, filter_width, filter_height, self.stride, self.pad)
        return dX
# ========================================================================================
class MaxPoolUnit(SimpleUnit):
    def __init__(self, pool_width, pool_height, stride=1, pad=0):
        self.pool_width, self.pool_height = pool_width, pool_height
        self.stride, self.pad = stride, pad
        self.X, self.maxouts = None, None
    def forward(self, X):
        # 获得输入的各项尺寸
        batch_size, channel, input_width, input_height = X.shape
        # 计算输出的大小
        output_width = int(1 + (input_width + self.pad * 2 - self.pool_width) / self.stride)
        output_height = int(1 + (input_height + self.pad * 2 - self.pool_height) / self.stride)
        # 将输入的数据组织成二维数组，列数为池化滤波器参数数=滤波器通道数x滤波器宽度x滤波器高度
        # 行数将输入数据按池化滤波器的参数长度，与输入数据位置一一对应(其大小已经和待卷积后输出的大小相同了)
        inputs = im2col(X, self.pool_width, self.pool_height, self.stride, self.pad)
        # 按每个池化滤波器分组，这样最大池化就是对每行数据，求滤波器大小中最大的
        inputs = inputs.reshape(-1, self.pool_height * self.pool_width)
        outputs = Math.max(inputs, axis=1)
        # 要把池化时的，每行数据中最大的那个单元记录下来，反向传播只与它有关
        maxouts = Math.argmax(inputs, axis=1)
        # 重新组织输出数据
        outputs = outputs.reshape(batch_size, output_width, output_height, channel)
        outputs = outputs.transpose(0, 3, 1, 2)
        self.X, self.maxouts = X, maxouts
        return outputs
    def backward(self, dY):
        # 为了后续计算，要把批大小、滤波器数量、输出宽度、输出高度的形状，变成批大小、输出宽度、输出高度、滤波器数量
        dY = dY.transpose(0, 2, 3, 1)
        # 生成一个梯度形状全为0的二维数组
        dX = Math.zeros((dY.size, self.pool_width * self.pool_height), dtype=np.float32)
        dX[Math.arange(self.maxouts.size), self.maxouts.flatten()] = dY.flatten()
        dX = dX.reshape(dY.shape + (self.pool_width * self.pool_height,))
        dX = dX.reshape(dX.shape[0] * dX.shape[1] * dX.shape[2], -1)
        dX = col2im(dX, self.X.shape, self.pool_width, self.pool_height, self.stride, self.pad)
        return dX
# ========================================================================================
class DropoutUnit(SimpleUnit):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        self.mask = None
    def forward(self, x):
        if self.train:
            self.mask = Math.rand(*x.shape) > self.ratio
            return x * self.mask
        else:
            return x * (1.0 - self.ratio)
    def backward(self, dy):
        return dy * self.mask
# ========================================================================================
class EmbeddingUnit(ParameterUnit):
    def __init__(self, vocab_size, hidden_size, weight_init_std=0.01):
        super().__init__()
        weight = weight_init_std * Math.randn(vocab_size, hidden_size).astype(np.float32)
        self.params = {"weight": weight}
        self.grads = {"weight": None}
        self.idx = None
    # xs is one-hot (batch_size, context_size, one-hot)
    def forward(self, x):
        W = self.params["weight"]
        # 由于x是one-hot向量，因此与W_in的乘积相当于抽取W_in指定的行
        self.idx = Math.argmax(x, axis=1)
        return W[self.idx]
    def backward(self, dy):
        dW = Math.zeros_like(self.params["weight"])
        Math.add_at(dW, self.idx, dy)
        self.grads["weight"] = dW
        return None
# ========================================================================================
class EmbeddingBagUnit(ParameterUnit):
    def __init__(self, vocab_size, hidden_size, weight_init_std=0.01):
        super().__init__()
        weight = weight_init_std * Math.randn(vocab_size, hidden_size).astype(np.float32)
        self.params = {"weight": weight}
        self.grads = {"weight": None}
        self.idxs = None
    # xs is one-hot (batch_size, context_size, one-hot)
    def forward(self, xs):
        batch_size, context_size, feature_count, = xs.shape
        y = 0
        self.idxs = []
        # 对所有上下文求嵌入并将结果相加
        for i in range(context_size):
            x = xs[:, i]
            W = self.params["weight"]
            # 由于x是one-hot向量，因此与W_in的乘积相当于抽取W_in指定的行
            idx = Math.argmax(x, axis=1)
            y += W[idx]
            self.idxs.append(idx)
        # 取平均值作为输出
        return y / context_size
    def backward(self, dy):
        dW = Math.zeros_like(self.params["weight"])
        context_size = len(self.idxs)
        dy = dy / context_size
        for i in range(context_size):
            Math.add_at(dW, self.idxs[i], dy)
        self.grads["weight"] = dW
        return None
# ========================================================================================
class RNNUnit(ParameterUnit):
    def __init__(self, input_dimension, output_dimension, type="rnn"):
        super().__init__()
        self.type = type
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        weight_x, weight_h, bias = self.init_matrix(input_dimension, output_dimension, type)
        self.params = {"weight_x": weight_x, "weight_h": weight_h, "bias": bias}
        self.grads = {"weight_x": None, "weight_h": None, "bias": None}
        self.reset()
    def init_matrix(self, input_dimension, output_dimension, type):
        X, H = input_dimension, output_dimension
        if type == "rnn":
            weight_x = (Math.randn(X, H) / Math.sqrt(X)).astype(np.float32)
            weight_h = (Math.randn(H, H) / Math.sqrt(H)).astype(np.float32)
            bias = Math.zeros(H).astype(np.float32)
            return weight_x, weight_h, bias
        elif type == "lstm":
            weight_x = (Math.randn(X, H * 4) / Math.sqrt(X)).astype(np.float32)
            weight_h = (Math.randn(H, H * 4) / Math.sqrt(H)).astype(np.float32)
            bias = Math.zeros(H * 4).astype(np.float32)
            return weight_x, weight_h, bias
        else:
            raise Exception("Error RNN type. The value must be: 'rnn' or 'lstm'.")
    def reset(self, h=None, c=None):
        self.h = h
        self.xs = []
        self.hs = []
        self.dh = None
        if self.type == "lstm":
            self.c = c
            self.cs = []
            self.fs = []
            self.gs = []
            self.Is = []
            self.os = []
            self.dc = None
    def get_dh(self):
        return self.dh
    def get_dc(self):
        return self.dc
    def forward(self, x):
        batch_size, dimension = x.shape
        # 得到参数
        Wx, Wh, b = self.params["weight_x"], self.params["weight_h"], self.params["bias"]
        # 获取上一个隐状态，若为None，则初始化为0
        h = Math.zeros((batch_size, self.output_dimension), dtype=np.float32) if self.h is None else self.h
        # 前向线性计算
        t = Math.dot(x, Wx) + Math.dot(h, Wh) + b
        # 前向非线性计算
        if self.type == 'rnn':
            y = self.forward_rnn(x, t)
            self.xs.append(x)
            self.hs.append(y)
            return self.hs
        else:
            (f, g, i, o, s, y) = self.forward_lstm(x, t)
            self.fs.append(f)
            self.gs.append(g)
            self.Is.append(i)
            self.os.append(o)
            self.cs.append(s)
            self.hs.append(y)
            return self.hs
    def backward(self, dys):
        # 获取dys的形状
        length, batch_size, dimension = dys.shape
        # 初始化梯度
        self.grads["weight_x"], self.grads["weight_h"], self.grads["bias"] = 0, 0, 0
        # 初始化中间梯度缓存
        dh, dc = 0, 0
        dxs = Math.empty((length, batch_size, self.input_dimension), np.float32)
        # 依次向前进行反向传播
        for j in reversed(range(length)):
            if self.type == "rnn":
                (dh, dx, dWx, dWh, db) = self.backward_rnn(dy=dys[j], dh=dh, y=self.hs[j], h=self.hs[j - 1], x=self.xs[j])
            else:
                (dh, dc, dx, dWx, dWh, db) = self.backward_lstm(dy=dys[j], dh=dh, dc=dc, x=self.xs[j],
                                                                s=self.cs[j], c=self.cs[j - 1], h=self.hs[j - 1],
                                                                f=self.fs[j], i=self.Is[j], g=self.gs[j], o=self.os[j])
            # 记录输入的梯度
            dxs[j] = dx
            # 加到梯度上
            self.grads["weight_x"] += dWx
            self.grads["weight_h"] += dWh
            self.grads["bias"] += db
        self.dh = dh
        self.dc = dc
        return dxs
    def forward_rnn(self, x, t):
        y = Math.tanh(t)
        # 缓存当前状态、输入和输出，并将所有状态输出
        self.h = y
        return y
    def backward_rnn(self, dy, dh, y, h, x):
        # 得到参数
        Wx, Wh = self.params["weight_x"], self.params["weight_h"]
        # 非线性的反向传播
        dt = (dy + dh) * (1 - y ** 2)
        # 线性的反向传播
        db = Math.sum(dt, axis=0)
        dWh = Math.dot(h.T, dt)
        dh = Math.dot(dt, Wh.T)
        dWx = Math.dot(x.T, dt)
        dx = Math.dot(dt, Wx.T)
        return (dh, dx, dWx, dWh, db)
    def forward_lstm(self, x, t):
        def sigmoid(x):
            return 1 / (1 + Math.exp(-x))
        # 获取形状
        H = self.output_dimension
        N, D = x.shape
        # 依次抽取矩阵的每一个列，计算四个门
        f = sigmoid(t[:, 0 * H: 1 * H])
        g = Math.tanh(t[:, 1 * H: 2 * H])
        i = sigmoid(t[:, 2 * H: 3 * H])
        o = sigmoid(t[:, 3 * H: 4 * H])
        # 获取上一个隐状态，若为None，则初始化为0
        c = Math.zeros((N, H), dtype=np.float32) if self.c is None else self.c
        # 计算下一时刻的c和h
        s = f * c + g * i
        y = o * Math.tanh(s)
        # 缓存当前状态、输入和输出，并将所有状态输出
        self.c = s
        self.h = y
        return (f, g, i, o, s, y)
    def backward_lstm(self, dy, dh, dc, x, s, c, h, f, i, g, o):
        # 得到参数
        Wx, Wh = self.params["weight_x"], self.params["weight_h"]
        # 计算对dy对do和ds的导数
        tanhc = Math.tanh(s)
        do = dy * tanhc
        ds = dc + (dy * o) * (1 - tanhc ** 2)
        # 计算ds对df、dc（上一状态）、dg、di的导数
        df = ds * c
        dc = ds * f
        dg = ds * i
        di = ds * g
        # 计算df、dg、di、do对各自权重和偏置的导数
        df = df * f * (1 - f)
        dg = dg * (1 - g ** 2)
        di = di * i * (1 - i)
        do = do * o * (1 - o)
        # 拼接四个导数为一个矩阵
        dt = np.hstack((df, dg, di, do))
        # 计算dt对dw、dh和db的导数
        dWh = Math.dot(h.T, dt)
        dWx = Math.dot(x.T, dt)
        db = dt.sum(axis=0)
        # 计算dt对dx和dh的导数
        dh = Math.dot(dt, Wh.T)
        dx = Math.dot(dt, Wx.T)
        return (dh, dc, dx, dWx, dWh, db)
# ========================================================================================
class TimeRNNUnit(RNNUnit):
    def __init__(self, input_dimension, output_dimension, type="rnn", stateful=False):
        super().__init__(input_dimension, output_dimension, type)
        self.stateful = stateful
    def forward(self, xs):
        batch_size, time_length, dimension = xs.shape
        # 得到参数
        Wx, Wh, b = self.params["weight_x"], self.params["weight_h"], self.params["bias"]
        # 如果不是有状态的，则每一次都清空隐状态
        if not self.stateful:
            self.reset()
        # 初始化变量内部缓存
        self.xs = xs
        self.hs = Math.empty((batch_size, time_length, self.output_dimension), np.float32)
        self.cs = Math.empty((batch_size, time_length, self.output_dimension), np.float32) if self.type == "lstm" else None
        self.fs = Math.empty((batch_size, time_length, self.output_dimension), np.float32) if self.type == "lstm" else None
        self.gs = Math.empty((batch_size, time_length, self.output_dimension), np.float32) if self.type == "lstm" else None
        self.Is = Math.empty((batch_size, time_length, self.output_dimension), np.float32) if self.type == "lstm" else None
        self.os = Math.empty((batch_size, time_length, self.output_dimension), np.float32) if self.type == "lstm" else None
        # 依次前向计算每个时刻
        for j in range(time_length):
            # 获取上一个隐状态，若为None，则初始化为0
            h = Math.zeros((batch_size, self.output_dimension), np.float32) if self.h is None else self.h
            # 前向线性计算
            x = xs[:, j, :]
            t = Math.dot(x, Wx) + Math.dot(h, Wh) + b
            # 前向非线性计算
            if self.type == "rnn":
                y = self.forward_rnn(x, t)
                self.hs[:, j, :] = y
            else:
                (f, g, i, o, s, y) = self.forward_lstm(x, t)
                self.hs[:, j, :], self.cs[:, j, :] = y, s
                self.fs[:, j, :], self.gs[:, j, :], self.Is[:, j, :], self.os[:, j, :] = f, g, i, o
        # 返回所有隐状态
        return self.hs
    def backward(self, dys):
        # 获取dys的形状
        batch_size, time_length, dimension = dys.shape
        # 缓存所有时刻的梯度
        self.grads["weight_x"], self.grads["weight_h"], self.grads["bias"] = 0, 0, 0
        # 初始化中间梯度缓存
        dh, dc = 0, 0
        dxs = Math.empty((batch_size, time_length, self.input_dimension), np.float32)
        # 从后向前反向传播
        for j in reversed(range(time_length)):
            if self.type == "rnn":
                (dh, dx, dWx, dWh, db) = self.backward_rnn(dy=dys[:, j, :], dh=dh,
                                       y=self.hs[:, j, :], h=self.hs[:, j - 1, :], x=self.xs[:, j, :])
            else:
                (dh, dc, dx, dWx, dWh, db) = self.backward_lstm(dy=dys[:, j, :], dh=dh, dc=dc, x=self.xs[:, j, :],
                    s=self.cs[:, j, :], c=self.cs[:, j - 1, :], h=self.hs[:, j - 1, :],
                    f=self.fs[:, j, :], i=self.Is[:, j, :], g=self.gs[:, j, :], o=self.os[:, j, :], )
            # 记录输入的梯度
            dxs[:, j, :] = dx
            # 添加梯度
            self.grads["weight_x"] += dWx
            self.grads["weight_h"] += dWh
            self.grads["bias"] += db
        self.dh = dh
        self.dc = dc
        return dxs
# ========================================================================================
class PositionEncodingUnit(SimpleUnit):
    def forward(self, x):
        batch_size, time_length, feature_count = x.shape
        code = self.encode(time_length, feature_count)
        return x + code
    def encode(self, P, D):
        # 生成位置索引向量(0,...,P)和维度索引向量(0,2,4,...2i)
        pos_idx = Math.arange(P).astype("i")
        dem_idx = Math.arange(0, D, 2).astype("i")
        # 计算1/(10000^(2i/d))，数学上相当于e^(-2i/d*log(10000))
        dem = Math.exp(-dem_idx * Math.log(100000) / D, dtype="f")
        # 计算pos/(10000^(2i/d))
        dem = np.broadcast_to(np.reshape(dem, (1, -1)), (P, D // 2))
        dem = (pos_idx * dem.T).T
        # 创建一个和输入一样的位置编码张量
        encode = Math.zeros((P, D), dtype="f")
        # 偶数列使用正弦，奇数列使用余弦
        encode[:, ::2] = Math.sin(dem, dtype="f")
        # 选择偶数列（实际上是索引为奇数的列）
        encode[:, 1::2] = Math.cos(dem, dtype="f")
        return encode
# ========================================================================================
class AttentionUnit(SimpleUnit):
    def __init__(self):
        self.attention_weight = None
        # 用于step1：计算评分
        self.broadcast = BroadcastUnit(axis=1)
        self.multiply = MultiplyUnit()
        self.sum = SumUnit(axis=2)
        self.softmax = SoftmaxUnit()
        # 用于step2：加权求和
        self.broadcast2 = BroadcastUnit(axis=2)
        self.multiply2 = MultiplyUnit()
        self.sum2 = SumUnit(axis=1)
    def forward(self, x):
        hs, q = x
        # ---------------------------------------------------------------------------------
        # PART1：根据某个查询向量q，对指定hs的每个h，按内积方式求相似程度，进而得到对每个h的重要程度得分a
        # ---------------------------------------------------------------------------------
        qr = self.broadcast.forward((q, hs.shape))
        # hs的每一个h'和传递的h求内积，得到的s就是hs每一个h'与传递h的内积相似度得分
        t = self.multiply.forward((hs, qr))
        s = self.sum.forward(t)
        # 过softmax归一化，得到的就是每一个h的重要程度
        a = self.softmax.forward(s)
        self.attention_weight = a
        # ---------------------------------------------------------------------------------
        # PART2：根据重要度得分a，以加权求和的方式组合hs中的每一个h'，重要度越高的h'占比越大
        # ---------------------------------------------------------------------------------
        # 把a扩展到H个宽度
        ar = self.broadcast2.forward((a, hs.shape))
        # 根据重要程度，将hs中的h加和为c，其中越重要的h占比越大
        t = self.multiply2.forward((hs, ar))
        c = self.sum2.forward(t)
        self.hs, self.qr, self.ar = hs, qr, ar
        # 将上下文向量返回
        return c
    def backward(self, dc):
        dt = self.sum2.backward(dc)
        dhs1, dar = self.multiply2.backward(dt)
        da = self.broadcast2.backward(dar)
        # 两个步骤反向传播的分界线
        ds = self.softmax.backward(da)
        dt = self.sum.backward(ds)
        dhs2, dqr = self.multiply.backward(dt)
        dq = self.broadcast.backward(dqr)
        dhs = dhs1 + dhs2
        return dhs, dq
# ========================================================================================
class TimeAttentionUnit(SimpleUnit):
    def __init__(self):
        self.layers = None
        self.attention_weights = None
    def forward(self, x):
        hs, hd = x
        batch_size, seq_len, feature_count = hd.shape
        out = np.empty_like(hd)
        self.layers = []
        self.attention_weights = []
        for t in range(seq_len):
            layer = AttentionUnit()
            out[:, t, :] = layer.forward((hs, hd[:, t, :]))
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        return out
    def backward(self, dout):
        batch_size, seq_len, feature_count = dout.shape
        dhs = 0
        dhd = np.empty_like(dout)
        for t in range(seq_len):
            dhs_, dh = self.layers[t].backward(dout[:, t, :])
            dhs += dhs_
            dhd[:, t, :] = dh
        return dhs, dhd
# ========================================================================================
class KeyValueAttentionUnit(ParameterUnit):
    def __init__(self, k_size, v_size, mask=False):
        super().__init__()
        self.k_size, self.v_size, self.mask = k_size, v_size, mask
        self.softmax = SoftmaxUnit()
        self.x_shape, self.h_shape = None, None
        self.cache = None
    def init_params(self, D, TX, TH):
        Wq = (Math.randn(D, self.k_size) / Math.sqrt(TX)).astype(np.float32)
        Wk = (Math.randn(D, self.k_size) / Math.sqrt(TH)).astype(np.float32)
        Wv = (Math.randn(D, self.v_size) / Math.sqrt(TH)).astype(np.float32)
        self.params = {"Wq": Wq, "Wk": Wk, "Wv": Wv}
        self.grads = {"Wq": None, "Wk": None, "Wv": None}
    def forward(self, x):
        # 获取输入的X和H，以及它们的形状
        X, H = x
        N, TX, D = X.shape
        N, TH, D = H.shape
        # 第一次传播需要初始化参数
        if self.x_shape is None:
            self.init_params(D, TX, TH)
        # 在后续的形状中：X=(N TX D) H=(N TH D) Wqk=(D K) Wv=(D V) Q=(N TX K) K=(N TH K) V=(N TH V)
        if self.x_shape != X.shape or self.h_shape != H.shape:
            self.x_shape = X.shape
            self.h_shape = H.shape
        # 开始正向传播
        Wq, Wk, Wv = self.params['Wq'], self.params['Wk'], self.params['Wv']
        Q, K, V = Math.dot(X, Wq), Math.dot(H, Wk), Math.dot(H, Wv)
        # 在后续的形状中：Q=(N|TX K) K=(N|TH K) V=(N|TH V) S=(N|TX N|TH) A=(N|TX N|TH) Y=(N|TX V)
        Q = Q.reshape(-1, self.k_size); K = K.reshape(-1, self.k_size); V = V.reshape(-1, self.v_size)
        S = Math.dot(Q, K.T)
        # 如果需要掩码则进行掩码处理
        if self.mask is True:
            M = np.triu(Math.ones(S.shape), k=1)
            M = np.where(M == 1, -np.inf, 0)
            S = S + M
        A = self.softmax.forward(S / Math.sqrt(self.k_size))
        Y = Math.dot(A, V) ; Y = Y.reshape(N, TX, self.v_size)
        self.cache = X, H, Q, K, V, A
        return Y
    def backward(self, dY):
        # 取出所有参数
        N, TX, D = self.x_shape
        N, TH, D = self.h_shape
        X, H, Q, K, V, A = self.cache
        Wq, Wk, Wv = self.params['Wq'], self.params['Wk'], self.params['Wv']
        # 反向传播到QKV线性计算之前
        dY = dY.reshape(N * TX, self.v_size)
        dA = Math.dot(dY, V.T); dV = Math.dot(A.T, dY)
        dS = self.softmax.backward(dA) / Math.sqrt(self.k_size)
        dQ = Math.dot(dS, K); dK = Math.dot(Q.T, dS).T
        # QKV线性计算的反向传播
        dXq = Math.dot(dQ, Wq.T); dXk = Math.dot(dK, Wk.T); dXv = Math.dot(dV, Wv.T)
        X = X.reshape(N * TX, -1); H = H.reshape(N * TH, -1)
        dWq = Math.dot(X.T, dQ); dWk = Math.dot(H.T, dK); dWv = Math.dot(H.T, dV)
        self.grads = {"Wq": dWq, "Wk": dWk, "Wv": dWv}
        return dXq.reshape(N, TX, D), dXk.reshape(N, TH, D) + dXv.reshape(N, TH, D)
# ========================================================================================
class SelfAttentionUnit(KeyValueAttentionUnit):
    def __init__(self, k_size, v_size, mask=False):
        super().__init__(k_size, v_size, mask)
    def forward(self, x):
        return super().forward((x, x))
    def backward(self, dy):
        dx1, dx2 = super().backward(dy)
        return dx1 + dx2
# ========================================================================================
class CrossAttentionUnit(KeyValueAttentionUnit):
    def __init__(self, k_size, v_size):
        super().__init__(k_size, v_size)
        self.h, self.dh = None, None
    def set_h(self, h):
        self.h = h
    def get_dh(self):
        return self.dh
    def forward(self, x):
        return super().forward((x, self.h))
    def backward(self, dy):
        dx, self.dh = super().backward(dy)
        return dx
# ========================================================================================
class BatchNormalizationUnit(SimpleUnit):
    # gamma beta设置为上一层节点数
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, running_mean=None, running_var=None):
        self.train = False
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    def forward(self, x):
        self.input_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = Math.zeros(D)
            self.running_var = Math.zeros(D)
        if self.train is True:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = Math.sqrt(var + 10e-7)
            xn = xc / std
            self.xc = xc
            self.xn = xn
            self.std = std
            self.batch_size = x.shape[0]
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((Math.sqrt(self.running_var + 10e-7)))
        y = self.gamma * xn + self.beta
        return y.reshape(*self.input_shape)
    def backward(self, dy):
        dy = dy.reshape(dy.shape[0], -1)
        dbeta = dy.sum(axis=0)
        dgamma = Math.sum(self.xn * dy, axis=0)
        dxn = self.gamma * dy
        dxc = dxn / self.std
        dstd = -Math.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = Math.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx.reshape(*self.input_shape)
# ========================================================================================
class LayerNormalizationUnit(SimpleUnit):
    def __init__(self, gamma=1.0, beta=0.0):
        # TODO LayerNormalizationUnit测试的不对，不确定是否有问题
        # TODO 另外gamma和beta应该是自学习的，因此需要放到参数里面
        self.gamma, self.beta = gamma, beta
    def forward(self, x):
        # 查看形状看特征维度
        shape = x.shape
        dem = x.shape[len(shape) - 1]
        # 变形到到每个批每个样本的维度
        x = x.reshape(-1, dem)
        mean = np.mean(x, axis=1)
        var = np.var(x, axis=1)
        std = np.sqrt(var + 1e-8)
        h = ((x.T - mean) / std).T
        y = self.gamma * h + self.beta
        # 记录标准差并恢复形状
        self.std, self.h = std, h
        y = y.reshape(shape)
        return super().forward(y)
    def backward(self, dy):
        # 查看形状看特征维度
        shape = dy.shape
        dem = dy.shape[len(shape) - 1]
        # 变形到到每个批每个样本的维度
        dy = dy.reshape(-1, dem)
        # 依次计算dbeta和dgamma
        dbeta = dy.sum(axis=0)
        dgamma = Math.sum(self.h * dy, axis=0)
        # 接着计算dh、dx
        dh = dy
        dx = (dh.T / self.std).T
        dx = dx.reshape(shape)
        return super().backward(dx)
# ========================================================================================
class MeanSquaredErrorUnit(OutputUnit):
    def __init__(self):
        self.y, self.t = None, None
    def forward(self, x):
        self.y = x
        return self.y
    def loss(self, t):
        return LossFunction.mean_squared_error(self.y, self.t)
    def backward(self, dy=1):
        d = self.y - self.t
        dx = dy * d * (2.0 / len(d))
        return dx
# ========================================================================================
class SoftmaxCrossEntropyOutputUnit(OutputUnit, SoftmaxUnit):
    def __init__(self):
        super().__init__()
        self.y, self.t = None, None
    # override super class method, just store latest y
    def forward(self, x):
        y = self.y = SoftmaxUnit().forward(x)
        return y
    def loss(self, t):
        # If the target is a tag instead of a one-hot vector, it needs to be restored accordingly.
        if t.ndim == 1:
            # self.latest_y.shape[1] is output dimension
            self.t = Math.eye(self.y.shape[1], dtype=np.uint8)[t]
        else:
            self.t = t
        return LossFunction.cross_entropy_error(self.y, self.t)
    # override super class method
    def backward(self, dy=1):
        # the sample count is the number of rows in the matrix
        count = self.y.shape[0]
        dx = dy * (self.y - self.t) / count
        return dx
# ========================================================================================
# Network Layer Definitions
# ========================================================================================
class FCLayer(Component):
    # sigma is activation function type
    def __init__(self, input_dimension, output_dimension, sigma, weight_init_std=0.01):
        super().__init__()
        # create linear unit
        self.add_unit(AffineUnit(input_dimension, output_dimension, weight_init_std))
        # create nonlinear unit
        self.add_unit(self.create_nonlinear(sigma))
    @staticmethod
    def create_nonlinear(sigma):
        # create nonlinear unit
        if sigma == 'identity':
            return IdentityUnit()
        elif sigma == 'relu':
            return ReluUnit()
        elif sigma == 'sigmoid':
            return SigmoidUnit()
        elif sigma == 'gelu':
            return GeluUnit()
        else:
            raise Exception('Unsupported sigma type of FCLayer class.')
# ========================================================================================
class ConvolutionLayer(Component):
    def __init__(self, filter_count, channel, filter_width, filter_height, stride, pad, weight_init_std=0.01):
        super().__init__()
        self.add_unit(ConvolutionUnit(filter_count, channel, filter_width, filter_height, stride, pad, weight_init_std))
        self.add_unit(ReluUnit())
# ========================================================================================
# GoogleNet Inception Module
class InceptionLayer(Component):
    def __init__(self, channel, weight_init_std=0.01):
        super().__init__()
        self.c11 = ConvolutionUnit(128, channel, 1, 1, 1, 0, weight_init_std)
        self.c21 = ConvolutionUnit(64, channel, 1, 1, 1, 0, weight_init_std)
        self.c22 = ConvolutionUnit(192, 64, 3, 3, 1, 1, weight_init_std)
        self.c31 = ConvolutionUnit(64, channel, 1, 1, 1, 0, weight_init_std)
        self.c32 = ConvolutionUnit(96, 64, 5, 5, 1, 2, weight_init_std)
        self.c41 = MaxPoolUnit(3, 3, 1, 1)
        self.c42 = ConvolutionUnit(64, channel, 1, 1, 1, 0, weight_init_std)
    def forward(self, x):
        y11 = self.c11.forward(x)
        y21 = self.c21.forward(x)
        y22 = self.c22.forward(y21)
        y31 = self.c31.forward(x)
        y32 = self.c32.forward(y31)
        y41 = self.c41.forward(x)
        y42 = self.c42.forward(y41)
        # 深度拼接
        y = np.concatenate((y11, y22, y32, y42), axis=1)
        return y
    def backward(self, dy):
        dy42 = self.c42.backward(dy[:, 128 + 192 + 96: 480, :, :])
        dy41 = self.c41.backward(dy42)
        dy32 = self.c32.backward(dy[:, 128 + 192: 128 + 192 + 96, :, :])
        dy31 = self.c31.backward(dy32)
        dy22 = self.c22.backward(dy[:, 0 + 128: 128+192, :, :])
        dy21 = self.c21.backward(dy22)
        dy11 = self.c11.backward(dy[:, 0:128, :, :])
        return dy41 + dy31 + dy21 + dy11
# ========================================================================================
class ResidualLayer(Component):
    def forward(self, x):
        # shortcut就是x
        s = x
        # 通过父类所有单元的处理
        h = super().forward(x)
        # 相加的结果过非线性作为返回
        y = h + s
        return y
    def backward(self, dy):
        # 加法的梯度传递给各自分量
        dh, ds = dy, dy
        # 通过各单元的反向传播
        dx = super().backward(dh)
        # 梯度整合
        return dx + ds
# ========================================================================================
class ResidualConvolutionLayer(ResidualLayer):
    def __init__(self, input_channel, filter_channels: list, filter_sizes: list):
        super().__init__()
        # 最后一层必须和输入一致
        if filter_channels[len(filter_channels) - 1] != input_channel:
            raise Exception("Channel of last layer must same as input channel.")
        # 依次建立多个卷积层
        for i in range(len(filter_channels)):
            # 为了保持输入输出大小不变，要计算一下padding的大小
            pad = (filter_sizes[i] - 1) // 2
            # 建立卷积层
            self.add_unit(ConvolutionUnit(filter_channels[i], input_channel, filter_sizes[i], filter_sizes[i], stride=1, pad=pad))
            input_channel = filter_channels[i]
        self.add_unit(ReluUnit())
# ========================================================================================
class TimeEmbeddingLayer(Component):
    def __init__(self, vocab_size, wordvec_size):
        super().__init__()
        self.add_unit(ReshapeToTimeBatchUnit(), 'reshape')
        self.add_unit(EmbeddingUnit(vocab_size, wordvec_size), 'embed')
        self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape']))
# ========================================================================================
class Transformer(Network):
    class MultiHeadSelfAttention(SimpleUnit):
        def __init__(self, head_num, k_size, v_size, mask):
            super().__init__()
            self.heads = []
            self.head_num = head_num
            self.k_size, self.v_size = k_size, v_size
            self.x_shape = None
            for i in range(head_num):
                self.heads.append(SelfAttentionUnit(k_size, v_size, mask))
            self.linear = None
        def forward(self, x):
            batch_size, seq_len, feature_count = x.shape
            # 建立最后的线性层
            if self.x_shape is None:
                self.linear = LinearUnit(self.head_num * self.v_size * seq_len, seq_len * feature_count)
            # 缓存输入形状
            if self.x_shape != x.shape:
                self.x_shape = x.shape
            # 依次处理每一个头，得到结果
            zs = []
            for head in self.heads:
                z = head.forward(x)
                z = z.reshape(batch_size, -1)
                zs.append(z)
            h = np.concatenate(tuple(zs), axis=1)
            y = self.linear.forward(h)
            y = y.reshape(batch_size, seq_len, feature_count)
            return y
        def backward(self, dys):
            dx = 0
            batch_size, seq_len, feature_count = self.x_shape
            # 先计算到线性层的导数dh
            dys = dys.reshape(batch_size, seq_len * feature_count)
            dh = self.linear.backward(dys)
            # 计算一下拆分后的形状
            batch_size, multi_size = dh.shape
            single_size = multi_size // self.head_num
            # 因为线性层的输入h是拼接的，因此这里必须依次拆分
            for i in range(self.head_num):
                dz = dh[:, i * single_size: (i + 1) * single_size]
                dz = dz.reshape(batch_size, seq_len, self.v_size)
                dx += self.heads[i].backward(dz)
            return dx
    class ResidualMultiHeadSelfAttention(ResidualLayer):
        def __init__(self, head_num, k_size, v_size, mask):
            super().__init__()
            self.add_unit(Transformer.MultiHeadSelfAttention(head_num, k_size, v_size, mask))
    class ResidualFeedForwardLayer(ResidualLayer):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.add_unit(ReshapeToTimeBatchUnit(), 'reshape')
            self.add_unit(FCLayer(input_size, hidden_size, 'gelu'))
            self.add_unit(DropoutUnit(0.1))
            self.add_unit(FCLayer(hidden_size, input_size, 'identity'))
            self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape']))
    class MultiHeadCrossAttention(SimpleUnit):
        def __init__(self, head_num, k_size, v_size):
            super().__init__()
            self.heads = []
            self.head_num = head_num
            self.k_size = k_size
            self.v_size = v_size
            self.x_shape = None
            for i in range(head_num):
                self.heads.append(CrossAttentionUnit(k_size, v_size))
            self.linear = None
        def set_h(self, h):
            for head in self.heads:
                head.set_h(h)
        def get_dh(self):
            dh = 0
            for head in self.heads:
                dh += head.get_dh()
            return dh
        def forward(self, x):
            batch_size, seq_len, feature_count = x.shape
            # 建立最后的线性层
            if self.x_shape is None:
                self.linear = LinearUnit(self.head_num * self.v_size * seq_len, seq_len * feature_count)
            # 缓存输入形状
            if self.x_shape != x.shape:
                self.x_shape = x.shape
            # 依次处理每一个头，得到结果
            zs = []
            for head in self.heads:
                z = head.forward(x)
                z = z.reshape(batch_size, -1)
                zs.append(z)
            h = np.concatenate(tuple(zs), axis=1)
            y = self.linear.forward(h)
            y = y.reshape(batch_size, seq_len, feature_count)
            return y
        def backward(self, dys):
            dx = 0
            batch_size, seq_len, feature_count = self.x_shape
            # 先计算到线性层的导数dh
            dys = dys.reshape(batch_size, seq_len * feature_count)
            dh = self.linear.backward(dys)
            # 计算一下拆分后的形状
            batch_size, multi_size = dh.shape
            single_size = multi_size // self.head_num
            # 因为线性层的输入h是拼接的，因此这里必须依次拆分
            for i in range(self.head_num):
                dz = dh[:, i * single_size: (i + 1) * single_size]
                dz = dz.reshape(batch_size, seq_len, self.v_size)
                dx += self.heads[i].backward(dz)
            return dx
    class ResidualMultiHeadCrossAttention(ResidualLayer):
        def __init__(self, head_num, k_size, v_size):
            super().__init__()
            self.add_unit(Transformer.MultiHeadCrossAttention(head_num, k_size, v_size), 'mhca')
    class EncoderModule(Component):
        def __init__(self, wordvec_size, head_num, k_size, v_size, h_size):
            super().__init__()
            self.add_unit(Transformer.ResidualMultiHeadSelfAttention(head_num, k_size, v_size, mask=False))
            self.add_unit(LayerNormalizationUnit())
            self.add_unit(Transformer.ResidualFeedForwardLayer(wordvec_size, h_size))
            self.add_unit(LayerNormalizationUnit())
    class DecoderModule(Component):
        def __init__(self, wordvec_size, head_num, k_size, v_size, h_size):
            super().__init__()
            self.add_unit(Transformer.ResidualMultiHeadSelfAttention(head_num, k_size, v_size, mask=True))
            self.add_unit(LayerNormalizationUnit())
            self.add_unit(Transformer.ResidualMultiHeadCrossAttention(head_num, k_size, v_size), 'rmhca')
            self.add_unit(LayerNormalizationUnit())
            self.add_unit(Transformer.ResidualFeedForwardLayer(wordvec_size, h_size))
            self.add_unit(LayerNormalizationUnit())
        def set_kv(self, h):
            self.units['rmhca'].units['mhca'].set_h(h)
        def get_dkv(self):
            return self.units['rmhca'].units['mhca'].get_dh()
    class Encoder(Component):
        def __init__(self, vocab_size, wordvec_size, module_num, head_num, k_size, v_size, h_size):
            super().__init__()
            self.add_unit(TimeEmbeddingLayer(vocab_size, wordvec_size), 'embed')
            self.add_unit(DropoutUnit(0.1))
            self.add_unit(PositionEncodingUnit())
            for i in range(module_num):
                self.add_unit(Transformer.EncoderModule(wordvec_size, head_num, k_size, v_size, h_size))
    class Decoder(Component):
        def __init__(self, vocab_size, wordvec_size, module_num, head_num, k_size, v_size, h_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.module_num = module_num
            self.add_unit(TimeEmbeddingLayer(vocab_size, wordvec_size), 'embed')
            self.add_unit(DropoutUnit(0.1))
            self.add_unit(PositionEncodingUnit())
            for i in range(module_num):
                self.add_unit(Transformer.DecoderModule(wordvec_size, head_num, k_size, v_size, h_size), 'module' + str(i))
            self.add_unit(ReshapeToTimeBatchUnit(), 'reshape')
            self.add_unit(LinearUnit(wordvec_size, vocab_size))
            self.add_unit(SoftmaxCrossEntropyOutputUnit(), 'softmax')
            self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape']))
        def loss(self, t):
            return self.units['softmax'].loss(t)
        def forward(self, x):
            enc_y, ts = x
            for i in range(self.module_num):
                self.units['module' + str(i)].set_kv(enc_y)
            return super().forward(ts)
        def backward(self, dy):
            dx = super().backward(dy)
            d_enc_y = 0
            for i in range(self.module_num):
                d_enc_y += self.units['module' + str(i)].get_dkv()
            return d_enc_y, dx
        def generate(self, enc_y, sample_size, sos, eos):
            # 存放中间变量(默认从sos开始)
            sampled = Math.zeros((len(enc_y), sample_size, self.vocab_size), dtype=np.int32)
            sampled[:, 0, :] = sos
            # 存放结果
            results = Math.empty((len(enc_y), sample_size, self.vocab_size), dtype=np.int32)
            # 依次生成下一个字符
            for i in range(sample_size):
                dec_y = self.forward((enc_y, sampled))
                # 只取当前时刻的结果(batch_size, vocab_size)
                dec_y = dec_y[:, i, :]
                # 创建一个与dec_y形状相同的全零数组，然后argmax变成one-hot
                sample = Math.zeros_like(dec_y, dtype=np.int32)
                sample[:, dec_y.argmax(axis=1)] = 1
                # 设置采样和最终结果
                if i < sample_size - 1:
                    sampled[:, i + 1, :] = sample
                results[:, i, :] = sample
            return results
    def __init__(self, sampled_size, sos, eos, vocab_size, wordvec_size, module_num, head_num, k_size, v_size, h_size):
        super().__init__()
        self.sampled_size, self.sos, self.eos = sampled_size, sos, eos
        self.add_unit(Transformer.Encoder(vocab_size, wordvec_size, module_num, head_num, k_size, v_size, h_size), 'encoder')
        self.add_unit(Transformer.Decoder(vocab_size, wordvec_size, module_num, head_num, k_size, v_size, h_size), 'decoder')
        # 权重共享
        self.units['decoder'].units['embed'].units['embed'].params['weight'] = self.units['encoder'].units['embed'].units['embed'].params['weight']
    def loss(self, xs, ts):
        self.ts = ts
        self.forward(xs)
        # 计算损失（bos不能到损失里）
        ls = self.ts[:, 1:]
        ls = ls.reshape(-1, ls.shape[2])
        return self.units['decoder'].loss(ls)
    def forward(self, xs):
        enc_y = self.units['encoder'].forward(xs)
        # 训练和预测的逻辑不同，训练时解码器的输入包括标记，生成时只用解码器上一时刻的输出
        if self.train is True:
            return self.units['decoder'].forward((enc_y, self.ts[:, :-1, :]))
        else:
            return self.units['decoder'].generate(enc_y, self.sampled_size, self.sos, self.eos)
    def backward(self, dys):
        d_enc_y, dts = self.units['decoder'].backward(dys)
        return self.units['encoder'].backward(d_enc_y)
    def accuracy(self, samples, labels):
        # 输出的分值最大的作为预测
        outputs = self.forward(samples)
        outputs = outputs.reshape(-1, labels.shape[2]).argmax(axis=1)
        # 变形标签，然后是1的作为标签
        labels = labels[:, 1:]
        labels = labels.reshape(-1, labels.shape[2])
        labels = labels.argmax(axis=1)
        # 统计有多少一样的
        return Math.sum(outputs == labels) / len(outputs)
    def generate(self, xs):
        enc_y = self.units['encoder'].forward(xs)
        return self.units['decoder'].generate(enc_y, self.sampled_size, self.sos, self.eos)
# ========================================================================================
class SoftmaxOutputLayer(OutputUnit, Component):
    def __init__(self, input_dimension: int, output_dimension: int, weight_init_std=0.01):
        super().__init__()
        self.add_unit(AffineUnit(input_dimension, output_dimension, weight_init_std), 'linear')
        self.add_unit(SoftmaxCrossEntropyOutputUnit(), 'nonlinear')
    def loss(self, t):
        return self.units['nonlinear'].loss(t)
# ========================================================================================
# Basic Neural Network Definitions
# ========================================================================================
class SimpleNetwork(Network):
    def accuracy(self, samples, labels) -> float:
        return self.classify_accuracy(samples, labels)
    def gradient(self, samples, labels) -> None:
        for module in self.units.values():
            module.gradient(lambda w: self.loss(samples, labels))
    def classify_accuracy(self, samples, labels, batch=500):
        count = 0
        for i in range(0, len(samples), batch):
            batch_input = samples[i: i + batch]
            batch_output = self.forward(batch_input)
            batch_argmax = Math.argmax(batch_output, axis=1)
            batch_labels = labels[i: i + batch]
            # covert one-hot to scalar
            if batch_labels.ndim > 1:
                batch_labels = batch_labels.argmax(axis=1)
            count += Math.sum(batch_argmax == batch_labels)
        return float(count) / len(samples)
# ========================================================================================
class BPNetwork(SimpleNetwork):
    def add_layers(self, layers_size):
        layer_count = len(layers_size) - 1
        for i in range(0, layer_count):
            if i < layer_count - 1:
                self.add_unit(FCLayer(layers_size[i], layers_size[i + 1], 'sigmoid', 1 / Math.sqrt(layers_size[i])))
            else:
                self.add_unit(SoftmaxOutputLayer(layers_size[i], layers_size[i + 1], 1 / Math.sqrt(layers_size[i])))
# ========================================================================================
class CNN(SimpleNetwork):
    def __init__(self, input_shape, filter_count, filter_size, filter_pad, filter_stride, hidden_size, output_size):
        super().__init__()
        input_size = input_shape[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_count * (conv_output_size / 2) * (conv_output_size / 2))
        self.add_unit(ConvolutionLayer(filter_count, input_shape[0], filter_size, filter_size, filter_stride, filter_pad))
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        self.add_unit(FCLayer(pool_output_size, hidden_size, "relu"))
        self.add_unit(SoftmaxOutputLayer(hidden_size, output_size))
# ========================================================================================
class VGG16(SimpleNetwork):
    def __init__(self, input_shape):
        super().__init__()
        # CS1
        self.add_unit(ConvolutionLayer(64, input_shape[0], 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(64, 64, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ReluUnit())
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        # CS2
        self.add_unit(ConvolutionLayer(128, 64, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(128, 128, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ReluUnit())
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        # CS3
        self.add_unit(ConvolutionLayer(256, 128, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(256, 256, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(256, 256, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ReluUnit())
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        # CS4
        self.add_unit(ConvolutionLayer(512, 256, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(512, 512, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(512, 512, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ReluUnit())
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        # CS5
        self.add_unit(ConvolutionLayer(512, 512, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(512, 512, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ConvolutionLayer(512, 512, 3, 3, 1, 1, 'fan_out'))
        self.add_unit(ReluUnit())
        self.add_unit(MaxPoolUnit(2, 2, 2, 0))
        # FCC
        self.add_unit(FCLayer(512 * 7 * 7, 4096, "relu", 0.01))
        self.add_unit(FCLayer(4096, 4096, "relu", 0.01))
        self.add_unit(SoftmaxOutputLayer(4096, 1000, 0.01))
# ========================================================================================
class CBOW(SimpleNetwork):
    def __init__(self, vocab_size, hidden_size):
        super(CBOW, self).__init__()
        self.add_unit(EmbeddingBagUnit(vocab_size, hidden_size), 'embedding')
        self.add_unit(MatMultiplyUnit(hidden_size, vocab_size), 'matmul')
        self.add_unit(SoftmaxCrossEntropyOutputUnit(), 'softmax')
# ========================================================================================
class RNNLM(SimpleNetwork):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        # 改变形状是为了一次处理所有时间
        self.add_unit(ReshapeToTimeBatchUnit(), 'reshape1')
        self.add_unit(EmbeddingUnit(vocab_size, wordvec_size))
        self.add_unit(DropoutUnit(0.4))
        self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape1']), 'dereshape1')
        self.add_unit(TimeRNNUnit(wordvec_size, hidden_size, "lstm", stateful=True), 'rnn')
        # 改变形状是为了一次处理所有时间
        self.add_unit(ReshapeToTimeBatchUnit(), 'reshape2')
        self.add_unit(AffineUnit(hidden_size, vocab_size, 1.0 / Math.sqrt(hidden_size)))
        self.add_unit(SoftmaxCrossEntropyOutputUnit(), 'softmax')
        self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape2']), 'dereshape2')
        self.x_shape = None
    def reset_rnn(self):
        self.units['rnn'].reset()
    def forward(self, x):
        # 批大小改变则h一定形状不匹配了
        if self.x_shape != x.shape:
            self.reset_rnn()
        self.x_shape = x.shape
        return super().forward(x)
    def loss(self, samples, labels):
        self.forward(samples)
        batch_size, time_length, feature_count = labels.shape
        labels = labels.reshape(batch_size * time_length, -1)
        # 获取softmax层的的损失
        return self.units['softmax'].loss(labels)
    def accuracy(self, samples, labels):
        batch_size, time_length, feature_count = samples.shape
        loss = self.loss(samples, labels)
        # 困惑度计算
        ppl = Math.exp(loss)
        # 换算准确度
        return 1 - ppl / (feature_count - 1)
    def generate(self, start_id, stop_ids=None, sample_size=100):
        # 记录采样得到的单词序列，开始的单词作为第一个词
        word_ids = [start_id]
        # x是当前的单词
        x = start_id
        # 依次采样sample_size个词
        while len(word_ids) < sample_size:
            # 由于x是个id，因此需要转换成one-hot输入
            sample = Math.zeros((1, 1, self.vocab_size), dtype=np.int32)
            sample[0, 0, x] = 1
            # x是向量，则输出应该是1*1*vocab_size
            score = self.forward(sample)
            # score已经在softmax之后了，因此扁平化直接作为概率分布即可
            p = score.flatten()
            # 根据概率采样下一个词
            sampled = Math.choice(len(p), size=1, p=p)
            # 若未指定停止词或当前词不是停止词，则继续生成
            if (stop_ids is None) or (sampled not in stop_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids
# ========================================================================================
class Seq2Seq(SimpleNetwork):
    class Encoder(Component):
        def __init__(self, vocab_size, wordvec_size, hidden_size):
            super().__init__()
            self.add_unit(TimeEmbeddingLayer(vocab_size, wordvec_size))
            self.add_unit(TimeRNNUnit(wordvec_size, hidden_size, "lstm", stateful=False))
    class Decoder(Component):
        def __init__(self, vocab_size, wordvec_size, hidden_size):
            super().__init__()
            # 改变形状是为了一次处理所有时间
            self.add_unit(TimeEmbeddingLayer(vocab_size, wordvec_size))
            self.add_unit(TimeRNNUnit(wordvec_size, hidden_size, "lstm", stateful=True), 'rnn')
            # 改变形状是为了一次处理所有时间
            self.add_unit(ReshapeToTimeBatchUnit(), 'reshape2')
            self.add_unit(AffineUnit(hidden_size, vocab_size, 1.0 / Math.sqrt(hidden_size)))
            self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape2']))
        def set_state(self, hs):
            self.hs = hs
            # 将最后一个隐藏状态作为解码器的初始状态
            self.units['rnn'].reset(h=hs[:, -1, :])
        def backward(self, dy):
            super().backward(dy)
            # 只有RNN的最后一个状态需要反向传播，但是形状要保持和编码器一致
            dhs = Math.zeros_like(self.hs)
            dhs[:, -1, :] = self.units['rnn'].get_dh()
            return dhs
        def generate(self, hs, start_char, sample_size):
            self.set_state(hs)
            # 存放所有的采样结果(sample_size+1是为了在第一元素存储start_char)
            sampled = Math.zeros((len(hs), sample_size + 1, len(start_char)), dtype=np.float32)
            sampled[:, 0, :] = start_char
            # 依次生成下一个字符
            for i in range(sample_size):
                # 将当前的字符改变形状(?, 1, ?)以适应时间序列
                x = Math.reshape(sampled[:, i, :], (len(hs), 1, -1))
                y = self.forward(x)
                # 采样得到的结果也需要改变形状(?, 1, ?)以适应时间序列
                y = Math.reshape(y, (len(hs), -1))
                # 创建一个与原数组形状相同的全零数组，然后argmax变成one-hot
                sample = Math.zeros_like(y)
                sample[Math.arange(len(y)), y.argmax(axis=1)] = 1
                # 存放到下一状态
                sampled[:, i + 1, :] = sample
            # 除了起始字符都是生成的结果
            return sampled[:, 1:, :]
    def __init__(self, vocab_size, wordvec_size, hidden_size, sampled_size, hint=None):
        super().__init__()
        self.add_unit(Seq2Seq.Encoder(vocab_size, wordvec_size, hidden_size), 'encoder')
        self.add_unit(Seq2Seq.Decoder(vocab_size, wordvec_size, hidden_size), 'decoder')
        self.add_unit(SoftmaxCrossEntropyOutputUnit(), 'softmax')
        self.sampled_size = sampled_size
        self.hint = hint
    def loss(self, xs, ts):
        self.ts = ts
        self.forward(xs)
        # 计算损失（第一个下划线不能到损失里）
        ls = self.ts[:, 1:]
        ls = ls.reshape(-1, ls.shape[2])
        return self.units['softmax'].loss(ls)
    def forward(self, xs):
        batch_size, seq_len, feature_count = xs.shape
        # 先通过编码器，然后设置解码器的初始状态
        hs = self.units['encoder'].forward(xs)
        self.units['decoder'].set_state(hs)
        # 训练和预测的逻辑不同，训练时解码器的输入包括标记，生成时只用解码器上一时刻的输出
        if self.train is True:
            # 解码器输入的是第一个词到倒数第二个词
            y = self.units['decoder'].forward(self.ts[:, :-1, :])
        else:
            # 预测时通过generate方法依次生成
            y = self.units['decoder'].generate(hs, self.hint, self.sampled_size)
        # 变形后通过softmax
        y = y.reshape(-1, feature_count)
        return self.units['softmax'].forward(y)
    def accuracy(self, samples, labels):
        # 输出的分值最大的作为预测
        outputs = self.forward(samples)
        outputs = outputs.argmax(axis=1)
        # 变形标签，然后是1的作为标签
        labels = labels[:, 1:].reshape(-1, labels.shape[2])
        labels = labels.argmax(axis=1)
        # 统计有多少一样的
        return Math.sum(outputs == labels) / len(outputs)
    def generate(self, xs):
        hs = self.units['encoder'].forward(xs)
        return self.units['decoder'].generate(hs, self.hint, self.sampled_size)
# ========================================================================================
class AttentionSeq2Seq(Seq2Seq):
    class Decoder(Seq2Seq.Decoder):
        def __init__(self, vocab_size, wordvec_size, hidden_size):
            self.units = {}
            self.train = False
            self.add_unit(TimeEmbeddingLayer(vocab_size, wordvec_size), 'embed')
            self.add_unit(TimeRNNUnit(wordvec_size, hidden_size, "lstm", stateful=True), 'rnn')
            self.add_unit(TimeAttentionUnit(), 'attention')
            self.add_unit(ConcatenateUnit(axis=2), 'concatenate')
            self.add_unit(ReshapeToTimeBatchUnit(), 'reshape2')
            # 乘以二是为了peek时拼接向量
            self.add_unit(AffineUnit(2 * hidden_size, vocab_size, 1.0 / Math.sqrt(2 * hidden_size)), 'affine')
            self.add_unit(UnreshapeToTimeBatchUnit(self.units['reshape2']), 'deshape2')
        def forward(self, xs):
            xs = self.units['embed'].forward(xs)
            hd = self.units['rnn'].forward(xs)
            c = self.units['attention'].forward((self.hs, hd))
            # 把注意力输出的上下文和之前的hd拼接共同作为下一层输入
            chd = self.units['concatenate'].forward((c, hd))
            chd = self.units['reshape2'].forward(chd)
            score = self.units['affine'].forward(chd)
            score = self.units['deshape2'].forward(score)
            return score
        def backward(self, dscore):
            dscore = self.units['deshape2'].backward(dscore)
            dchd = self.units['affine'].backward(dscore)
            dchd = self.units['reshape2'].backward(dchd)
            dc, dhd2 = self.units['concatenate'].backward(dchd)
            dhs, dhd1 = self.units['attention'].backward(dc)
            dhd = dhd2 + dhd1
            dxs = self.units['rnn'].backward(dhd)
            dh = self.units['rnn'].get_dh()
            dhs[:, -1] += dh
            dxs = self.units['embed'].backward(dxs)
            return dhs
    def __init__(self, vocab_size, wordvec_size, hidden_size, sampled_size, hint=None):
        super().__init__(vocab_size, wordvec_size, hidden_size, sampled_size, hint)
        # 使用带有Attention的编码器
        self.units['decoder'] = AttentionSeq2Seq.Decoder(vocab_size, wordvec_size, hidden_size)
# ========================================================================================
# Optimizers
# ========================================================================================
class SGDOptimizer(Optimizer):
    def __init__(self, lr, max_grad=None):
        self.lr = lr
        self.max_grad = max_grad
    def update(self, params, grads):
        for i in range(len(params)):
            params_map = params[i]
            grads_map = grads[i]
            for key in params_map:
                if self.max_grad is not None:
                    Optimizer.clip_grads(grads_map[key], self.max_grad)
                params_map[key] -= self.lr * grads_map[key]
# ========================================================================================
class MomentumOptimizer(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        # 首次初始化v
        if self.v is None:
            self.v = []
            for i in range(len(params)):
                map = {}
                for key in params[i]:
                    map[key] = Math.zeros_like(params[i][key])
                self.v.append(map)
        # 下降
        for i in range(len(params)):
            params_map = params[i]
            grads_map = grads[i]
            for key in params_map:
                self.v[i][key] = self.momentum * self.v[i][key] - self.lr * grads_map[key]
                params_map[key] += self.v[i][key]
# ========================================================================================
class AdamOptimizer(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self, params, grads):
        # 首次初始化m
        if self.m is None:
            self.m, self.v = [], []
            for i in range(len(params)):
                mmap = {}
                vmap = {}
                for key in params[i]:
                    mmap[key] = Math.zeros_like(params[i][key])
                    vmap[key] = Math.zeros_like(params[i][key])
                self.m.append(mmap)
                self.v.append(vmap)
        # 增加迭代次数，计算当前的学习率
        self.iter += 1
        lr_t = self.lr * Math.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for i in range(len(params)):
            for key in params[i]:
                self.m[i][key] += (1 - self.beta1) * (grads[i][key] - self.m[i][key])
                self.v[i][key] += (1 - self.beta2) * (grads[i][key] ** 2 - self.v[i][key])
                params[i][key] -= lr_t * self.m[i][key] / (Math.sqrt(self.v[i][key]) + 1e-7)
# ========================================================================================
# DataSet and DataLoader
# ========================================================================================
class DataSet:
    def __init__(self):
        self.samples = None
        self.labels = None
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, item):
        if self.labels is None:
            return self.samples[item]
        else:
            return self.samples[item], self.labels[item]
# ========================================================================================
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_count = len(dataset)
        self.iter_count = max(round(self.sample_count / batch_size), 1)
        self.iter_num = 0
        self.reset()
    @property
    def samples(self):
        return self.dataset.samples
    @property
    def labels(self):
        return self.dataset.labels
    def reset(self):
        self.iter_num = 0
        if self.shuffle:
            self.indexes = Math.permutation(self.sample_count)
        else:
            self.indexes = Math.arange(self.sample_count)
    def __next__(self):
        if self.iter_num >= self.iter_count:
            self.reset()
            raise StopIteration
        iter_num, batch_size = self.iter_num, self.batch_size
        batch_index = self.indexes[iter_num * batch_size: (iter_num + 1) * batch_size]
        batch_data = [self.dataset[i] for i in batch_index]
        batch_samples = Math.array([data[0] for data in batch_data])
        batch_labels = Math.array([data[1] for data in batch_data])
        self.iter_num += 1
        return batch_samples, batch_labels
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        return self
# ========================================================================================
# Trainer
# ========================================================================================
class Trainer:
    def __init__(self, network: Network, train_data_loader, optimizer="sgd", lr=0.1):
        self.network = network
        self.data = train_data_loader
        # create optimizer
        if optimizer == "sgd":
            self.optimizer = SGDOptimizer(lr)
        elif optimizer == 'momentum':
            self.optimizer = MomentumOptimizer(lr)
        elif optimizer == "adam":
            self.optimizer = AdamOptimizer(lr)
    def train(self, test_x, test_y, max_iter=1, stop=0.01):
        train_loss_list = []
        test_loss_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        # while ... stop
        for j in range(max_iter):
            train_loss = self.step()
            # 计算损失和准确率
            data = self.data
            test_loss = self.network.loss(test_x, test_y)
            train_accuracy = self.network.accuracy(data.samples, data.labels)
            test_accuracy = self.network.accuracy(test_x, test_y)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            # 输出信息
            print("Epoch: %s, Train Loss: %.4f, Test Loss: %.4f, Train Accuracy: %.4f, Test Accuracy: %.4f" %
                  (j + 1, train_loss, test_loss, train_accuracy, test_accuracy))
            # 判断停止
            if train_loss < stop:
                break
        # 建立图表
        fig = plt.figure()
        # 画数据点
        plt.plot(train_loss_list, c="orange")
        plt.plot(test_loss_list, c="blue")
        plt.plot(train_accuracy_list, c="orange")
        plt.plot(test_accuracy_list, c="blue")
        return train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list
    def step(self):
        loss, count = 0, 0
        now_samples, total_samples = 0, len(self.data.samples)
        self.network.set_train(True)
        for samples, labels in self.data:
            count += 1
            now_samples += len(samples)
            loss += self.network.loss(samples, labels)
            self.network.backward(1)
            self.optimizer.update(self.network.parameters(), self.network.gradients())
            print("samples: %s/%s " % (now_samples, total_samples), end='\r')
        self.network.set_train(False)
        return loss / count
# ========================================================================================
# Framework Initialization
# ========================================================================================
# check python version
ver = sys.version_info
if ver.major <= 3 and ver.minor <= 7:
    raise Exception('unsupported python version')
# enabled warnings
Global.enable_warning()
# ========================================================================================