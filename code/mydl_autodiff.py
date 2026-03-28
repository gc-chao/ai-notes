# ========================================================================================
# 这部分自动微分代码是阅读斋藤康毅《深度学习入门2：自制框架》时整理的
# 目前这部分代码未在CODE.md中引用
# 主要内容：
# 1、实现了变量类。
# 2、定义了操作接口类，实现了基本运算，包括四则运算、三角函数、指数、矩阵乘法。
# 3、实现了基于动态计算图的自动微分机制。
# 4、实现了张量及基本的数学运算。
# ========================================================================================
import weakref
import numpy as np
# ========================================================================================
# Variable Class
# ========================================================================================
class Variable:
    # 构造方法，设置默认值
    def __init__(self, data):
        # 自动转换标量为Numpy数组
        self.data = data if not np.isscalar(data) else np.array(data)
        # 存储梯度，在反向传播之前是None值
        self.grad = None
        # 存储创建该变量的操作
        self.creator = None
        # 节点在计算图中的层次
        self.level = 0
    def get_data(self):
        return self.data
    def get_grad(self):
        return self.grad
    # 设置创建该变量的运算对象和运算层次
    def set_creator(self, creator, level):
        self.creator = creator
        self.level = level + 1
    # 变量的反向传播
    def backward(self):
        # 如果该变量没有梯度(因为可能是操作的最后一步)，则默认设置为1
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 创建一个运算对象的列表，默认将自己的创建者放入列表
        operations = [self.creator]
        # 不停的从列表中取出元素，直到它变为空
        while operations:
            operation = operations.pop()
            # 从操作中得到所有输出变量的上游梯度，形成列表(output加括号的原因是它是一个弱引用对象，要取到里面的引用对象)
            dys = [output().grad for output in operation.get_outputs()]
            # 调用操作的反向传播，使用星号解开列表，结果如果不是元祖则转换为元祖
            dxs = operation.backward(*dys)
            dxs = dxs if isinstance(dxs, tuple) else (dxs,)
            # 将正向传播的输入变量对象和反向传播的梯度值打组，对每一组进行梯度更新
            for x, dx in zip(operation.get_inputs(), dxs):
                # 当重复使用一个变量时，梯度需要累计的增加
                x.grad = dx if x.grad is None else x.grad + dx
                # 如果x仍有创建的操作对象，那么添加到列表，继续反向传播
                if x.creator is not None:
                    # 如果列表中已经有这个运算，就不用添加了
                    if operations.count(x.creator) == 0:
                        operations.append(x.creator)
                        # 根据运算的层次对运算列表进行排序，以便按层次反向传播
                        operations.sort(key=lambda x: x.level)
            # 上游变量中的梯度后续不再需要，清除以节省空间
            for output in operation.get_outputs():
                output().grad = None
    @property
    def shape(self):
        return self.data.shape
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return "Variable(\nNone\n)"
        return "Variable(\n" + str(self.data) + "\n)"
# ========================================================================================
# Variable Operation Base Class
# (Other operations such as addition or division can be derived from it)
# ========================================================================================
class Operation:
    def __init__(self):
        # 缓存操作的输入变量、输出变量、以及运算层次(层次用于反映运算的先后顺序)
        self.inputs = None
        self.outputs = None
        self.level = None
    # 默认的调用方法
    def __call__(self, *inputs: list) -> Variable:
        # 解析传递的输入
        inputs = [self.parse_input(input) for input in inputs]
        # 将当前输入变量的最大层次值作为该运算的层次
        self.level = max([input.level for input in inputs])
        # 将不定参数的每个Variable元素的data取出形成列表
        xs = [input.get_data() for input in inputs]
        # 正向传播，使用星号解开列表，返回值如果是标量则转换为元祖
        ys = self.forward(*xs)
        ys = ys if isinstance(ys, tuple) else (ys,)
        # 根据结果生成Variable对象数组形式的输出
        outputs = [Variable(y) for y in ys]
        # 所有的输出需要设置创建者为当前操作
        for output in outputs:
            output.set_creator(self, self.level)
        # 将输入和输出缓存起来，便于反向传播(输出没有采用self.outputs=outputs的原因是使用弱引用节省内存)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        # 如果只有一个元素，直接返回第一个即可
        return outputs if len(outputs) > 1 else outputs[0]
    # 获取输入变量列表
    def get_inputs(self):
        return self.inputs
    # 获取输出变量对象列表，列表的每一个元素是使用若引用包装的对象
    def get_outputs(self):
        return self.outputs
    # 分析数据，如果不是Variable对象，自动进行转换
    @staticmethod
    def parse_input(input):
        if not isinstance(input, Variable):
            input = Variable(input)
        return input
# ========================================================================================
# Simple Variable Operations (for overloaded operators)
# ========================================================================================
class Neg(Operation):
    def forward(self, x: np.array) -> np.array:
        return -x
    def backward(self, dy: np.array) -> np.array:
        return -dy
class Add(Operation):
    def forward(self, x1: np.array, x2: np.array) -> np.array:
        return x1 + x2
    def backward(self, dy: np.array) -> np.array:
        return dy, dy
class Sub(Operation):
    def forward(self, x1: np.array, x2: np.array) -> np.array:
        return x1 - x2
    def backward(self, dy: np.array) -> np.array:
        return dy, -dy
class Multiply(Operation):
    def forward(self, x1: np.array, x2: np.array) -> np.array:
        return x1 * x2
    def backward(self, dy: np.array) -> np.array:
        x1, x2 = self.inputs[0].data, self.inputs[1].data
        return dy * x2, dy * x1
class Div(Operation):
    def forward(self, x1: np.array, x2: np.array) -> np.array:
        return x1 / x2
    def backward(self, dy: np.array) -> np.array:
        x1, x2 = self.inputs[0].data, self.inputs[1].data
        return dy / x2, dy * (-x1 / x2 ** 2)
class Pow(Operation):
    def __init__(self, idx):
        self.idx = idx
    def forward(self, x: np.array) -> np.array:
        return x ** self.idx
    def backward(self, dy: np.array) -> np.array:
        x = self.inputs[0].data
        dx = self.idx * x ** (self.idx - 1) * dy
        return dx
# Operator overloading, defining Variable simple operations
Variable.__neg__ = lambda x: Neg()(x)
Variable.__add__ = lambda x1, x2: Add()(x1, x2)
Variable.__radd__ = lambda x1, x2: Add()(x1, x2)
Variable.__sub__ = lambda x1, x2: Sub()(x1, x2)
Variable.__rsub__ = lambda x1, x2: Sub()(x2, x1)
Variable.__mul__ = lambda x1, x2: Multiply()(x1, x2)
Variable.__rmul__ = lambda x1, x2: Multiply()(x1, x2)
Variable.__truediv__ = lambda x1, x2: Div()(x1, x2)
Variable.__rtruediv__ = lambda x1, x2: Div()(x2, x1)
Variable.__pow__ = lambda x, idx: Pow(idx)(x)
# ========================================================================================
# Variable Mathematical Operations (for common using)
# ========================================================================================
class Square(Operation):
    def forward(self, x: np.array) -> np.array:
        return x ** 2
    def backward(self, dy: np.array) -> np.array:
        x = self.inputs[0].data
        dx = 2 * x * dy
        return dx
class Sin(Operation):
    def forward(self, x: np.array) -> np.array:
        return np.sin(x)
    def backward(self, dy: np.array) -> np.array:
        x = self.inputs[0].data
        dx = np.cos(x) * dy
        return dx
class Cos(Operation):
    def forward(self, x: np.array) -> np.array:
        return np.cos(x)
    def backward(self, dy: np.array) -> np.array:
        x = self.inputs[0].data
        dx = -np.sin(x) * dy
        return dx
class Exp(Operation):
    def forward(self, x: np.array) -> np.array:
        return np.exp(x)
    def backward(self, dy: np.array) -> np.array:
        x = self.inputs[0].data
        dx = np.exp(x) * dy
        return dx
class Tanh(Operation):
    def forward(self, x: np.array) -> np.array:
        return np.tanh(x)
    def backward(self, dy: np.array) -> np.array:
        y = self.outputs[0]().data
        dx = (1 - y ** 2) * dy
        return dx
# ========================================================================================
# Variable Mathematical Operations (for shape changes and broadcasting)
# ========================================================================================
class Dot(Operation):
    def forward(self, x: np.array, y: np.array) -> np.array:
        return x.dot(y)
    def backward(self, dy: np.array) -> np.array:
        x, y = self.inputs
        dx = np.dot(dy, y.T)
        dy = np.dot(x.T, dy)
        return dx, dy
class Sum(Operation):
    def __init__(self, axis=None):
        self.axis = axis
    def forward(self, x: np.array) -> np.array:
        self.shape = x.shape
        return x.sum(axis=self.axis)
    def backward(self, dy: np.array) -> np.array:
        dx = np.broadcast_to(dy, self.shape)
        return dx
class BroadcastTo(Operation):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    def backward(self, dy):
        ndim = len(self.x_shape)
        lead = dy.ndim - ndim
        lead_axis = tuple(range(lead))
        axis = tuple([i + lead for i, sx in enumerate(self.x_shape) if sx == 1])
        dx = dy.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            dx = dx.squeeze(lead_axis)
        return dx
# ========================================================================================
# Variable External Operation Functions
# ========================================================================================
class ExternalOperation:
    @staticmethod
    def square(x):
        return Square()(x)
    @staticmethod
    def sin(x):
        return Sin()(x)
    @staticmethod
    def cos(x):
        return Cos()(x)
    @staticmethod
    def exp(x):
        return Exp()(x)
    @staticmethod
    def tanh(x):
        return Tanh()(x)
    @staticmethod
    def dot(x, y):
        return Dot()(x, y)
    @staticmethod
    def sum(x, axis):
        return Sum(axis)(x)
# ========================================================================================
# 使用示例
# ========================================================================================
x1 = Variable(2)
x2 = Variable(3) + 1
y = ExternalOperation.square(-x1 * x2) + ExternalOperation.square(x2)
y.backward()
print("y.data:", y.data)
print("y.grad:", y.grad)
print("x1.grad:", x1.grad)
print("x2.grad:", x2.grad)
print("x1:", x1)
# ========================================================================================