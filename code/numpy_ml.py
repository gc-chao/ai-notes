# ========================================================================================
# 这部分是使用numpy来编写经典的传统机器学习模型示例
# ========================================================================================
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# ========================================================================================
# 线性回归示例
# ========================================================================================
# 准备数据x, y
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 3, 5, 9, 11, 12, 13, 17])
# 将x扩展为(x, 1)的形式
X = []
for item in x:
    X.append([item, 1])
# 得到X矩阵
X = np.array(X).T
# 计算XX^T
val1 = np.dot(X, X.T)
# 计算(XX^T)^{-1}
val2 = np.linalg.inv(np.array([val1]))
# 计算Xy
val3 = np.dot(X, y)
# 计算w=(XX^T)^{-1}Xy
w = np.dot(val2, val3)
# w包含了k和b
k = w[0, 0]
b = w[0, 1]
# 创建图形
fig = plt.figure()
# 画数据点
plt.plot(x.tolist(), y.tolist(), 'o')
plt.plot([0, 8], [b, 8 * k + b])
# ========================================================================================
# 感知机示例
# ========================================================================================
# 准备数据集，包括样本和标签，其中两个正例一个反例
samples = [[3, 3], [4, 3], [1, 1]]
labels = [1, 1, -1]
# 将样本扩展为[x1, x2, ... , 1]的形式
for item in samples:
    item.append(1)
# 将数据集组织成numpy形式的训练集
train = []
for item in samples:
    train.append(np.array(item))
# 初始化参数和超参数
eta = 1.0
w = np.zeros(train[0].size)
# 梯度下降，默认迭代6次
for epoch in range(6):
    # 遍历数据集
    for i in range(len(train)):
        x = train[i]
        # 计算一个样本的分类结果y=sign(w^Tx)
        y = np.sign(np.dot(w.T, x))
        # 如果分类不一致，调整权重
        if y != labels[i]:
            # w = w + η * y * x
            w = w + eta * labels[i] * x
            # 输出每一次调整的w
            print(w)
# 计算完成的w
print(w)
# 得到k和b
k = -w[0] / w[1]
b = -w[2] / w[1]
# 创建图形
fig = plt.figure()
# 画数据点
plt.plot([sublist[0] for sublist in samples], [sublist[1] for sublist in samples], 'o')
plt.plot([0, 4], [b, 4 * k + b])
# ========================================================================================
# 逻辑回归示例
# ========================================================================================
# 定义Logistic函数
def sigma(w, x):
    return 1 / (1 + np.exp(np.dot(-w.T, x)))
# 定义预测函数
def perdit(w, x):
    p = sigma(w, x)
    if p > 0.5:
        return 1
    else:
        return 0
# 定义交叉熵损失
def loss(w, X, y):
    sum = 0.0
    for i in range(len(X)):
        positive = y[i] * np.log(sigma(w, X[i]))
        negative = (1 - y[i]) * np.log(1 - sigma(w, X[i]))
        sum += positive + negative
    return -sum / len(X)
# 定义梯度计算函数
def gradient(w, X, y):
     grad = np.zeros(len(w))
     # 分别对每个维度求梯度
     for i in range(len(w)):
          dw = w[i]
          # 利用差分求导
          w[i] = dw - 0.01
          loss1 = loss(w, X, y)
          w[i] = dw + 0.01
          loss2 = loss(w, X, y)
          w[i] = dw
          grad[i] = (loss2 - loss1) / (0.01 + 0.01)
     return grad
# 准备数据四个正例，四个反例
samples = [[5, 6], [1, 1], [3, 2], [3, 4], [2, -4], [4, -1], [5, -2], [1, -3]]
labels = [0, 0, 0, 0, 1, 1, 1, 1]
# 初始化参数和超参数以及画图缓存
params = np.zeros(len(samples[0]))
eta = 0.01
points = []
# 先计算初步损失
l = loss(params, samples, labels)
# 训练代码(梯度下降)
while l > 0.1:
    # 看看损失
    l = loss(params, samples, labels)
    points.append(l)
    print("loss:" + str(l))
    # 根据梯度调整权重
    params = params - eta * gradient(params, samples, labels)
# 训练完成进行预测
print(perdit(params, [0, 5]))
print(perdit(params, [1, 2]))
print(perdit(params, [0, -3]))
print(perdit(params, [-1, -5]))
# 创建图形
fig = plt.figure()
# 画数据点
plt.plot(points, 'o')
# ========================================================================================
# Softmax回归示例
# ========================================================================================
# 定义Softmax函数，可以一次性计算x对若干W=[w0,w1,...]的Softmax
def softmax(W, x):
    vec = np.exp(np.dot(W.T, x))
    sum = np.sum(vec)
    return vec / sum
# 定义预测函数，返回Softmax最大值得下标
def perdit(W, x):
    vec = softmax(W, x)
    out = np.argmax(vec)
    return out
# 定义交叉熵损失
def loss(W, X, y):
    sum = 0.0
    for i in range(len(X)):
        vec = np.log(softmax(W, X[i]))
        sum += vec[y[i]]
    return -sum / len(X)
# 使用数值差分来计算梯度（效率慢，但是便于理解）
def gradient(W, X, y):
    grad = np.zeros_like(W)
    # 分别对每个维度求梯度
    for i in range(len(W)):
        for j in range(len(W[i])):
            w = W[i][j]
            # 利用差分求导
            W[i][j] = w - 0.01
            loss1 = loss(W, X, y)
            W[i][j] = w + 0.01
            loss2 = loss(W, X, y)
            W[i][j] = w
            grad[i][j] = (loss2 - loss1) / (0.01 + 0.01)
    return grad
# 准备数据四个正例，四个反例
samples = [[5, 6], [1, 1], [3, 2], [3, 4],
     [2, -4], [4, -1], [5, -2], [1, -3],
     [-4, -3], [-2, -2], [-5, -1], [-3, -5],
     [-1, 1], [-4, 3], [-2, 5], [-3, 6]]
labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
# 初始化参数和超参数以及画图缓存
params = np.zeros((2, 4))
eta = 0.1
points = []
# 先计算初步损失
l = loss(params, samples, labels)
# 训练代码(梯度下降)
while l > 0.1:
    # 看看损失
    l = loss(params, samples, labels)
    points.append(l)
    print("loss:" + str(l))
    # 根据梯度调整权重
    params = params - eta * gradient(params, samples, labels)
# 训练完成进行预测
print(perdit(params, [0, 3]))
print(perdit(params, [3, 3]))
print(perdit(params, [3, -3]))
print(perdit(params, [-3, -3]))
print(perdit(params, [-3, 3]))
# 创建图形
fig = plt.figure()
# 画数据点
plt.plot(points, 'o')
# ========================================================================================
# 支持向量机示例
# ========================================================================================
# 定义模型，根据w和b判别x
def predict(w, b, x):
    return 1 if np.dot(w.T, x) + b > 0 else -1
# 在训练数据上，根据当前alpha值计算w
def calc_w(alpha, X, y):
    w = np.zeros_like(X[0], dtype=float)
    for i in range(len(X)):
        w += alpha[i] * y[i] * X[i]
    return w
# 在训练数据上，根据当前alpha值计算b
def calc_b(alpha, X, y):
    # 获取第一个大于0的alpha，约束生效，用于解出b
    idx = np.argmax(alpha > 0)
    b = y[idx]
    for i in range(len(X)):
        b -= alpha[i] * y[i] * np.dot(X[i].T, X[idx])
    return b
# 在训练数据(X,y)上，根据当前对偶问题alpha计算w和b，并判别样本x的结果
def f(alpha, X, y, x):
    w = calc_w(alpha, X, y)
    b = calc_b(alpha, X, y)
    # 判别结果是f(x)=w^Tx+b
    return np.dot(w.T, x) + b
# 定义SMO算法
def SMO(alpha, X, y, C):
    # 随机选取两个下标（正常应该依据下标选择算法选择更好的两个）
    p1 = random.randint(0, len(alpha) - 1)
    p2 = random.randint(0, len(alpha) - 1)
    if p1 == p2:
        return
    # 计算f(x1)和f(x2)
    f_x1 = f(alpha, X, y, X[p1])
    f_x2 = f(alpha, X, y, X[p2])
    # 计算eta
    eta = np.dot(X[p1].T, X[p1]) + np.dot(X[p2].T, X[p2]) - 2 * np.dot(X[p1].T, X[p2])
    # 计算M,L,R
    M = alpha[p2] + y[p2] * ((f_x1 - y[p1]) - (f_x2 - y[p2])) / eta
    L = 0.0
    R = 0.0
    if y[p1] == y[p2]:
        L = max(0, alpha[p2] + alpha[p1] - C)
        R = min(C, alpha[p2] + alpha[p1])
    else:
        L = max(0, alpha[p2] - alpha[p1])
        R = min(C, C + alpha[p2] - alpha[p1])
    # 计算新的alpha1和alpha2
    alpha2 = L if (M < L) else (R if M > R else M)
    alpha1 = alpha[p1] + y[p1] * y[p2] * (alpha[p2] - alpha2)
    # 更新到参数中
    alpha[p1] = alpha1
    alpha[p2] = alpha2
# 准备数据四个正例，四个反例
samples = np.array([[5, 6], [1, 1], [3, 2], [3, 4], [2, -4], [4, -1], [5, -2], [1, -3]])
labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])
# 初始化参数alpha向量
params = np.zeros_like(labels, dtype=float)
# SMO算法
for epoch in range(20):
    SMO(params, samples, labels, 1)
# 创建图形
fig = plt.figure()
# 计算w和b
w = calc_w(params, samples, labels)
b = calc_b(params, samples, labels)
points_x = [0, (-4 * w[1] - b)/w[0]]
points_y = [-b/w[1], 4]
# 画数据点
plt.plot([sublist[0] for sublist in samples], [sublist[1] for sublist in samples], 'o')
plt.plot(points_x, points_y)
# ========================================================================================