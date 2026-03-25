# ========================================================================================
# sklearn的应用示例
# ========================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import MultinomialHMM
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
# ========================================================================================
# 数据集使用示例
# ========================================================================================
# ----------------------------------------------------------------------------------------
# 鸢尾花数据集示例
# ----------------------------------------------------------------------------------------
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 包括150个样本，每个样本四个特征，分别是花萼长度、花萼宽度、花瓣长度、花瓣宽度
print(iris.data)
# 包括150个样本的标签
print(iris.target)
# ----------------------------------------------------------------------------------------
# 分类数据集示例
# ----------------------------------------------------------------------------------------
# 生成数据集
X, y = datasets.make_classification(
    n_samples=1000,          # 样本数量，默认100
    n_features=3,            # 特征数量，默认20
    n_informative=2,         # 信息特征数量，默认2
    n_redundant=0,           # 冗余特征数量，默认2
    n_classes=4,             # 类别数量，默认2
    n_clusters_per_class=1,  # 每个类别中的簇数量
    class_sep=3,             # 类别之间的分离度，默认为1.0
    random_state=10)         # 随机数种子，默认None
# 建立图表
figure = plt.figure()
# 添加坐标轴
ax = figure.add_subplot(111, projection="3d")
# 刚好用分类标签y作为颜色映射
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
plt.show()
# ========================================================================================
# 降维示例
# ========================================================================================
# 给出8个示例数据
X = np.array([[10, 1], [9, 0], [10, -1], [11, 0], [0, 9], [1, 10], [0, 11], [-1, 10]])
# 使用PCA算法降维到1维
pca = PCA(n_components=1)
Y = pca.fit_transform(X)
print("主成分系数矩阵:\n", pca.components_)
print("主成分方差比例:\n", pca.explained_variance_ratio_)
print("降维后的数据:\n", Y)
# ========================================================================================
# 使用典型的分类器
# ========================================================================================
# 初始化数据集X，其中每一个二维样本对应y中的一个分类
X = [[5, 6], [1, 1], [3, 2], [3, 4],
     [2, -4], [4, -1], [5, -2], [1, -3],
     [-4, -3], [-2, -2], [-5, -1], [-3, -5],
     [-1, 1], [-4, 3], [-2, 5], [-3, 6]]
y = [1, 1, 1, 1,
     2, 2, 2, 2,
     3, 3, 3, 3,
     4, 4, 4, 4]
# 定义各个有监督学习分类器
classifiers = [
    # 创建KNN模型示例，采用4个邻居计算的2范数欧氏距离
    KNeighborsClassifier(n_neighbors=4, p=2),
    # 创建逻辑回归模型实例
    LogisticRegression(),
    # 建立高斯朴素贝叶斯模型
    GaussianNB(),
    # 建立决策树模型
    DecisionTreeClassifier(),
    # 建立支持向量机，使用RBF核，弹性1.0
    SVC(kernel="rbf", C=1.0),
]
# 对每一个分类器进行训练和预测
for i in range(len(classifiers)):
    # 取出一个分类器
    classifier = classifiers[i]
    # 训练过程，调用分类器的fit即可
    classifier.fit(X, y)
    # 预测五个点的分类并打印结果
    result = classifier.predict([[0, 0], [3, 3], [3, -3], [-3, -3], [-3, 3]])
    print(result)
# ========================================================================================
# 聚类示例
# ========================================================================================
# 初始化数据集X
X = [[5, 6], [1, 1], [3, 2], [3, 4],
     [2, -4], [4, -1], [5, -2], [1, -3],
     [-4, -3], [-2, -2], [-5, -1], [-3, -5],
     [-1, 1], [-4, 3], [-2, 5], [-3, 6]]
# 定义各个无监督学习聚类模型
clusters = [
    # 建立K均值聚类模型，K设置为4
    KMeans(n_clusters=4, random_state=42),
    # 建立谱系聚类模型，设置聚类数为4
    SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=4),
    # 建立高斯混合模型，设置为由4个模型混合
    GaussianMixture(n_components=4)
]
# 对每一个聚类模型进行训练和预测
for i in range(len(clusters)):
    # 取出一个聚类模型
    cluster = clusters[i]
    print(cluster)
    # 训练模型，调用fit方法即可
    cluster.fit(X)
    # 对于K均值聚类模型，输出K均值簇中心点
    if i == 0:
        print("K均值聚类簇中心:\n", cluster.cluster_centers_)
    # 预测四组测试数据，每个类别两个样本
    if i != 1:
        print('预测结果:')
        print(cluster.predict([[2, 1], [3, 3]]))
        print(cluster.predict([[2, -1], [3, -3]]))
        print(cluster.predict([[-2, -1], [-3, -3]]))
        print(cluster.predict([[-2, 1], [-3, 3]]))
    else:
        # 谱系聚类有自己的预测方法，就是在适配数据的同时进行标注
        print('谱系聚类标记：')
        print(cluster.fit_predict(X))
# ========================================================================================
# 集成学习示例
# ========================================================================================
# 初始化自带的moons数据集，并拆分成训练集和测试集
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 建立AdaBoost分类器，基学习器使用决策树
classifier = AdaBoostClassifier(DecisionTreeClassifier())
# 训练并打印测试结果
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
# 建立随机森林分类器
classifier = RandomForestClassifier()
# 训练并打印测试结果
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
# ========================================================================================
# 使用隐马尔科夫模型（请注意hmmlearn需要独立安装）
# ========================================================================================
# 定义隐状态
u = ['晴', '阴', '雨', '雪']
# 定义观测状态
v = ['散步', '购物', '做家务']
# 给定初始概率
pi = np.array([0.5, 0.3, 0.1, 0.1])
# 给定转移矩阵
A = np.array([
    [0.5, 0.3, 0.1, 0.1],
    [0.3, 0.3, 0.3, 0.1],
    [0.4, 0.2, 0.3, 0.1],
    [0.5, 0.2, 0.1, 0.2]
])
# 给定发射矩阵
B = np.array([
    [0.5, 0.4, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6],
    [0.1, 0.2, 0.7]
])
# 建立处理离散问题的HMM模型
hmm = MultinomialHMM(n_trials=4)
hmm.n_components = len(u)
hmm.n_features = len(v)
hmm.startprob_ = pi
hmm.transmat_ = A
hmm.emissionprob_ = B
# 观测到连续三天散步、做家务、购物
ob_seq = np.array([0, 2, 1])
# 转换为one-hot
encoder = OneHotEncoder(sparse=False, dtype=np.int32)
onehot = encoder.fit_transform(ob_seq.reshape(-1, 1))
# 计算问题，产生观测序列的概率多大
log_score = hmm.score(onehot)
print("产生观测序列'" + "->".join(map(lambda i: v[i], ob_seq)) + "'的概率是：" + str(np.exp(log_score)))
# 解码问题，在观测序列下判断天气状况
log_prob, state_sequence = hmm.decode(onehot, algorithm='viterbi')
print("观测到连续三天：" + "->".join(map(lambda i: v[i], ob_seq)))
print("最有可能的天气：" + "->".join(map(lambda i: u[i], state_sequence)))
# 从模型中采样（正好用于稍后训练一个hmm用）
samples, state_sequence = hmm.sample(1000)
# 学习问题，只知道观测序列，不知道模型
hmm = MultinomialHMM(n_trials=4)
hmm.n_components = len(u)
hmm.n_features = len(v)
# 用生成的样本训练
hmm.fit(samples)
# 打印转移矩阵
print(np.round(hmm.transmat_, 1))
# 打印发射矩阵
print(np.round(hmm.emissionprob_, 1))
# ========================================================================================
# 神经网络示例
# ========================================================================================
# 初始化自带数据集moons，并拆分成训练数据和测试数据
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
# 定义多层感知机分类器，使用SGD优化器
classifier = MLPClassifier(solver="sgd")
# 训练模型
classifier.fit(X_train, y_train)
# 在测试集上评分
print(classifier.score(X_test, y_test))
# ========================================================================================