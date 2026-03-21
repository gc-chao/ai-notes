# ========================================================================================
# 导入组件
# ========================================================================================
import pickle
import numpy as np
from mydl import Math
from mydl import DataSet, DataLoader, Trainer
from mydl import BPNetwork, CNN, VGG16
from mydl import RNNUnit, TimeRNNUnit, RNNLM, Seq2Seq, AttentionSeq2Seq, Transformer
# ========================================================================================
# 定义数据集
# ========================================================================================
class MnistDataSet(DataSet):
    def __init__(self, train=True, to_image=False):
        super(MnistDataSet, self).__init__()
        with open("mnist.pkl", 'rb') as f:
            dataset = pickle.load(f)
        # 归一化
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
        # 扁平化
        if to_image is True:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
        if train is True:
            self.samples = dataset['train_img']
            self.labels = dataset['train_label']
        else:
            self.samples = dataset['test_img']
            self.labels = dataset['test_label']
    # 为VGG16准备数据（因为数据量太大，这里做了限制）
    def for_vgg16(self, limit_to):
        self.samples = self.samples[:limit_to]
        self.labels = self.labels[:limit_to]
        results = Math.empty((limit_to, 3, 224, 224))
        for i in range(limit_to):
            channel = np.pad(self.samples[i, 0], ((98, 98), (98, 98)), 'constant', constant_values=0)
            results[i][0] = channel
            results[i][1] = channel
            results[i][2] = channel
        self.samples = results
# ========================================================================================
class PTBDataSet(DataSet):
    def __init__(self, time_length=20, to_one_hot=True, type='train', corpus_size=1001):
        super(PTBDataSet, self).__init__()
        # 加载数据，获取语料库、词和ID的映射
        with open('ptb.vocab.pkl', 'rb') as f:
            self.word_to_id, self.id_to_word = pickle.load(f)
            self.corpus = np.load("ptb." + type + ".npy")
        # 仅使用语料库前corpus_size个数据
        if corpus_size is not None:
            corpus = self.corpus[:corpus_size]
        # 计算词汇表大小
        self.vocab_size = int(max(corpus) + 1)
        # 转换为one-hot
        if to_one_hot is True:
            one_hot = Math.zeros((corpus.shape[0], self.vocab_size), dtype=np.int32)
            for idx, word_id in enumerate(corpus):
                one_hot[idx, word_id] = 1
            corpus = one_hot
        # 从第一个词到倒数第二个词作为样本，第二个词到最后一个词为标记
        samples = corpus[:-1]
        labels = corpus[1:]
        # 按时间长度划分为批
        count = len(samples) // time_length
        time_samples = Math.empty((count, time_length, self.vocab_size), dtype=np.int32)
        time_labels = Math.empty((count, time_length, self.vocab_size), dtype=np.int32)
        for i in range(count):
            time_samples[i] = samples[time_length * i: time_length * (i + 1)]
            time_labels[i] = labels[time_length * i: time_length * (i + 1)]
        self.samples = time_samples
        self.labels = time_labels
# ========================================================================================
class QADataSet(DataSet):
    def __init__(self, file_name, limit_to=100):
        super(QADataSet, self).__init__()
        self.id_to_char = {}
        self.char_to_id = {}
        self.load_data(file_name, limit_to)
    def load_data(self, file_name, limit_to):
        questions, answers = [], []
        # 读取所有的数据，以"_"作为分割，前面是输入，后面是输出
        count = 1
        for line in open(file_name, 'r'):
            if count > limit_to: break
            else: count += 1
            idx = line.find('_')
            questions.append(line[:idx])
            answers.append(line[idx:-1])
        # 根据每一对QA，更新词汇表
        for i in range(len(questions)):
            q, a = questions[i], answers[i]
            self.update_vocab(q)
            self.update_vocab(a)
        # 创建样本和标记，形状是样本数、序列长度、字符数（存放one-hot)
        samples = Math.zeros((len(questions), len(questions[0]), len(self.char_to_id)), dtype=np.int32)
        labels = Math.zeros((len(questions), len(answers[0]), len(self.char_to_id)), dtype=np.int32)
        for i, seq in enumerate(questions):
            sample = [self.char_to_id[c] for c in list(seq)]
            for j, char_id in enumerate(sample):
                samples[i, j, char_id] = 1
        for i, seq in enumerate(answers):
            label = [self.char_to_id[c] for c in list(seq)]
            for j, char_id in enumerate(label):
                labels[i, j, char_id] = 1
        self.samples = samples
        self.labels = labels
        self.vocab_size = len(self.char_to_id)
        # 提示字符，也就是下划线
        self.hint = labels[0, 0, :]
        self.seq_length = len(labels[0])
    # 对每一个文本更新字符表
    def update_vocab(self, seq):
        chars = list(seq)
        for i, char in enumerate(chars):
            if char not in self.char_to_id:
                tmp_id = len(self.char_to_id)
                self.char_to_id[char] = tmp_id
                self.id_to_char[tmp_id] = char
    def ids_2_str(self, ids):
        chars = ''
        ids = ids.argmax(axis=1)
        for i in range(len(ids)):
            chars += self.id_to_char[ids[i]]
        return chars
# ========================================================================================
# 应用一：训练前馈神经网络
# ========================================================================================
if False:
    # 加载数据集
    train_dataset = MnistDataSet(train=True)
    test_dataset = MnistDataSet(train=False)
    train_data = DataLoader(train_dataset, 1000)
    (x_test, y_test) = test_dataset.samples, test_dataset.labels
    # 创建网络
    model = BPNetwork()
    model.add_layers([784, 50, 10])
    # 如果已经有参数可以通过这个加载
    model.load()
    better_accuracy = model.classify_accuracy(x_test, y_test)
    trainer = Trainer(model, train_data)
    train_loss, test_loss, train_accuracy, test_accuracy = trainer.train(x_test, y_test, 50)
    current_accuracy = test_accuracy[len(test_accuracy) - 1]
    if current_accuracy > better_accuracy:
        print("Accuracy: " + str(better_accuracy) + " => " + str(current_accuracy))
        better_accuracy = current_accuracy
        model.save()
# ========================================================================================
# 应用二：训练卷积神经网络
# ========================================================================================
if False:
    train_dataset = MnistDataSet(train=True, to_image=True)
    test_dataset = MnistDataSet(train=False, to_image=True)
    data_loader = DataLoader(train_dataset, 200)
    (x_test, y_test) = test_dataset.samples, test_dataset.labels
    network = CNN(input_shape=(1, 28, 28), filter_count=1, filter_size=5, filter_pad=0,
                  filter_stride=1, hidden_size=100, output_size=10)
    trainer = Trainer(network, data_loader, "sgd")
    trainer.train(x_test, y_test, 5)
# ========================================================================================
# 应用三：训练VGG16
# ========================================================================================
if False:
    train_dataset = MnistDataSet(train=True, to_image=True)
    test_dataset = MnistDataSet(train=False, to_image=True)
    # 针对VGG16对数据集进行处理
    train_dataset.for_vgg16(limit_to=100)
    test_dataset.for_vgg16(limit_to=100)
    data_loader = DataLoader(train_dataset, 2)
    network = VGG16(input_shape=(3, 224, 224))
    trainer = Trainer(network, data_loader, "sgd")
    trainer.train(test_dataset.samples, test_dataset.labels, 5)
# ========================================================================================
# 应用四：使用RNN单元和TimeRNN层
# ========================================================================================
if False:
    layer = RNNUnit(4, 10)
    ys = layer.forward(Math.array([[1, 2, 1, 1], [3, 4, 5, 6], [5, 6, 8, 7]]))
    ys = layer.forward(Math.array([[-1, -2, 0, 0], [-3, -4, 0, 3], [-5, -6, 1, 0]]))
    ys = Math.array(ys)
    dys = ys - 1
    dxs = layer.backward(dys)
    print(dxs.shape)
    layer = TimeRNNUnit(4, 10)
    ys = layer.forward(Math.array([[[1, 2, 1, 1], [-1, -2, 0, 0]], [[3, 4, 5, 6], [-3, -4, 0, 3]], [[5, 6, 8, 7], [-5, -6, 1, 0]]]))
    dys = ys - 1
    dxs = layer.backward(dys)
    print(dxs.shape)
# ========================================================================================
# 应用五：训练RNNLM模型
# ========================================================================================
if False:
    dataset = PTBDataSet()
    dataloader = DataLoader(dataset, 10, shuffle=False)
    model = RNNLM(dataset.vocab_size, wordvec_size=80, hidden_size=100)
    model.load()
    trainer = Trainer(model, dataloader, "sgd", lr=5)
    # 使用梯度截断，开始训练
    trainer.optimizer.max_grad = 5.0
    trainer.train(dataset.samples, dataset.labels, 80)
    model.save()
    # 使用模型生成句子，先假定you是第一个输入的单词
    start_word = 'you'
    start_id = dataset.word_to_id[start_word]
    # 三个停止单词
    stop_words = ['N', '<unk>', '$']
    skip_ids = [dataset.word_to_id[w] for w in stop_words]
    # 文本生成
    generated_ids = model.generate(start_id, skip_ids)
    txt = ' '.join([dataset.id_to_word[i] for i in generated_ids])
    txt = txt.replace(' <eos>', '.\n')
    print(txt)
# ========================================================================================
# 应用六：训练Seq2Seq模型
# ========================================================================================
if False:
    dataset = QADataSet('addition.txt', limit_to=100)
    dataloader = DataLoader(dataset, 10, shuffle=True)
    # seq_length - 1的意思是预测的是下划线之后的四个字符
    model = Seq2Seq(dataset.vocab_size, wordvec_size=16, hidden_size=100,
                    sampled_size=dataset.seq_length - 1, hint=dataset.hint)
    trainer = Trainer(model, dataloader, "sgd", lr=5)
    # 使用梯度截断
    trainer.optimizer.max_grad = 5.0
    trainer.train(dataset.samples, dataset.labels, 100)
    # 给出一个测试的问题看结果（需要改变形状(1, ?, ?)来适应批处理）
    question = dataset.samples[0]
    questions = Math.reshape(question, (1, len(dataset.samples[0]), -1))
    # 结果只有一个，就是所求
    answer = model.generate(questions)[0]
    print('question "' + dataset.ids_2_str(question) + '" answer is "' + dataset.ids_2_str(answer) + '".')
# ========================================================================================
# 应用七：训练AttentionSeq2Seq模型
# ========================================================================================
if False:
    dataset = QADataSet('date.txt', limit_to=100)
    dataloader = DataLoader(dataset, 10, shuffle=True)
    model = AttentionSeq2Seq(dataset.vocab_size, wordvec_size=16, hidden_size=100,
                             sampled_size=dataset.seq_length - 1, hint=dataset.hint)
    trainer = Trainer(model, dataloader, "sgd", lr=5)
    # 使用梯度截断
    trainer.optimizer.max_grad = 5.0
    trainer.train(dataset.samples, dataset.labels, 30)
    # 给出一个测试的问题看结果（需要改变形状(1, ?, ?)来适应批处理）
    question = dataset.samples[0]
    questions = Math.reshape(question, (1, len(dataset.samples[0]), -1))
    # 结果只有一个，就是所求
    answer = model.generate(questions)[0]
    print('convert "' + dataset.ids_2_str(question) + '" to "' + dataset.ids_2_str(answer) + '".')
# ========================================================================================
# 应用八：训练Transformer模型
# ========================================================================================
if False:
    dataset = QADataSet('datet.txt', limit_to=500)
    dataloader = DataLoader(dataset, 8, shuffle=False)
    model = Transformer(sampled_size=dataset.seq_length-1, sos=dataset.hint, eos=None,
                        vocab_size=dataset.vocab_size, wordvec_size=40,
                        module_num=1, head_num=2, k_size=20, v_size=20, h_size=80)
    trainer = Trainer(model, dataloader, "sgd", lr=0.01)
    # 使用梯度截断
    trainer.optimizer.max_grad = 5.0
    trainer.train(dataset.samples, dataset.labels, 30)
    # 给出一个测试的问题看结果（需要改变形状(1, ?, ?)来适应批处理）
    question = dataset.samples[0]
    questions = Math.reshape(question, (1, len(dataset.samples[0]), -1))
    # 结果只有一个，就是所求
    answer = model.generate(questions)[0]
    print('convert "' + dataset.ids_2_str(question) + '" to "' + dataset.ids_2_str(answer) + '".')
# ========================================================================================