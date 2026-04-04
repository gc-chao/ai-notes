# ========================================================================================
# 在mydl之上的CBOW模型应用
# ========================================================================================
import mydl
import numpy as np
class TextDataSet(mydl.DataSet):
    def __init__(self):
        super(TextDataSet, self).__init__()
        text = 'This is my family. There are four people in my family. They are my father, my mother, my brother and I. My father is a teacher. He teaches English in a school. My mother is a doctor. She works in a hospital. My brother is a student. He studies in a primary school. I am also a student. I study in the same school as my brother. We all love each other and we always help each other.'
        corpus, word_to_id, id_to_word = self.preprocess(text)
        vocab_size = len(word_to_id)
        contexts, labels = self.create_contexts_labels(corpus)
        contexts = self.convert_one_hot(contexts, vocab_size)
        labels = self.convert_one_hot(labels, vocab_size)
        self.size = vocab_size
        self.samples = contexts
        self.labels = labels
    # 预处理文本为语料库
    def preprocess(self, text):
        # 词编号列表
        word_to_id = {}
        id_to_word = {}
        # 分词
        words = text.lower().replace('.', ' .').replace(',', ' ,').split(' ')
        # 填充列表
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
        # 形成语料库
        corpus = [word_to_id[word] for word in words]
        return np.array(corpus), word_to_id, id_to_word
    def create_contexts_labels(self, corpus, window_size=1):
        labels = corpus[window_size:-window_size]
        contexts = []
        # 遍历每个词（窗口大小上的词没有上下文，因此边界去除）
        for j in range(window_size, len(corpus) - window_size):
            context = []
            # 找到这个词的上下文
            for i in range(-window_size, window_size + 1):
                if i == 0:
                    continue
                context.append(corpus[j + i])
            contexts.append(context)
        return np.array(contexts), np.array(labels)
    def convert_one_hot(self, corpus, vocab_size):
        N = corpus.shape[0]
        if corpus.ndim == 1:
            one_hot = np.zeros((N, vocab_size), dtype=np.int32)
            for idx, word_id in enumerate(corpus):
                one_hot[idx, word_id] = 1
        elif corpus.ndim == 2:
            C = corpus.shape[1]
            one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
            for idx_0, word_ids in enumerate(corpus):
                for idx_1, word_id in enumerate(word_ids):
                    one_hot[idx_0, idx_1, word_id] = 1
        return one_hot
np.set_printoptions(precision=3)
dataset = TextDataSet()
dataloader = mydl.DataLoader(dataset, batch_size=3)
network = mydl.CBOW(vocab_size=dataset.size, hidden_size=5)
trainer = mydl.Trainer(network, dataloader, "sgd", lr=0.2)
trainer.train(dataset.samples, dataset.labels, 200)
# ========================================================================================