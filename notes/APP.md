
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

# 自然语言处理

## 历史

NLP的历史以表示的变化为标志，早期的输入是符号或词，后来是字符序列甚至是Unicode字符的单个字节。

NLP遇到的首要问题是基于词的语言模型，由于词空间很大，必须在极高维度和稀疏的离散空间上工作。按照处理的思想，可以将语言模型的发展分为几个阶段。

- 早期自然语言处理(1950s–1990s)：主要采用规则驱动的方法，依赖专家编写规则，难以处理歧义和复杂句式。
- 统计方法语言模型(1990s–2000s)：主要基于计数，通过对整个语料库的一次学习来获得单词的分布式表示。这一阶段主要分为两条技术路线。一类基于词袋(Bag-of-words)模型，通常是根据单词和文档的共现矩阵，用各类降维方法（如LSA、PLSA、LDA）得到分布式词向量，使得相似的单词向量相近。第二类基于马尔可夫假设，通过词序列概率建模（例如n-gram模型和HMM模型）。它们的缺点是依赖人工特征工程且难以捕捉长期依赖。
- 神经网络语言模型(2000s-2010s中期)：主要基于推理，通过神经网络在语料库上迭代的学习上下文关系，来获得词的分布式表示，使得语义相近的词在向量空间中相近。这类方法可以克服维数灾难，将每个词映射为低维稠密向量，并且更适合增量学习。代表性工作包括：2003年提出的神经网络语言模型(NNLM)首次将词嵌入(word embedding)引入NLP、2013年提出的Word2Vec（包括CBOW和skip-gram）实现了快速学习高质量词嵌入、以及2014年提出的通用编码器-解码器框架Seq2Seq开创了神经机器翻译的时代。
- 大语言模型(2017-)：以Transformer架构的提出为关键节点，并在此发展出BERT、GPT等模型，进而发展成大规模语言模型。

## 文本表示与话题模型

### 向量空间模型

向量空间模型(VSM)于20世纪70年代提出，它是SMART信息检索系统的核心算法。VSM的主要贡献有三点。一是它基于词袋(bag-of-words)模型将文档(document)表示为高维空间中的向量，每个维度对应一个词条(term)，取值代表词条的权重。二是提出了TF-IDF用来计算词条的权重，其中词频TF表示词条在文档中的重要性，逆文档频率IDF表示词条在整个语料库中的重要性。三是引入了余弦相似度来度量文档之间的相似性。

若有$N$个文档$D=\{d_1,d_2,...,d_N\}$和$M$个词条$W=\{w_1,w_2,...,w_M\}$，则词条和文档之间可以构成一个矩阵$X$。其每一元素$(X)_{ij}$表示第$i$个词条在第$j$个文档下的权重，并按TF-IDF计算得出。TF-IDF的定义为$\mathtt{TFIDF}_{ij}=tf_{ij}\log\frac{N}{df_{i}}=\frac{f_{ij}}{\sum_{k=1}^{M}f_{kj}}\log\frac{N}{df_{i}}$。其中，$f_{ij}$是词条$w_i$出现在文档$d_j$中的原始频率，$\sum_{k=1}^{M}f_{kj}$是文档$d_j$的总词条数，$df_{i}$是含有词条$w_i$的所有文档数。可见，在单一文档中，词条出现的频数越高，其对该文档的重要性越高；而在整个语料库中，词条覆盖的文档数越多，其对区分文档的重要性越低。将词条文档矩阵$X$写成列向量组$[x_1,x_2,...,x_N]$，每一个向量$x_j$就可表示第$j$个文档在所有词条上的权重，因此文档$x_1,x_2$之间的相似度可以通过计算向量的余弦$\frac{x_1·x_2}{||x_1||·||x_2||}$来评估。

VSM的优点在于模型简单，且计算上因矩阵稀疏而效率更高，缺点是词条向量无法建模一词多义和同义词。

### 潜在语义分析

潜在语义分析(LSA)也称为潜在语义索引(LSI)于1990年提出。它是一种无监督学习方法，主要用于文档检索和文档话题分析。其话题向量空间模型(TVSM)在VSM的基础上引入了话题(topic)，并利用矩阵分解的思想来发现文档与词条之间关于话题的关系。这里的话题泛指文档的内容主题，通常由若干语义相关的词条构成，一个文档对应多个话题。文档的相似度可以体现在话题上的相似度。

已知原始词条文档矩阵$X^{(M×N)}$，假设所有文档共包含$L$个话题，则可以进一步定义两个矩阵。词条和话题之间可以构成词条话题矩阵$T^{(M×L)}$，其元素$(T)_{ik}$表示第$i$个词条在第$k$个话题下的权重，并可以写成列向量组形式$T=[t_1,t_2,...,t_L]$，每个话题向量$t_k=(t_k^{(1)},t_k^{(2)},...,t_k^{(M)})^T$。话题和文档之间可以构成话题文档矩阵$Y^{(L×N)}$，其元素$(Y)_{kj}$表示第$k$个话题在第$j$个文档下的权重，并可以写成列向量组形式$Y=[y_1,y_2,...,y_N]$，每个向量$y_i=(y_i^{(1)},y_i^{(2)},...,y_i^{(L)})^T$。在此基础上，原始词条向量$x_j$可以通过它在话题空间中的向量$y_j$表示为$x_j=\sum\limits_{k=1}^{L}y_j^{(k)}t_k$，或写成矩阵乘积的形式$X=TY$。两个文档的相似度可由它们向量的内积或余弦（标准化内积）来衡量。

潜在语义分析旨在利用矩阵分解，求得低维的表示$T$和$Y$来近似$X$，即$X≈\widetilde{T}\widetilde{Y}$。典型的方式是使用截断的奇异值分解，得$X=UΣV^T$，保留前$L$个奇异值以及对应的奇异向量以满足$X≈U_LΣ_LV_L^T$。令$\widetilde{T}=U_LΣ_L$，$\widetilde{Y}=Σ_LV_L^T$，$\widetilde{Y}$列向量即文档在$L$维话题空间中的低维表示。另一种常用的分解方式是非负矩阵分解（其特点是矩阵的所有元素非负）。

### 概率潜在语义分析

概率潜在语义分析(PLSA)受潜在语义分析的启发，是一种采用概率模型的无监督学习方法。它的基本思想可以从实际场景来引入，在撰写文档时，我们往往是先确定某一个话题，然后再根据话题撰写文本。于是，PLSA用概率统计来建模这一过程，将词条、文档、话题看作随机变量，将观测到的词条文档共现矩阵看作依据话题分布（概率）采样的结果。只不过话题是观测不到的隐变量，而能观测到的是文档和词条。

定义词条文档共现矩阵$T$，行对应词条、列对应文本，每个元素表示每个词条文档对$(w,d)$出现的次数。$p(d)$为文档$d$的概率分布、$p(z|d)$为文档生成话题的条件概率分布、$p(w|z)$为话题生成词条的条件概率分布，$p(d|z)$为话题生成文档的条件概率分布，它们均服从多项分布。这样可以得到两个模型，分别是生成模型和共现模型。

生成模型认为词条文档共现矩阵的数据是按照如下的步骤生成出来的。首先跟定文档$d$，然后对每个文档依据分布$p(z|d)$生成话题$z$，最后再根据$p(w|z)$生成词条$w$。这一过程可以表达为$p(w,d)=p(w|d)p(d)=p(d)\sum\limits_zp(w,z|d)$，若进一步考虑给定$z$的条件下$w$和$d$独立，则上式可进一步写为$p(w,d)=p(d)\sum\limits_zp(z|d)p(w|z)$。这样词条文档共现矩阵的概率分布可以利用上式通过公式$p(T)=\prod_\limits{(w,d)}p(w,d)^{n(w,d)}$计算。

共现模型认为词条文档共现矩阵的数据是按照如下的步骤生成出来的。首先依据话题分布$p(z)$采样得到一个话题，然后对每个话题根据分布$p(w,d|z)$生成词条文档对。这一过程可以表达为$p(w,d)=\sum\limits_{z} p(z)p(w,d|z)$，若进一步考虑给定$z$的条件下$w$和$d$独立，则上式可进一步写为$p(w,d)=\sum\limits_{z}p(z)p(w|z)p(d|z)$。这样词条文档共现矩阵的概率分布可以利用上式通过公式$p(T)=\prod_\limits{(w,d)}p(w,d)^{n(w,d)}$计算。

生成模型和共现模型虽然在问题描述上不同，但在数学意义上是等价的。$w$和$d$在生成模型中是非对称的，在共现模型中是对称的。PLSA通过引入话题隐变量$z$，把直接建立词条文本共现分布$p(w,d)$所需$O(M·N)$的参数量降低为$O(M·L+N·L)$。若令$X=p(w,d)$、$U=p(w|z)$、$V=p(d|z)$、$Σ=p(z)$，则共现模型刚好可以写成形式上类似于LSA的分解形式$X=UΣV$。

由于含有隐变量$z$，因此PLSA通常采用EM算法来求解。

E步计算$$p(z_k|w_i,d_j)=\frac{p(w_i|z_k)p(z_k|d_j)}{\sum\limits_{k=1}^Lp(w_i|z_k)p(z_k|d_j)}$$

M步计算$$p(w_i|z_k)=\frac{\sum\limits_{j=1}^Nn(w_i,d_j)p(z_k|w_i,d_j)}{\sum\limits_{h=1}^M\sum\limits_{j=1}^Nn(w_h,d_j)p(z_k|w_h,d_j)}\quad\,p(z_k|d_j)=\frac{\sum\limits_{i=1}^Mn(w_i,d_j)p(z_k|w_i,d_j)}{n(d_j)}$$

## 序列概率模型

从概率统计的角度，将长度$T$的文本序列看作一个随机向量$X=\{X_1,...,X_T\}$，每个随机变量的样本空间为一个词表$V$，整个向量的样本空间是$|V|^T$。序列的概率是$T$个词的联合概率$p(x_1,...,x_T)$=$P\{X_1=x_1,...,X_T=x_T\}$。序列概率模型有两个基本问题，一个是概率质量估计，一个是样本的生成。

### 质量估计和样本生成

**概率质量估计**

考虑到整个序列的样本空间太大很难直接建模，因此可以通过概率乘法公式将联合概率转换为单变量的条件概率$p(x_1,...,x_T)=p(x_1)\prod\limits_{t=2}^Tp(x_t|x_1,...,x_{t-1})$。学习的目标是最大化数据集上的对数似然$\arg\limits_θ\max\sum\limits_{n=1}^{N}\log\,p(x_1,...,x_{T_n};θ)=\arg\limits_θ\max\sum\limits_{n=1}^{N}\sum\limits_{t=1}^{T_n}\log\,p(x_{n,t}|x_{n,1},...,x_{n,t-1};θ)$。这里每一步的输入都需要上一步的输出，因此是一种自回归生成模型。

**序列的生成**

自回归能产生无限长的序列，因此通常设置EOS符号来表示序列结束。生成时，既可以从左到右进行贪心搜索，每一步生成最可能的词，也可以采用束搜索，它是一种启发式方法，每一步生成$K$个当前累积概率最高的候选序列。

### N元统计模型

N元统计模型(n-gram)基于$n-1$阶马尔科夫假设。其定义给定前$n-1$个标记后的第$n$个标记的条件概率为$p(x_t|x_1,...,x_{t-1})=p(x_t|x_{t-n+1},...,x_{t-1})$。 $n=1,2,3$时分别称为一元语法、二元语法、三元语法。这样，一元语法的模型中，每个位置的词可看作从多项分布中独立同分布采样得到，因此在模型上的最大似然估计等价于频率估计。

**N元模型**

训练n-gram和n-1-gram后就能用概率简单的计算：$$p(x_t|x_{t-n+1},...,x_{t-1})=\frac{p_n(x_{t-n+1},...,x_t)}{p_{n-1}(x_{t-n+1},...,x_{t-1})}$$  

例如$n=3$时，在已知前两个词出现的情况下，出现第三个词的概率为$$p(x_t|x_{t-2},x_{t-1})=\frac{p(x_{t-2},x_{t-1},x_t)}{p(x_{t-2},x_{t-1})}$$

**稀疏样本问题**

n-gram的主要局限表现在当训练数据中没有出现某种序列组合时，由训练集得到的$p_n$可能为零。但是训练集中没有观测到某种序列组合
不代表在测试集中真的不会出现。同时，zipf定律暗示了通过增加训练数据来采样低频词的回报是低于线性的。因此，一种典型的解决办法是采用平滑的概率形式，以将观察到的组合迁移到类似的未观察部分。另一种办法是同时采用高阶n-gram模型和低阶n-gram模型，高阶模型可以提供更多容量，低阶模型用于避免零计数，当在高阶模型上的上下文计数为0时，回退(backoff)方法会转而使用低阶模型。

**维数灾难问题**

经典n-gram存在$|V|^n$种可能的模型，这会导致维数灾难。为了提高统计效率，基于类的语言模型引入了词类别，属于同一类别的词共享统计强度。但这种方法会忽略同类词的细粒度差异。

### 深度序列模型

深度序列模型利用神经网络来估计条件概率$p(x_t|x_1,...x_{t-1})$。网络的输入为历史时刻的表示$h_t$，输出是词表$V$中每个词的概率。

**网络结构**

深度序列模型的网络结构通常包括嵌入层、特征层、输出层。嵌入层将序列$x_1,...,x_{t-1}$通过嵌入映射为词向量序列$e_1,...,e_{t-1}$。特征层接收$e_1,...,e_{t-1}$，提取特征后输出当前的信息向量$h_t$。若采用简单加权平均方式，即$h_t=\sum\limits_{i=1}^{t-1}w_ie_i$（其中$w_i$是权重，可以固定也可以通过注意力动态计算）；若采用前馈神经网络，则只能保留固定长度的历史词向量序列，即$h_t=g(e_{t-1},e_{t-2},...,e_{t-n})$；若采用RNN，则可利用全部历史词向量序列或$h_{t-1}$，再结合当前时刻输入的词向量$e_t$，即$h_t=g(h_{t-1},e_t)$。输出层一般使用Softmax，输入是向量$h_t$，输出是词表中所有词的后验概率分布，即$o_t=softmax(Wh_t+b)$。

**词嵌入**

深度序列模型通常会在网络的前级进行词嵌入学习。典型的词嵌入方法是通过嵌入矩阵直接将符号映射为向量。令$M$是嵌入矩阵，共有$|V|$行，每行对应一个词的向量表示，$χ_t∈\{0,1\}^{|V|}$为词$x_t$的one-hot表示，则词$x_t$的向量表示为$e_t=Mχ_t$。

Mikolov团队在2013年提出了更高效的词嵌入方法Word2Vec，它包括CBOW(continuous bag-of-words)和skip-gram两个模型。CBOW通过两侧的上下文预测中间的目标词。典型的网络结构包括多个输入层（数量为上下文的大小）、中间层和输出层，层次之间是前馈全连接的，并且不使用激活函数。每个上下文的one-hot特征（例如$χ_{t-1},χ_{t+1}$）对应输入层，中间层的神经元远小于输入层，每个输入层和中间层之间共享权重$W_{in}$，中间层的输出是所有输入的平均。中间层到输出层权重为$W_{out}$，输出层与输入层具有相同的神经元且输出的$\hat{x_t}$是经过Softmax后的概率分布。前向传播可以表示为$h=W_{in}^Tχ_{t-1}+W_{in}^Tχ_{t+1}$、$\hat{x}_{t}=Softmax(\frac{1}{2}W_{out}^Th)$，因此从输入层到中间层相当于编码，从中间层到输出层相当于解码。$W_{in},W_{out}$就是学到的表示，既可以单独使用一个，也可以把二者组合使用。skip-gram则刚好与CBOW相反，它用中间的词来预测两侧的上下文，并且实际表现往往比CBOW更好。

**参数学习**

深度序列模型的参数学习方式通常是构造负对数似然，并通过梯度下降法找到极小点。

### 结合n-gram和神经语言模型

n-gram模型由于需要存储大量词频而通常参数量较大，但计算上只与上下文几个有限的词相关，而神经语言模型的计算开销正相关于参数数量，故可以考虑将两种模型进行结合（例如特征融合）以获得平衡。

# 机器视觉

## 目标检测

### YOLO

**YOLOv1**

- 简介：YOLOv1发布于2015年，它是仅使用一个卷积网络实现的端到端目标检测。
- 网络结构：YOLOv1的网络结构仿照GoogLeNet，但考虑性能开销没有采用Inception的不同大小卷积核的并行卷积，而是采用串联的形式（具体由表给出）。
- 工作原理：YOLOv1接收448x448的图像，经过主干网络提取特征后得到1024个通道的7x7特征图，最后通过全连接层输出30个7x7的检测结果。这相当于对原始图像划分了7x7的网格来识别，一个网格仅识别一个分类，并采用矩形边界框框出目标物体。为了简化问题，YOLOv1规定只有检测目标中心点坐落的网格才认为包含目标（其它网格即使包含相同目标，但因不是目标的中心，也不作为有目标的网格）。
- 输出格式：输出的7x7个网格，每个格子的预测结果是一个30维向量，由两个边界框信息（包含置信度$c^{(k)}$+中心点横向偏移量$t_x^{(k)}$+中心点纵向偏移量$t_y^{(k)}$+宽度$w^{(k)}$+高度$h^{(k)}$，$k=1,2$）和20个类别的得分$p(c_1),...,p(c_{20})$构成。每个边界框独立预测自己的置信度，而偏移量指边界框的中心点与该网格左上角的相对位置。
- 数据集处理：边界框中心点偏移量和边界框都会经过尺度变换以防止学习发散。
- 损失函数：包括正负样本的置信度损失（后面解释如何区分正负样本）、正样本边界框位置损失、正样本处的类别损失（使用$L_2$损失，而未采用交叉熵）。样本边界框的置信度可以使用交并比(IoU)来计算，具体是真实边界框和预测边界框的交集和并集做除法。计算损失时，中心所在网格中IoU最大的作为正样本，而正样本候选区域外的所有预测边界框均视为负样本，并将它们的置信度设置为0。
- 前向推理：推理时，首先计算所有预测的边界框得分（边界框置信度乘最高类别置信度$c·max[p(c_1),...,p(c_{20})]$)，然后通过预定义的阈值去除得分低的边框，最后再利用非极大值抑制等方法剔除对同一目标的重复检测，最终得到检测结果。

|层|类型|特性|输出大小|
|--|----|----|-------|
|0|输入|-|3×(448×448)|
|1|卷积   |64:3×(7×7+3)/2|64×(224×224)|
|2|最大池化|(2×2)/2|64×(112×112)|
|3|卷积   |192:64×(3×3+1)/1|192×(112×112)|
|4|最大池化|(2×2)/2|192×(56×56)|
|5|串行卷积|128:192×(1×1)/1<br>256:128×(3×3+1)/1<br>256:256×(1×1)/1<br>512:256×(3×3+1)/1|128×(56×56)<br>256×(56×56)<br>256×(56×56)<br>512×(56×56)|
|6|最大池化|(2×2)/2|512x(28×28)|
|7|串行卷积|256:512×(1×1)/1<br>512:256×(3×3)/1<br>256:512×(1×1)/1<br>512:256×(3×3)/1<br>256:512×(1×1)/1<br>512:256×(3×3)/1<br>256:512×(1×1)/1<br>512:256×(3×3)/1<br>512:512×(1×1)/1<br>1024:512×(3×3)/1|256×(28×28)<br>512×(28×28)<br>256×(28×28)<br>512×(28×28)<br>256×(28×28)<br>512×(28×28)<br>256×(28×28)<br>512×(28×28)<br>512×(28×28)<br>1024×(28×28)|
|8|最大池化|(2×2)/2|1024x(14×14)|
|9|串行卷积|512:1024×(1×1)/1<br>1024:512×(3×3)/1<br>512:1024×(1×1)/1<br>1024:512×(3×3)/1<br>1024:1024×(3×3)/1<br>1024:1024×(3×3)/2|512×(14×14)<br>1024×(14×14)<br>512×(14×14)<br>1024×(14×14)<br>1024×(14×14)<br>1024×(7×7)|
|10|串行卷积|1024:1024×(3×3)/1<br>1024:1024×(3×3)/1|1024×(7×7)<br>1024×(7×7)|
|11|全连接|输入展平|4096|
|12|全连接|输出变形|1470=(7×7×30)|

**YOLOv4**

YOLOv4是一个重要里程碑，从它开始的若干版本均延续了它的设计思想。

# 其它应用领域

## AI安全

### 联邦学习

联邦学习指在保护多方不交换彼此数据的前提下进行协作学习，它由Google在2016年提出。

联邦学习中分为全局模型和本地模型。服务端会先给客户端下发一个初步(primary)权重，每个客户端都有初始数据。客户端的数据自己进行训练和测试，优化本地模型。参与训练的客户端将更新后的权重提交给服务端聚合。最终的评估以聚合后的全局模型为准。典型的聚合方式是联邦平均(FedAvg)，它对客户端上传的权重进行加权平均。

系统效率、安全性与可信度是联邦学习重要的研究方向。系统效率方面重点关注通信开销的降低，例如通过模型压缩技术减少传输数据量。系统安全与可信度则涵盖数据隐私保护、模型鲁棒性和系统公平性三个方面。

# 工程

## 模型示例

### 使用Scikit-Learn的预定义数据集和模型

#### 数据集

**datasets介绍**

sklearn的datasets包提供了常用数据集，包括三种类型。

- 小型经典数据集（以load开头的函数，无需下载）。包括`load_digits`（手写数据集）、`load_wine`（红酒数据集）、`load_iris`（鸢尾花数据集）、`load_linnerud`（健身数据集）、`load_diabetes`（糖尿病数据集）、`load_breast_cancer`（乳腺癌数据集）。
- 中大型数据集（以fetch开头的函数，需要下载）。包括新闻分类数据集、加州房价数据集和人脸识别数据集等。
- 算法生成数据集（以make开头的函数，通过程序生成数据）。

**鸢尾花数据集例子**

```python
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 包括150个样本，每个样本四个特征，分别是花萼长度、花萼宽度、花瓣长度、花瓣宽度
print(iris.data)
# 包括150个样本的标签
print(iris.target)
```

**分类数据集例子**

```python
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
```

#### 降维模型示例

PCA降维示例

```python
# 给出8个示例数据
X = np.array([[10, 1], [9, 0], [10, -1], [11, 0], [0, 9], [1, 10], [0, 11], [-1, 10]])
# 使用PCA算法降维到1维
pca = PCA(n_components=1)
Y = pca.fit_transform(X)
print("主成分系数矩阵:\n", pca.components_)
print("主成分方差比例:\n", pca.explained_variance_ratio_)
print("降维后的数据:\n", Y)
```

#### 分类模型示例

```python
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
```

#### 聚类模型示例

```python
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
```

#### 集成学习模型示例

```python
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
```

#### 隐马尔科夫模型示例

```python
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
```

#### 神经网络示例

sklearn提供了BernoulliRBM、MLPClassifier、MLPRegressor三种神经网络，下面是MLPClassifier的示例。

```
# 初始化自带数据集moons，并拆分成训练数据和测试数据
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
# 定义多层感知机分类器，使用SGD优化器
classifier = MLPClassifier(solver="sgd")
# 训练模型
classifier.fit(X_train, y_train)
# 在测试集上评分
print(classifier.score(X_test, y_test))
```

#### 经验

超参数对于模型的性能至关重要。在sklearn等框架下，一些时候任务的效果不好并不一定是模型选择本身有问题，很有可能是超参数未调整或设置的不当。 例如，随机森林模型经常使用的超参数有n_estimators（子树的数量）、max_features（最大特征数）和max_depth（最大深度）。

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