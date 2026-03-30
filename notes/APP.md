
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

## 应用进展

### 典型应用

由于近几年AI发展十分迅速，故在应用上的新概念层出不穷。本节旨在汇总一系列AI相关的应用领域和概念。为了叙述方便，这里简要的分为几个类别进行说明。

**AI基础处理能力**

- 自然语言处理：任务包括语言理解、机器翻译、命名实体识别、文本生成等。
- 机器视觉：任务包括图像分类、目标检测、语义分割、视频理解、视觉生成等。
- 跨媒体理解与生成：让计算机理解和生成多模态内容。典型任务包括图文检索、视觉问答、文生图、文生视频等。

**AI支撑技术**

- AI安全：包括数据投毒、对抗攻击、模型窃取、联邦学习、差分隐私等。
- 智能芯片：专用于AI计算的芯片，例如神经网络处理单元(NPU)、张量处理单元(TPU)、类脑芯片。

**AI在计算机学科内的应用**

- 神经渲染：神经渲染指利用神经网络来解决图形学的渲染问题。典型的任务包括神经辐射场(NeRF)、视频或三维内容生成。
- 自动软件工程：将AI用于软件开发与运维，如代码生成、缺陷检测、软件自动化测试等。

**AI在与其它学科的交叉应用**

- 生物信息学：生物信息学指利用应用数学、信息学、统计学和计算机科学的方法研究生物学的问题。典型的任务包括蛋白质结构预测、基因序列分析、药物分子设计等。
- 计算美学：将人类对美学的认知转化为计算机能处理的符号化表示，量化的描述人类对艺术品的风格以及情感的感知，并进一步实现计算机对美学风格的模仿和生成。典型任务包括AI绘画、艺术风格迁移、构图评分等。
- 情感识别：通过语音、表情、手势、姿态等信号建立可计算的情感模型，属于AI与心理学和生理学的交叉方向。

**AI在更广泛领域的应用**

- 推荐系统：利用推荐算法为用户提供个性化服务（电商、内容分发、广告等）。
- 智能创作：基于文本生成或编辑图像、视频等创意内容，是多模态生成在创意产业的具体应用。
- 自动驾驶：业界将自动驾驶分为5个等级，从L1辅助驾驶到L5完全自动驾驶。

### 应用进展案例

- 2019年MIT的CSAIL实验室发布Speech2Face，可通过输入6秒的声音片段来推断出说话者的大致容貌。
- 2021年MIT研究者受生物神经元启发涉及了一种liquid神经网络，用于模拟大脑动力学。
- 2022年5月清华大学团队研发出基于原形学习的空间循环神经网络模型，用于唇动信号学习，开发唇语解释系统。
- 2023年7月DeepMind发布Dramatron，可通过输入的文字提示自动生成结构化的剧本框架。
- 2024年MIT团队发表论文*SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning*，其基于多智能体与知识图谱推理，实现自动化科学发现与假设生成。

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

### 序列概率模型的评价

**困惑度**

困惑度是信息论概念。单一离散随机变量$x$的困惑度定义为$PPL(p)=2^{\mathbb{H}(p)}=2^{-\sum_xp(x)log_2p(x)}$，它通过衡量平均的等可能选择性来度量分布的不确定性。两个概率分布之间的困惑度由交叉熵来定义$PPL(p,q)=2^{\mathbb{H}(p,q)}=2^{-\sum_xp(x)log_2q(x)}$。

由于困惑度可以衡量两个分布之间的差异，因此可通过计算从真实分布中采样数据$x_1,x_2,...,x_N$在模型分布上的困惑度$PPL(p_\theta;D)=2^{\mathbb{H}(p_r,p_\theta)}=2^{-\frac{1}{N}\sum_{n=1}^{N}log_2p_\theta(x_n)}$，来衡量模型的优劣。

在序列生成任务中，已知数据集上的联合分布为$\prod\limits_{n=1}^{N}p_\theta(x_{1:T_n}^{(n)})=\prod\limits_{n=1}^{N}\prod\limits_{t=1}^{T_n}p_\theta(x_t^{(n)}|x_{1:t-1}^{(n)})$。则模型的困惑度定义为$PPL(p_\theta)=2^{-\frac{1}{\sum_{n=1}^{N}T_n}\sum_{n=1}^N\log_2p_\theta(x_{1:T_n}^{(n)})}=\left[\prod\limits_{n=1}^N\prod\limits_{t=1}^{T_n}p_\theta(x_t^{(n)}|x_{1:t-1}^{(n)})\right]^{-1/\sum_{n=1}^NT_n}$。

### 序列概率模型的学习问题

#### 曝光偏差问题

自回归模型在训练时的输入采用的是真实序列，而非模型自身产生的预测序列，这种方式称为教师强制。在某个中间时刻，尽管前序已经输出了一系列预测，但训练时的输入与这些预测值无关。

#### 高维输出计算效率问题

序列生成模型的输出层需要进行归一化Softmax，其复杂度与词汇表大小$|V|$直接相关。当词汇表很大时，将隐藏表示通过仿射变换映射到输出空间并计算 Softmax需要执行全矩阵乘法，复杂度为$O(|V|N_y)$。为缓解这一问题，有多种方法可以提高计算效率。

**使用短列表**

将词汇表分为高频词构成的短列表（由神经网络处理）和低频词构成的尾列表（由n-gram等传统方法处理）。预测时通过加权组合两部分的结果得到最终概率。这种方法的缺点是神经语言模型的优势仅限于高频词。

**分层Softmax**

另一个解决方法是分层分解概率，即建立词的类别以及类别的类别，以此将词汇表组织成树形结构。例如构建成平衡树或根据词频构建的最优二叉树，每个节点对应一个二分类器（例如逻辑回归），计算复杂度可从$O(|V|)$降至$O(\log|V|)$。推断时，一个词的概率定义为从根节点到叶节点路径上各节点概率的乘积。训练时，每个节点使用交叉熵损失进行优化。该方法的主要缺点是难以寻找在给定上下文时概率最大的候选词。

**重要采样**

在计算Softmax梯度时，大部分词的概率极低，对梯度的贡献很小，而枚举它们的计算成本很高。重要采样通过采样少量重要的词来近似完整梯度，从而避免枚举整个词汇表。

## 序列到序列模型

序列到序列模型将自然语言任务看做条件序列生成问题，目标是在给定序列$x_1,...,x_S$的情况下，生成另一个序列$y_1,...,y_T$，即估计条件概率$p(y_1,...,y_T|x_1,...,x_S)$。其主要用于机器翻译、文本摘要和对话系统等场景。与序列概率模型类似的，可以通过$p(y_1,...,y_T|x_1,...,x_S)\,$=$\,\prod\limits_{t=1}^Tp(y_t|y_1,...,y_{t-1},x_1,...,x_S)$将联合概率转换为单变量的条件概率。

序列到序列模型通常涉及多个模块。例如在统计机器翻译(SMT)架构中，一部分组件用于给出诸多候选词汇，另一部分（即语言模型）对候选评估。早期的模型主要采用n-gram，包括传统的回退n-gram和最大熵语言模型。后来的探索中，研究者使用MLP来建模条件概率分布并取得了不错的效果。但MLP的缺点是序列需要预处理为固定长度。为了解决这一局限性，随后的研究中使用RNN。2014年，Google团队提出了Seq2Seq模型，同期Bengio团队也提出了类似的编码器-解码器结构，共同推动了序列到序列学习的发展。

### 基于循环神经网络的序列到序列模型

使用两个循环神经网络（典型的是RNN，也可以是LSTM等）分别实现编码和解码。编码器将$x_1,...,x_S$通过$h_t^{en}=f_{en}(h_{t-1}^{en},e_{x_t})$编码为向量$u=h^{en}_S$。解码器使用另一个网络，通过$h_t^{de}=f_{de}(h_{t-1}^{de},e_{y_{t-1}})$以及$o_t=g(h_t^{de})$获得当前时刻词表中所有词的后验概率$o_t$（$g(·)$通常是Softmax）。这种序列到序列模型的缺点是编码$u$的容量有限，并且难以解决长程依赖问题。

### 基于注意力机制的序列到序列模型

#### 基于注意力机制的序列到序列模型

传统的基于循环神经网络的序列到序列模型，将整个输入序列压缩为一个固定长度的上下文向量。当句子较长时，这个向量难以保留所有信息。注意力机制通过让解码器在每一步动态聚焦于输入序列的不同部分，来解决这一问题。

基本流程是，编码器读取整个输入序列，生成每个位置对应的隐状态。解码器在生成每个输出词时，根据当前状态计算一个注意力分布，决定重点关注输入序列的哪些位置。而后将加权后的上下文向量与当前输入结合，生成下一个输出词。

形式化描述为，在解码器每个$t$步中，将上一步的隐状态$h_{t-1}^{de}$用作查询向量，从所有输入序列的隐状态$H^{en}$中选择相关的信息。也就是计算$c_t=att(H^{en},h_{t-1}^{de})=\sum_{i=1}^{S}softmax(s(h_i^{en},h_{t-1}^{de}))h_i^{en}$。将$c_t$作为第$t$步的输入计算$t$步的隐状态$h_t^{de}=f(h_{t-1}^{de},e_{y_{t-1}},c_t)$。

#### 基于自注意力机制的Transformer模型

以循环神经网络为主干网络的序列到序列模型有两个主要缺点：一是存在长程依赖问题、二是训练时不能并行计算序列的全部时刻。因此后续出现了以Transformer为代表的基于自注意力的序列到序列模型。

**位置编码**

由于自注意力机制本身忽略了序列$x_{1:T}$中每个$x_t$的顺序，因此Transformer通过位置编码为每个词嵌入添加位置信息。设编码器第一层的输入表示为$H_0∈\mathbb{R}^{T×D}$，则其第$t$行由$h_{0,t}=e_{x_t}+p_t$计算。其中，$e_{x_t}$是词$x_t$的词嵌入表示，$p_t$为位置$t$的向量表示。$p_t$既可以通过学习得到，也可以指定为$p_t^{(2i)}=\sin(t/10000^{2i/D})$、$p_t^{(2i+1)}=cos(t/10000^{2i/D})$。其中，$D$为嵌入维度，$i$为维度索引。

**多头自注意力**

- 自注意力：对于向量序列$H=h_{1:T}$，首先分别计算$Q=W_Q·H$、$K=W_K·H$、$V=W_V·H$，然后再计算自注意力$SelfAtt(Q,K,V)=softmax(\frac{QK^T}{\sqrt{D_K}})·V$。
- 多头自注意力：使用多头自注意力在多个不同的投影空间中捕捉不同的交互信息$MultiHead(H)=[...:SelfAtt_m(Q_m,K_m,V_m):...]·W_O$。

**模型结构**

- Transformer整体分为编码器和解码器两个部分。
- 训练时编码器输入序列，解码器输入真实标注（以起始标记`<start>`开始）和编码器输出、输出预测标注。
- 预测时编码器输入序列，解码器输入起始状态`<start>`和编码器输出、依次输出预测下一个标注直到`<eos>`。
- 编码器/解码器的输入都需要先通过词嵌入和位置编码，然后分别通过若干个编码器/解码器模块，所有模块的输入和输出大小均保持一致的$(N,T,D)$。编码器无论通过多少个模块，只有一个最终输出供解码器使用。
- 解码器最后还需要通过线性层和Softmax来综合得到的信息。
- 编码器模块包括两部分：自注意力模块、逐位置的前馈神经网络。
- 解码器模块包括三部分：掩码自注意力模块、编码器到解码器交叉注意力模块、逐位置的前馈神经网络。
- 自注意力模块：使用多头自注意力对输入计算$SelfAtt(Q,K,V)$，然后将多个头得到的结果进行向量拼接，再通过一个线性层（为了保持模块的输入输出一致）。另外，自注意力模块使用残差链接，且线性层结果进行了层标准化。
- 逐位置的前馈神经网络：是一个两层的线性神经网络，第一层使用ReLU或GELU作为激活函数，第二层不使用激活函数。这部分同样使用残差连接，且进行了层标准化。
- 掩码自注意力模块：就是在计算打分$S=QK^T$时，将矩阵上三角区置为$-\infty$，以确保打分只在已知的序列中进行，防止解码器在生成当前位置时看到未来位置的信息。
- 编码器到解码器交叉注意力模块：$K$和$V$从编码器最终的输出获得，而$Q$就是解码器前一模块的输出。
- 不使用偏置：由于使用了层归一化，线性层通常省略偏置项以减少参数量。

**层标准化**

Transformer使用层标准化。不同于批标准化，层标准化针对的是词向量，也就是Reshape到$(N*T,D)$之后沿着$D$而不是沿着$B$进行标准化。层标准化的方法是用每一样本值减去均值再除以标准差，然后再做一个线性变换，即$\hat{x}=γ\frac{(D_i-D_{mean})}{\sqrt{D_{var}}}+β$。原始Transformer中层标准化位于残差连接的外部，在一些任务中如果将标准化层置于内部可能会出现网络拟合能力减弱，损失居高不下的问题。

**正则化**

Transformer的正则化策略主要体现在两个方面。一是编码器嵌入层、解码器嵌入层、以及解码器输出前的线性变换层进行了权重绑定。二是每个子层（注意力层、前馈网络）的输出、在残差连接与层归一化之前，以及嵌入向量与位置编码求和之后，分别使用了0.1的Dropout。

**超参数**

Transformer的权重初始化可以使用He或Xavier方法。在学习率方面，Transformer采用了动态调整策略，由于参数量很大，训练初期往往需要较小的学习率（甚至小到0.00001）进行线性预热，有助于训练初期稳定性。相比较其它参数较小的模型来说，在资源受限的计算机上训练Transformer时，批大小也不易设置过大。因为其参数量本身就很大，而显存空间有限，这意味着可用于批训练数据的显存空间更小。同时，也正因批量减小，批和批之间的样本差异性更大，若选择了更大的学习率，会让每次的反向传播极不稳定。这也是需要先使用较小学习率进行预热的一个原因。

## 大规模语言模型

### 大语言模型的评估

大语言模型的评估不仅涉及技术本身，更需要考虑社会层面的影响。另外，由于大语言模型并非专为特定任务设计，因此针对单一任务的评估方法并不适用评估大语言模型。

**评估难点**

大语言模型的评估有三个主要难点。其一是文本生成的结果具有多样性，也就是同样的语义有多种不同的表达方式。因此，典型的评估过程多是人工化和半自动的，并且成本较高。对文本生成任务的自动化评估仍然是研究重点。其二是大语言模型是通用模型，涵盖了语言理解、逻辑推理、语言生成等诸多任务，因此如何构造适合的测试集也是关键问题。其三是由于大模型每个阶段的训练目标不同，因此需要针对不同阶段设计专门的评估方法。

### 大语言模型应用

大模型的应用场景包括：内容创作和生成、聊天机器人、机器翻译、信息抽取、代码生成、搜索与推荐、决策支持等。比较著名模型的有：ChatGPT(通用聊天机器人)、Deepseek(国内通用聊天机器人)、Cursor(智能编程辅助模型)。

# 机器视觉

## 预处理

使用大型数据集和大型模型训练时，预处理并不总是必要的。下面是一些典型的预处理：

- 标准化：使像素在相同且合理的范围内（比如$[0,1]$），有时甚至是唯一必要的预处理。
- 对比度标准化：例如全局对比度标准化和局部对比度标准化。
- 缩放或裁剪图像：对自适应模型（例如全卷积网络等可接受任意尺寸输入的模型）非必须操作。
- 数据集增强：对训练集的预处理（例如几何变换、颜色随机扰动），是减少泛化误差的常见方法。

**全局对比度标准化**

全局对比度标准化(Global Contrast Normalization，GCN)对数据集的每张图像先减去像素平均值，再除以像素标准差（通常缩放到标准差为1）。但这种方法对于对比度极低的图像会放大噪声或压缩伪影，以及对于同时含有大量暗区和亮区的图像效果也不理想。改进的办法是引入正则化项来平衡估计的标准差，或者强制约束分母大于某个常数。

花书P276页公式中的分母可以理解为L2范数的重新缩放，也就是L2与标准差成比例。因此可以把GCN理解为把像素取值范围限定到一个超球的表面。

**局部对比度标准化**

LCN在每个像素的局部邻域内进行归一化，常见的实现方式有多种。比如可以是每个像素减去邻域平均值并除以邻域标准差，或者以像素为中心的高斯权重的加权平均和加权标准差，或跨颜色通道组合。LCN可通过卷积实现且可微分，因此可作为神经网络的一层进行训练。同样，LCN也需要引入正则化来避免除以0的情况。

## 目标检测

### 概述

目标检测是机器视觉领域的重要主题，其主要是识别图像中的目标类别并定位目标在图像中的位置。传统的目标检测通常包括两个阶段，首先从图像中提取人工视觉特征(例如HOG)，然后再将其输入分类器(例如SVM)进而给出检测结果。这种两阶段模式尽管精度很高但工作效率较慢。

2014年R-CNN开启了基于深度学习的视觉目标检测，它同样以两阶段模式工作，先使用选择性搜索算法从图像中提取若干感兴趣的区域(RoI)，然后使用CNN分别处理每一个区域的特征，最后用SVM实现分类。但由于RoI较多导致工作耗时，后续出现了改进的Fast R-CNN（通过共享卷积提速）、Faster R-CNN。

2015年YOLO(You Only Look Once)发布，它打破了传统的阶段检测模式，而只用一个网络实现端到端的目标检测，其单阶段的工作模式使得其工作非常高效（在TITAN X的GPU上可以每秒处理40张图像）。

其它具有代表性的单阶段框架包括SSD、RetinaNet、FCOS、DETR。由于ImageNet竞赛具有图像分类领域最全面的数据集，受迁移学习思想启发，目标检测模型通常使用在ImageNet上预训练的分类网络作为骨干网络进行微调。但2019年的研究表明，在足够的数据和训练策略下，从零开始训练也能达到与预训练模型相当甚至更好的性能。

目标检测领域主要的数据集包括PASCAL VOC和MS COCO等。评估目标检测的主要指标包括：平均精度(mAP)表示所有类别AP的平均值，以及每秒检测张数(FPS)。

**深度学习下的目标检测网络框架**

深度学习下，目标检测的网络框架主要包括三部分：主干网络用于提取图像的高层特征并降维，颈部网络（典型代表是FPN）用于进一步整合特征，检测头用于提取类别信息和位置信息并输出结果。

主干网络常使用VGG(主要是VGG-16)、ResNet(包括ResNet-50和ResNet-101)、DarkNet(包括DarkNet-19、DarkNet-53)、MobileNet(包括其v1、v2和v3)、ShuffleNet。

颈部网络主要结构包括特征金字塔网络(FPN)和空间金字塔池化(SPP)等。FPN基于网络中不同大小的特征图包含的信息不同，浅层包含更多位置信息且便于检测小物体，而深层网络包含更多语义信息且便于检测大物体。因此FPN通过横向连接自顶向下将深层特征不断融合到浅层特征中。

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

联邦学习指在保护多方不交换彼此数据的前提下进行协作学习，它由Google在2016年提出。模型聚合是联邦学习的主要技术手段，它将各端侧的模型通过知识汇聚的形式聚合到一个全局模型中。

联邦学习中分为全局模型和本地模型。服务端会先给客户端下发一个初步(primary)权重，每个客户端都有初始数据。客户端的数据自己进行训练和测试，优化本地模型。参与训练的客户端将更新后的权重提交给服务端聚合。最终的评估以聚合后的全局模型为准。典型的聚合方式是联邦平均(FedAvg)，它对客户端上传的权重进行加权平均。

系统效率、安全性与可信度是联邦学习重要的研究方向。系统效率方面重点关注通信开销的降低，例如通过模型压缩技术减少传输数据量。系统安全与可信度则涵盖数据隐私保护、模型鲁棒性和系统公平性三个方面。数据隐私保护涉及差分隐私、同态加密等技术，系统公平性与可信度包括公平性、透明度、可解释性和模型水印等。以差分隐私为例，它通过在数据中添加干扰噪声，在保留整体统计学习特征的前提下掩盖个体特征，使得攻击者无法通过模型输出推断出具体个体的隐私信息。

# 工程

## 自动梯度计算

数值微分、符号微分和自动微分都可用于自动梯度计算。

数值微分通过中心差分公式近似计算导数$f'(x)=\lim\limits_{Δx\to{0}}\frac{f(x+Δx)-f(x-Δx)}{2Δx}$。当参数为向量时，需要逐个维度施加扰动才能得到完整的梯度。若参数数量为 $N$，每次前向传播的计算复杂度为 $O(N)$，则计算一次完整梯度的总复杂度为 $O(N^2)$。

符号微分通过解析方式对数学表达式进行求导，利用链式法则直接推出导数的符号表达式。 因此这种方法往往需要时间进行动态编译，甚至是需要设计专门的语言来实现符号解析，难于调试。

### 自动微分

反向传播是一种利用链式法则计算函数梯度的算法。在工程中，一种高效的实现方式是自动微分。尽管函数的形式会比较复杂，但自动微分的底层都能将其拆成一系列基本运算，比如（求和、乘积，以及初等函数），然后按照链式求导法则自动计算复合函数的导数。

**前向模式与反向模式**

按照计算导数的顺序，自动微分可以分为前向模式和反向模式。前向模式需要对每一个输入维度进行一遍正向遍历，反向模式则相应地需要对每一个输出维度进行一遍反向遍历。在深度学习中，损失函数通常是标量，因此只需要一次反向传播就能计算所有参数梯度，因此效率更高。

**静态计算图和动态计算图**

实现自动微分需要构建计算图，它可将计算的过程表示为有向无环图。计算图按照构建方式分为静态计算图和动态计算图。静态计算图在编译时构建，可进行全局优化和并行调度，性能更好但灵活性差。动态计算图在程序运行时构建、灵活性高且便于调试，但是运行时开销更大。

**框架实现**

- Define and Run：在计算之前定义计算图，再编译执行。TensorFlow 1.0和Caffe等早期框架使用这种机制，优点是性能好，但是相当于用一种新的约定语言建立计算图。
- Define by Run：在计算中动态生成计算图，前向执行的过程中同时建立反向图。TensorFlow 2.0和PyTorch都是基于这种机制。优点是完全与python语法融合。

记正向传播时的输入信号为$x$（若有多个操作数则为$x_1,x_2,...$）、输出信号为$y$，反向传播时的上游梯度$dy$、输出信号为$dx$，则根据链式法则反向传播上游梯度的策略如下。

- 加法节点直接传递上游梯度$dx_i←dy×1$。
- 乘法节点将上游梯度乘以另一个输入的值$dx_1←x_2×dy,\,dx_2←x_1×dy$。
- ReLU节点仅向正向传播时输入信号大于0的节点原封不动的传播上游梯度，即$dx←dy[x>0]$。
- Sigmoid节点的反向传播为$dx←y×(1-y)×dy$。
- 由于分支节点会被复制到多个下游节点，因此需要对上游梯度求和，即$dx←\sum\,dy_i$。

### 自动微分代码示例

```python
import numpy as np
from typing import Optional, List
# 定义函数类，相当于计算图的边
class Function:
    def __init__(self):
        # 一个函数对应一个输入变量和一个输出变量
        self.input: Optional[Variable] = None
        self.output: Optional[Variable] = None
    # 重载调用运算符，实现前向传播
    def __call__(self, input: 'Variable') -> 'Variable':
        x = input.get_value()
        y = self.forward(x)
        # 创建输出变量（值是y，创建者是本函数）
        output = Variable(y, self)
        # 保存输入和输出变量
        self.input = input
        self.output = output
        # 返回输出变量
        return output
    # 求自变量在某一点的函数值（抽象方法）
    def mapping(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    # 求自变量在某一点的梯度值（抽象方法）
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    # 前向传播，调用映射函数
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.mapping(x)
    # 反向传播
    def backward(self) -> 'Variable':
        # 先求函数在当前输入变量上的梯度
        grad = self.gradient(self.input.get_value())
        # 再把当前输出变量的梯度带上
        grad *= self.output.get_gradient()
        # 将梯度更新给当前输入变量
        self.input.set_gradient(grad)
        # 返回当前输入变量
        return self.input
# 定义变量类，每一个变量对应计算图的一个节点
class Variable:
    # 反向传播前，梯度先置为1.0，相当于默认把该节点当成最后一个节点
    def __init__(self, value: np.ndarray, creator: Optional[Function] = None):
        self.value = value                  # 正向的值
        self.gradient = np.array(1.0)       # 反向的梯度
        self.creator = creator              # 创建节点的函数
    def set_value(self, value: np.ndarray) -> None:
        self.value = value
    def get_value(self) -> np.ndarray:
        return self.value
    def set_gradient(self, gradient: np.ndarray) -> None:
        self.gradient = gradient
    def get_gradient(self) -> np.ndarray:
        return self.gradient
    # 反向传播（计算图的反向传播是从某个变量节点开始的）
    def backward(self) -> None:
        # 从该变量的创建者（函数）开始
        stack: List[Function] = [self.creator]
        # 只要栈中仍有函数，就继续
        while stack:
            # 取出一个函数
            function = stack.pop()
            # 调用函数的反向传播方法，得到上一个变量
            variable = function.backward()
            # 只要变量还有创建者（函数），就继续放入栈中进行下一轮传播
            if variable.creator:
                stack.append(variable.creator)
# 具体函数实现：平方函数
class Square(Function):
    def mapping(self, x: np.ndarray) -> np.ndarray:
        return x ** 2
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x
# 具体函数实现：指数函数
class Exp(Function):
    def mapping(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
# 使用示例
if __name__ == "__main__":
    # 创建三个函数
    A = Square()
    B = Exp()
    C = Square()
    # 创建一个输入变量
    x = Variable(np.array(0.5))
    # 依次通过三个函数的处理得到y
    h = A(x)      # h = x^2, dh = 2x
    z = B(h)      # z = exp(h) = exp(x^2), dz = exp(x^2) * 2x
    y = C(z)      # y = z^2 = exp(x^2)^2, dy = 2exp(x^2) * exp(x^2) * 2x
    # 调用输出节点的反向传播
    y.backward()
    # 结果就在输入节点的梯度值中
    print(f"x= {x.get_value()}, dx={x.get_gradient()}");
```

## 大规模深度学习

### 并行计算

大规模深度学习中，并行策略可大致分为三类。

**数据并行**

在小批量随机梯度下降中，单纯增加批大小通常不会得到线性的性能提高。为了支持大批量训练并提升吞吐量，可以将数据切分到多个计算节点上并行处理，每个节点持有完整的模型副本，这种方法称为数据并行。此时，各节点独立完成前向和反向计算，仅在参数更新时进行全局通信（如梯度同步）。

**模型并行**

将模型切分到多个计算节点，每个节点持有模型的一部分，共同处理同一批数据。主要用于突破单卡显存限制，分为两种常见形式。

- 张量并行：将单个张量操作（如矩阵乘法）的参数切分到多个节点，计算过程中通过通信协同完成运算，主要用于单层参数量多的情况。
- 流水线并行：模型的不同组件（层）分配到不同的计算节点，并按模块逐级计算，主要用于深层模型的情况。

**专家并行**

可视为模型并行的一种特殊形式。在混合专家模型(MoE)中，将不同的专家子网络分布到不同计算设备上。

### 参数同步

**异步梯度下降**

在数据并行中，由于多个节点独立计算梯度，因此要考虑如何进行参数的更新。一种解决办法是采用异步梯度下降，它可以减小同步等待开销。异步梯度下降的实现中，处理器核之间共享参数内存，读取和更新参数时可均不上锁。虽然这一过程可能出现“过时梯度”问题（即用旧参数计算的梯度更新已变化的新参数），但整体上仍可加速收敛。在实践中，参数可以通过专门的服务器管理，这种架构称为Parameter Server。

### 模型压缩

在生产场景中，模型的推理开销通常比训练开销更受重视，特别是在边缘设备上（如手机和嵌入式设备）。模型压缩是一类减小推理开销的重要技术，它用更小的模型代替原始模型，来降低存储模型的空间开销，或降低推理上的空间和时间开销。常见的模型压缩方法包括：

- 量化：将模型参数从高精度（如FP32）转换为低精度（如INT8），减少显存占用和计算量。
- 剪枝：移除模型中不重要的连接或通道，减小模型规模。
- 知识蒸馏：用大模型（教师）指导小模型（学生）训练，将知识迁移到更紧凑的模型中。

**知识蒸馏**

大模型通常具有过参数化的特点，其参数量远超任务所需，以学习丰富的特征表示。知识蒸馏通过大模型生成软标签（如类别概率分布），可以获得比原始标签更丰富的信息（如类别间的相似关系）。利用这些软标签训练一个小模型，可以在保持较高精度的同时大幅降低推理开销。

### 条件计算

神经网络中，模型的一些部分可能只跟输入的局部信息有关。这进一步启示了是否可以让神经网络增加类似于数据处理系统中的动态结构，使得根据输入内容，动态的决定哪一模块参与运算，这种引入动态结构的思想也称为条件计算。决策树可以看作动态结构的一个典型例子。

**级联分类器**

当分类的目标是罕见对象或事件时（例如恶意网络流量），可以采用级联分类器策略。它使用多个分类器串联，靠近输入端的分类器采用低容量、高召回率的模型快速过滤掉大部分负样本（确保样本出现时不会漏报），靠近输出的分类器采用更高精度的模型。推理时，任意一个分类器拒绝则直接丢弃，无需继续计算。

**多路选通器**

训练一个门控网络作为选通器，根据输入动态决定如何调用多个网络，即混合专家网络的思想。

- 软混合专家模型：门控网络输出权重，对所有专家的输出进行加权组合。
- 硬混合专家模型：门控网络仅选择一个专家网络进行计算（通常通过Top-K或概率采样实现）。

**开关**

神经单元可以根据具体情况从不同单元接收数据。注意力机制的本质也是一种动态开关结构，它通过计算查询与键的相似度得到注意力权重，对不同位置的值进行加权聚合，从而动态聚焦与当前任务最相关的信息。

### 算力产品

AI领域中，算力通常指基于GPU等加速芯片的并行计算能力。

#### 厂商和产品

**硬件算力**

国际厂商主要是NVIDIA和AMD，国内主要厂商有华为、寒武纪等。一些国产厂商的代表的产品有：

- 华为：昇腾910A/910B
- 寒武纪：思元370/590
- 摩尔线程：S3000/S4000
- 百度昆仑芯：R100/R200

**云算力**

一些云计算公司也推出了云算力平台，例如Amazon AWS、Microsoft Azure、腾讯云、阿里云、华为云，以及国内电信运营商的云服务。

#### 主要性能指标

- 显存：单位GB，普通显卡几至几十GB，多卡聚合甚至上TB。
- FLOPS：每秒执行浮点运算次数。具体还可分为半精度FP16、单精度FP32、双精度FP64。如今已经进入“T”时代了。
- TFLOPS：每秒执行浮点运算万亿次数。
- CUDA核心数：仅对于英伟达系列。CUDA核心是英伟达GPU中标量运算单元，是衡量并行计算能力的指标之一。

**一些行业术语**

- 魔改：泛指非官方的修改。
- 亮机卡：特指性能很低、仅能提供显示输出的显卡。
- 算力卡：指通常无视频输出接口、专为计算设计的加速卡。
- 1bit参数：极端的模型压缩技术，把权重量化到1bit。

#### 英伟达主要系列

下表是截止笔记整理时NVIDIA的典型产品系列。

| 系列 | 面项人群/任务 |主要型号|
|-----|---------|-------|
|MX     | 低端/轻薄本办公 | MX150 MX450     |
|GTX    | 中端/家庭和游戏 | GTX 1080Ti      |
|RTX    | 高端/游戏和计算 | RTX 3080/4090   |
|P      | 专业图形        | P1000/2000/5000 |
|A      | 数据计算        | A100            |
|特斯拉 | 数据计算        |                 |
|TITAN  | 半专业卡        |                 |

#### NVIDIA典型产品

下表是截止笔记整理时NVIDIA的典型产品。

| Product    | Single Flops | Double Flops |
|------------|--------------|--------------|
| RTX 3090   | 35.6T        | 0.56T        |
| RTX 3090Ti | 40T          | 0.63T        |
| RTX 4090   | 82.6T        | 1.29T        |
| V100       | 15.7T        | 7.8T         |
| A100       | 19.5T        | 9.7T         |

#### 一些已知模型的算力要求

下表列出的是一些主要模型的参数规模、以及若在单个NVIDIA V100上所需的估算训练时间。

|模型名称|参数规模|发布日期|数据集|训练时间|
|-------|-------|-------|-----|-------|
|GPT    |1.17亿 |2018年06月|4G文本  |3天   |
|BERT   |3.4亿  |2018年10月|16G文本 |数十天|
|GPT-2  |15亿   |2019年02月|40G文本 |数百天|
|GPT-3  |1750亿 |2020年07月|570G文本|数十年以上|

## 机器学习开发工具

### 语言

目前最适于机器学习开发的语言是Python，传统的Matlab也有一定市场。

### 库和框架

**库**

- NumPy：Python的数值计算扩展库，提供了对张量的各类运算。
- CuPy：Python中提供CUDA加速的张量操作扩展库，大部分的接口与NumPy保持一致。使用CuPy的前提是需要安装有带有CUDA核心的NVIDIA显卡（硬件和驱动程序），以及需要安装官方CUDA Toolkit工具包，之后再安装CuPy包。
- Scikit-learn：基于SciPy的传统机器学习库。稍需做说明的是，其HMM部分已经弃用，可使用hmmlearn（曾是scikit-learn项目的一部分，但现已独立成单独的包）。
- Pylearn2：基于Theano（Theano是一个用于数值计算的Python库）的机器学习库，目前已不维护。

**框架**

- Tensorflow：由Google公司开发，在学术研究中占比已经逐步减少了。
- PyTorch：由Facebook、NVIDIA、Twitter等公司开发，前身为Lua语言的Torch，是目前学术研究的主流框架。

**一些早期的库**

- Caffe：由伯克利人工智能研究小组和伯克利视觉和学习中心开发，Caffe2已经并入PyTorch。
- Keras：最初由François Chollet于2015年开发，后来被集成到TensorFlow中，作为基于Tensorflow的模块化神经网络库。

### 产品

- kaggle：kaggle是一个在线机器学习平台，官网https://www.kaggle.com。可以在平台中检索数据集、参加比赛。特别的是Kaggle提供了一个在线的Python环境，并且内置了常用的库，可以直接在线编写代码。
- ECS：阿里云的弹性计算服务，提供云服务器租用和GPU硬件加速服务。

## 模型示例

### 使用NumPy实现简单模型

**线性回归示例**

```python
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
```

**感知机示例**

```python
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
```

**逻辑回归示例**

```python
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
```

**Softmax回归示例**

```python
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
```

**支持向量机示例**

```python
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
```

### 使用Numpy实现全连接神经网络

```python
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
```

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