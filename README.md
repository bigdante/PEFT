# PEFT paper lists

我们选取超过100篇的PEFT相关论文。安装PEFT的不同思想，划分为4个模块，Addictive-代表添加了额外参数，Partial-代表在原始模型中选择部分参数，Reparameters-代表重参数方法，Hybrid-代表结合了前面3中方法的混合方法。
在每个大类中，我们将会具体展开每个更加细粒度的分类，并且介绍这些方法在NLP（Natural Language Processing）、CV(Computer Vision)，VLM(Vison Language Model)和Graph上的应用。

**CoRR (Computing Research Repository)** 是一个开放获取的电子预印本服务，主要用于存储和分发计算机科学领域的研究论文。它是arXiv.org的一部分，提供一个平台，供计算机科学家在正式发表之前分享他们的研究成果。CoRR 涵盖了计算机科学的各个子领域，包括人工智能、机器学习、计算机视觉、自然语言处理、理论计算机科学等。研究人员可以在 CoRR 上发布他们的论文，以便其他科学家在同行评审之前就能访问和评论这些研究成果。

## [Content](#content)

<table>
<tr><td><a href="#Addictive">1. Addictive</a></td></tr> 
<tr><td><a href="#Partial">2. Partial</a></td></tr>
<tr><td><a href="#Reparameters">3. Reparameters</a></td></tr>
<tr><td><a href="#Hybrid">4. Hybrid</a></td></tr>

</table>

## [Addictive](#content)

### NLP

1. **Parameter-Efficient Transfer Learning for NLP**. Neil Houlsby et al. CoRR'19. [paper](https://www.aminer.cn/pub/5c61606ae1cd8eae1501e0f5/parameter-efficient-transfer-learning-for-nlp)
2. **Graph Attention Networks**. Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. ICLR'18. [paper](https://arxiv.org/abs/1710.10903)
3. **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**. NeuIPS'16. [paper](https://arxiv.org/abs/1606.09375)
4. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank**. Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann. ICLR'19. [paper](https://arxiv.org/abs/1810.05997) 
5. **Gated Graph Sequence Neural Networks**. Li, Yujia N and Tarlow, Daniel and Brockschmidt, Marc and Zemel, Richard. ICLR'16. [paper](https://arxiv.org/abs/1511.05493)
6. **Inductive Representation Learning on Large Graphs**. William L. Hamilton, Rex Ying, Jure Leskovec. NeuIPS'17. [paper](https://arxiv.org/abs/1706.02216)
7. **Deep Graph Infomax**. Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm. ICLR'19. [paper](https://openreview.net/pdf?id=rklz9iAcKQ)
8. **Representation Learning on Graphs with Jumping Knowledge Networks**. Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka. ICML'18. [paper](https://arxiv.org/abs/1806.03536)
9. **DeepGCNs: Can GCNs Go as Deep as CNNs?**. Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem. ICCV'19. [paper](https://arxiv.org/abs/1904.03751)
10. **DropEdge: Towards Deep Graph Convolutional Networks on Node Classificatio**. Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang. ICLR'20. [paper](https://arxiv.org/abs/1907.10903)

### CV

### VLM

### Gragh




## [Partial](#content)

### Most Influential

1. **Inductive Representation Learning on Large Graphs**. William L. Hamilton, Rex Ying, Jure Leskovec. NeuIPS'17. [paper](https://arxiv.org/abs/1706.02216)
2. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling**. Jie Chen, Tengfei Ma, Cao Xiao. ICLR'18. [paper](https://arxiv.org/abs/1801.10247)
3. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**. Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh. KDD'19. [paper](https://arxiv.org/abs/1905.07953)
4. **GraphSAINT: Graph Sampling Based Inductive Learning Method**. Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna. ICLR'20. [paper](https://arxiv.org/abs/1907.04931)
5. **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings**. Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec. ICML'21. [paper](https://arxiv.org/abs/2106.05609)
6. **Scaling Graph Neural Networks with Approximate PageRank**. Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann. KDD'20. [paper](https://arxiv.org/abs/2007.01570)
7. **Stochastic training of graph convolutional networks with variance reduction**. *Jianfei Chen, Jun Zhu, and Le Song.* ICML'18. [paper](https://arxiv.org/abs/1710.10568)
8. **Adaptive sampling towards fast graph representation learning**. Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. NeuIPS'18. [paper](https://papers.nips.cc/paper/2018/file/01eee509ee2f68dc6014898c309e86bf-Paper.pdf)
9. **SIGN: Scalable Inception Graph Neural Networks**. Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, Federico Monti. [paper](https://arxiv.org/abs/2004.11198)
10. **Simplifying Graph Convolutional Networks**. Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger. ICML'19. [paper](https://arxiv.org/abs/1902.07153)

### Recent SOTA

1. **GraphSAINT: Graph Sampling Based Inductive Learning Method**. Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna. ICLR'20. [paper](https://arxiv.org/abs/1907.04931)
2. **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings**. Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec. ICML'21. [paper](https://arxiv.org/abs/2106.05609)
3. **Deep Graph Neural Networks with Shallow Subgraph Samplers**. Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor Prasanna, Long Jin, Ren Chen. [paper](https://arxiv.org/abs/2012.01380)
4. **Scalable Graph Neural Networks via Bidirectional Propagation**. Ming Chen, Zhewei Wei, Bolin Ding, Yaliang Li, Ye Yuan, Xiaoyong Du, Ji-Rong Wen. NeuIPS'20. [paper](https://arxiv.org/abs/2010.15421)
5. **A Unified Lottery Ticket Hypothesis for Graph Neural Networks**. Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, Zhangyang Wang. ICML'21. [paper](https://arxiv.org/abs/2102.06790).
6. **Scaling Graph Neural Networks with Approximate PageRank**. Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann. KDD'20. [paper](https://arxiv.org/abs/2007.01570)
7. **Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training**. Chuxiong Sun, Hongming Gu, Jie Hu. [paper](https://arxiv.org/abs/2104.09376)
8. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**. Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh. KDD'19. [paper](https://arxiv.org/abs/1905.07953)
9. **GraphZoom: A Multi-level Spectral Approach for Accurate and Scalable Graph Embedding**. Chenhui Deng, Zhiqiang Zhao, Yongyu Wang, Zhiru Zhang, Zhuo Feng. ICLR'20. [paper](https://arxiv.org/abs/1910.02370)
10. **Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs**. 	Jialin Dong, Da Zheng, Lin F. Yang, Geroge Karypis. KDD'21. [paper](https://arxiv.org/abs/2106.06150)


## [Reparameters](#content)

### Most Influential

1. **Strategies for pre-training graph neural networks.** *Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande,  Leskovec Jure.* ICLR 2020. [paper](https://openreview.net/forum?id=HJlWWJSFDH)
2. **Deep graph infomax.** *Velikovi Petar, Fedus William, Hamilton William L, Li Pietro, Bengio Yoshua, Hjelm R Devon.* ICLR 2019. [paper](https://arxiv.org/abs/1809.10341)
3. **Inductive representation learning on large graphs.** *Hamilton Will, Zhitao Ying, Leskovec Jure.* NeurIPS 2017. [paper](https://arxiv.org/abs/1706.02216)
4. **Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization.** *Sun Fan-Yun, Hoffmann Jordan, Verma Vikas, Tang Jian.* ICLR 2020. [paper](https://arxiv.org/pdf/1908.01000.pdf)
5. **GCC: Graph contrastive coding for graph neural network pre-training.** *Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, Jie Tang.* KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168)
6. **Contrastive multi-view representation learning on graphs.** *Hassani Kaveh, Khasahmadi Amir Hosein.* ICML 2020. [paper](https://arxiv.org/abs/2006.05582)
7. **Graph contrastive learning with augmentations.** *Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen.* NeurIPS 2020. [paper](https://arxiv.org/abs/2010.13902)
8. **GPT-GNN: Generative pre-training of graph neural networks.** *Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, Yizhou Sun.* KDD 2020. [paper](https://arxiv.org/abs/2006.15437)
9. **When does self-supervision help graph convolutional networks?.** *Yuning You, Tianlong Chen, Zhangyang Wang, Yang Shen.* ICML 2020. [paper](https://arxiv.org/abs/2006.09136)
10. **Deep graph contrastive representation learning.** *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang.* GRL+@ICML 2020. [paper](https://arxiv.org/abs/2006.04131)

### Recent SOTA

1. **Graph Contrastive Learning Automated.** *Yuning You, Tianlong Chen, Yang Shen, Zhangyang Wang.* ICML 2021. [paper](https://arxiv.org/abs/2106.07594)
2. **Graph contrastive learning with adaptive augmentation.** *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang.* WWW 2021. [paper](https://arxiv.org/abs/2010.14945)
3. **Self-supervised Graph-level Representation Learning with Local and Global Structure.** *Minghao Xu, Hang Wang, Bingbing Ni, Hongyu Guo, Jian Tang.* ICML 2021. [paper](https://arxiv.org/pdf/2106.04113)
4. **Negative Sampling Strategies for Contrastive Self-Supervised Learning of Graph Representations.** *Hakim Hafidi, Mounir Ghogho, Philippe Ciblat, Ananthram Swami.* Signal Processing 2021. [paper](https://www.sciencedirect.com/science/article/pii/S0165168421003479)
5. **Learning to pre-train graph neural networks.** *Yuanfu Lu, Xunqiang Jiang, Yuan Fang, Chuan Shi.* AAAI 2021. [paper](http://shichuan.org/doc/101.pdf)
6. **Graph representation learning via graphical mutual information maximization.** *Zhen Peng, Wenbing Huang, Minnan Luo, Qinghua Zheng, Yu Rong, Tingyang Xu, Junzhou Huang.* WWW 2020. [paper](https://arxiv.org/abs/2002.01169)
7. **Structure-Aware Hard Negative Mining for Heterogeneous Graph Contrastive Learning.** *Yanqiao Zhu, Yichen Xu, Hejie Cui, Carl Yang, Qiang Liu, Shu Wu.* arXiv preprint arXiv:2108.13886 2021. [paper](https://arxiv.org/abs/2108.13886)
8. **Self-supervised graph representation learning via global context prediction.** *Zhen Peng, Yixiang Dong, Minnan Luo, Xiao-Ming Wu, Qinghua Zheng.* arXiv preprint arXiv:2003.01604 2020. [paper](https://arxiv.org/abs/2003.01604)
9. **CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction.** *Chengshuai Zhao, Shuai Liu, Feng Huang, Shichao Liu, Wen Zhang.* IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0517.pdf)
10. **Pairwise Half-graph Discrimination: A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks.** *Pengyong Li, Jun Wang, Ziliang Li, Yixuan Qiao, Xianggen Liu, Fei Ma, Peng Gao, Sen Song, Guotong Xie.* IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0371.pdf)


## [Hybrid](#content)

### Most Influential

1. **Representation Learning on Graphs with Jumping Knowledge Networks.** *Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka.* ICML 2018. [paper](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf)
2. **Deeper insights into graph convolutional networks for semi-supervised learning.** *Qimai Li, Zhichao Han, Xiao-ming Wu.* AAAI 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16098)
3. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank.** *Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann.* ICLR 2019. [paper](https://openreview.net/forum?id=H1gL-2A9Ym)
4. **DeepGCNs: Can GCNs Go as Deep as CNNs?** *Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem.* ICCV 2019. [paper](https://arxiv.org/abs/1904.03751)
5. **Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks.** *Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, Quanquan Gu.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-6006)
6. **DeeperGCN: All You Need to Train Deeper GCNs.** *Guohao Li, Chenxin Xiong, Ali Thabet, Bernard Ghanem.* arXiv 2020. [paper](https://arxiv.org/abs/2006.07739)
7. **PairNorm: Tackling Oversmoothing in GNNs.** *Lingxiao Zhao, Leman Akoglu.* ICLR 2020. [paper](https://openreview.net/forum?id=rkecl1rtwB)
8. **DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.** *Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang.* ICLR 2020. [paper](https://openreview.net/pdf?id=Hkx1qkrKPr)
9. **Simple and Deep Graph Convolutional Networks.** *Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, Yaliang Li.* ICML 2020. [paper](http://proceedings.mlr.press/v119/chen20v.html)
10. **Towards Deeper Graph Neural Networks.** *Meng Liu, Hongyang Gao, and Shuiwang Ji.* KDD 2020. [paper](https://dl.acm.org/doi/10.1145/3394486.3403076)

### Recent SOTA

1. **Towards Deeper Graph Neural Networks with Differentiable Group Normalization.** *Kaixiong Zhou, Xiao Huang, Yuening Li, Daochen Zha, Rui Chen, Xia Hu.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/33dd6dba1d56e826aac1cbf23cdcca87-Abstract.html)
2. **Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks.** *Yimeng Min, Frederik Wenkel, Guy Wolf.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/a6b964c0bb675116a15ef1325b01ff45-Abstract.html)
3. **Optimization and Generalization Analysis of Transduction through Gradient Boosting and Application to Multi-scale Graph Neural Networks.** *Kenta Oono, Taiji Suzuki.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/dab49080d80c724aad5ebf158d63df41-Abstract.html)
4. **On the Bottleneck of Graph Neural Networks and its Practical Implications.** *Uri Alon, Eran Yahav.* ICLR 2021. [paper](https://openreview.net/forum?id=i80OPhOCVH2)
5. **Simple Spectral Graph Convolution.** *Hao Zhu, Piotr Koniusz.* ICLR 2021. [paper](https://openreview.net/forum?id=CYO5T-YjWZV)
6. **Training Graph Neural Networks with 1000 Layers.** *Guohao Li, Matthias Müller, Bernard Ghanem, Vladlen Koltun.* ICML 2021. [paper](http://proceedings.mlr.press/v139/li21o.html)
7. **Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth.** *Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, Kenji Kawaguchi.* ICML 2021. [paper](http://proceedings.mlr.press/v139/xu21k.html)
8. **GRAND: Graph Neural Diffusion.** *Ben Chamberlain, James Rowbottom, Maria I Gorinova, Michael Bronstein, Stefan Webb, Emanuele Rossi.* ICML 2021. [paper](http://proceedings.mlr.press/v139/chamberlain21a.html)
9. **Directional Graph Networks.** *Dominique Beani, Saro Passaro, Vincent Létourneau, Will Hamilton, Gabriele Corso, Pietro Lió.* ICML 2021. [paper](http://proceedings.mlr.press/v139/beani21a.html)
10. **Improving Breadth-Wise Backpropagation in Graph Neural Networks Helps Learning Long-Range Dependencies.** *Denis Lukovnikov, Asja Fischer.* ICML 2021. [paper](http://proceedings.mlr.press/v139/lukovnikov21a.html)

