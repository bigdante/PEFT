# PEFT paper lists

我们选取超过100篇的PEFT相关论文。根据PEFT的不同思想，划分为4个模块，Addictive-代表添加了额外参数，Partial-代表在原始模型中选择部分参数，Reparameters-代表重参数方法，Hybrid-代表结合了前面3中方法的混合方法。
在每个大类中，我们将会具体展开每个更加细粒度的分类，并且介绍这些方法在NLP（Natural Language Processing）、CV(Computer Vision)，VLM(Vison Language Model)和Graph上的应用。

**CoRR (Computing Research Repository)** 是一个开放获取的电子预印本服务，主要用于存储和分发计算机科学领域的研究论文。它是arXiv.org的一部分，提供一个平台，供计算机科学家在正式发表之前分享他们的研究成果。CoRR 涵盖了计算机科学的各个子领域，包括人工智能、机器学习、计算机视觉、自然语言处理、理论计算机科学等。研究人员可以在 CoRR 上发布他们的论文，以便其他科学家在同行评审之前就能访问和评论这些研究成果。

**EMNLP (Empirical Methods in Natural Language Processing)** 是由国际计算语言学协会 (ACL) 组织的一年一度的顶级会议，主要关注自然语言处理领域的最新研究成果。根据中国计算机学会 (CCF) 的会议分类，EMNLP 被评为 B 级会议。

**Intelligent Systems in Accounting, Finance and Management**是一个专注于会计、金融和管理领域智能系统应用的学术期刊。该期刊发表了许多关于使用人工智能和计算方法解决财务和管理问题的研究论文。根据中国计算机学会 (CCF) 的会议和期刊分类系统，《Intelligent Systems in Accounting, Finance and Management》并未被列入CCF的期刊分类列表。因此，它在CCF的分类系统中没有特定的评级。

## [Content](#content)

<table>
<tr><td><a href="#Addictive">1. Addictive</a></td></tr> 
<tr><td><a href="#Partial">2. Partial</a></td></tr>
<tr><td><a href="#Reparameters">3. Reparameters</a></td></tr>
<tr><td><a href="#Hybrid">4. Hybrid</a></td></tr>

</table>
<!-- ** **. . '18. [paper]() -->

## [Addictive](#content)

### NLP

#### Adapter
1. **Parameter-Efficient Transfer Learning for NLP**. *Neil Houlsby,Andrei Giurgiu,Stanislaw Jastrzebski,Bruna Morrone,Quentin de Laroussilhe,Andrea Gesmundo,Mona Attariyan,Sylvain Gelly*. CoRR'2019. [paper](https://www.aminer.cn/pub/5c61606ae1cd8eae1501e0f5/parameter-efficient-transfer-learning-for-nlp)
2. **Exploring versatile generative language model via parameter-efficient transfer learning**. *Lin Zhaojiang,Madotto Andrea,Fung Pascale*. EMNLP'2020. [paper](https://www.aminer.cn/pub/5e8ef2ae91e011679da0f112/exploring-versatile-generative-language-model-via-parameter-efficient-transfer-learning)
3. **MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer**. *Pfeiffer Jonas,Vulić Ivan,Gurevych Iryna,Ruder Sebastian*. Intelligent Systems In Accounting,
 Finance &Management'2020. [paper](https://www.aminer.cn/pub/5eafe7e091e01198d3986542/mad-x-an-adapter-based-framework-for-multi-task-cross-lingual-transfer)
4. **Counter-Interference Adapter for Multilingual Machine Translation**.*Yaoming Zhu, Jiangtao Feng, Chengqi Zhao, Mingxuan Wang, Lei Li*.EMNLP'2021.[paper](https://aminer.cn/pub/619799ec91e011c8223730c6/counter-interference-adapter-for-multilingual-machine-translation)
5. **AdapterDrop - On the Efficiency of Adapters in Transformers**.*Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, Iryna Gurevych*.EMNLP'2021.[paper](https://www.aminer.cn/pub/5f92b9db91e011edb3573b95/adapterdrop-on-the-efficiency-of-adapters-in-transformers)
6. **Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters**.*Hongyu Zhao, Hao Tan, Hongyuan Mei*.EMNLP'2022.[paper](https://www.aminer.cn/pub/636482d890e50fcafdccb0cc/Tiny-Attention%20Adapter:%20Contexts%20Are%20More%20Important%20Than%20the%20Number%20of%20Parameters)
7. 
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


