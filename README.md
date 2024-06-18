# PEFT paper lists

我们选取超过100篇的PEFT相关论文。根据PEFT的不同思想，划分为4个模块，Addictive-代表添加了额外参数，Partial-代表在原始模型中选择部分参数，Reparameters-代表重参数方法，Hybrid-代表结合了前面3中方法的混合方法。
在每个大类中，我们将会具体展开每个更加细粒度的分类，并且介绍这些方法在NLP（Natural Language Processing）、CV(Computer Vision)，VLM(Vison Language Model)和Graph上的应用。

**CoRR (Computing Research Repository)** 是一个开放获取的电子预印本服务，主要用于存储和分发计算机科学领域的研究论文。它是arXiv.org的一部分，提供一个平台，供计算机科学家在正式发表之前分享他们的研究成果。CoRR 涵盖了计算机科学的各个子领域，包括人工智能、机器学习、计算机视觉、自然语言处理、理论计算机科学等。研究人员可以在 CoRR 上发布他们的论文，以便其他科学家在同行评审之前就能访问和评论这些研究成果。

**EMNLP (Empirical Methods in Natural Language Processing)** 是由国际计算语言学协会 (ACL) 组织的一年一度的顶级会议，主要关注自然语言处理领域的最新研究成果。根据中国计算机学会 (CCF) 的会议分类，EMNLP 被评为 B 级会议。

**Intelligent Systems in Accounting, Finance and Management** 是一个专注于会计、金融和管理领域智能系统应用的学术期刊。该期刊发表了许多关于使用人工智能和计算方法解决财务和管理问题的研究论文。根据中国计算机学会 (CCF) 的会议和期刊分类系统，《Intelligent Systems in Accounting, Finance and Management》并未被列入CCF的期刊分类列表。因此，它在CCF的分类系统中没有特定的评级。

**NAACL(North American Chapter of the Association for Computational Linguistics)** 是自然语言处理领域的顶级学术会议之一。该会议涵盖了自然语言处理和计算语言学的各个方面，包括但不限于机器翻译、文本生成、语义分析、语音识别和自然语言理解等领域。根据中国计算机学会 (CCF) 的会议和期刊分类系统，NAACL 被评为 A 级会议。这是最高级别的分类，表示该会议在计算机科学和自然语言处理领域具有很高的学术影响力和声望。

**arXiv** 是一个开放获取的学术预印本平台，涵盖了物理、数学、计算机科学、定量生物学、定量金融和统计学等多个领域的研究论文。arXiv 允许研究人员在正式发表前分享他们的研究成果，以促进学术交流和讨论。尽管 arXiv 上的文章在上传时并未经过正式的同行评审，但许多重要的研究成果首先在 arXiv 上发布，然后才在同行评审的期刊或会议上发表。arXiv 没有特定的 CCF 分类等级。

**NIPS (Neural Information Processing Systems)**，现称为 NeurIPS (Conference on Neural Information Processing Systems)，是机器学习和计算神经科学领域的顶级学术会议之一。该会议自1987年开始举办，涵盖了人工智能、机器学习、统计学、计算神经科学和其他相关领域的前沿研究。根据中国计算机学会 (CCF) 的会议和期刊分类系统，NeurIPS 被评为 A 级会议。这是最高级别的分类，表明该会议在学术界具有很高的影响力和权威性。

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
1. **Parameter-Efficient Transfer Learning for NLP**. *Neil Houlsby,Andrei Giurgiu,Stanislaw Jastrzebski,Bruna Morrone,Quentin de Laroussilhe,Andrea Gesmundo,Mona Attariyan,Sylvain Gelly*. **CoRR'2019**. [paper](https://www.aminer.cn/pub/5c61606ae1cd8eae1501e0f5/parameter-efficient-transfer-learning-for-nlp)
2. **Exploring versatile generative language model via parameter-efficient transfer learning**. *Lin Zhaojiang,Madotto Andrea,Fung Pascale*. **EMNLP'2020**. [paper](https://www.aminer.cn/pub/5e8ef2ae91e011679da0f112/exploring-versatile-generative-language-model-via-parameter-efficient-transfer-learning)
3. **MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer**. *Pfeiffer Jonas,Vulić Ivan,Gurevych Iryna,Ruder Sebastian*. **Intelligent Systems In Accounting,
 Finance &Management'2020**. [paper](https://www.aminer.cn/pub/5eafe7e091e01198d3986542/mad-x-an-adapter-based-framework-for-multi-task-cross-lingual-transfer)
4. **Counter-Interference Adapter for Multilingual Machine Translation**.*Yaoming Zhu, Jiangtao Feng, Chengqi Zhao, Mingxuan Wang, Lei Li*.**EMNLP'2021**.[paper](https://aminer.cn/pub/619799ec91e011c8223730c6/counter-interference-adapter-for-multilingual-machine-translation)
5. **AdapterDrop - On the Efficiency of Adapters in Transformers**.*Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, Iryna Gurevych*.**EMNLP'2021**.[paper](https://www.aminer.cn/pub/5f92b9db91e011edb3573b95/adapterdrop-on-the-efficiency-of-adapters-in-transformers)
6. **Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters**.*Hongyu Zhao, Hao Tan, Hongyuan Mei*.**EMNLP'2022**.[paper](https://www.aminer.cn/pub/636482d890e50fcafdccb0cc/Tiny-Attention%20Adapter:%20Contexts%20Are%20More%20Important%20Than%20the%20Number%20of%20Parameters)
7. **Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks**.*Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, James Henderson*.**EMNLP'2022**.[paper](https://www.aminer.cn/pub/60c2da8091e0117e30ca2817/parameter-efficient-multi-task-fine-tuning-for-transformers-via-shared-hypernetworks)
8. **BAD-X: Bilingual Adapters Improve Zero-Shot Cross-Lingual Transfer**._Marinela Parovic, Goran Glavas, Ivan Vulic, Anna Korhonen_.**NAACL'2022**.[paper](https://www.aminer.cn/pub/634d80f190e50fcafd4ef432/bad-x-bilingual-adapters-improve-zero-shot-cross-lingual-transfer)
9. **AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning**. _Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, Jianfeng Gao_.**EMNLP'2022**.[paper](https://www.aminer.cn/pub/628ef0485aee126c0f82d92e/AdaMix:%20Mixture-of-Adaptations%20for%20Parameter-efficient%20Model%20Tuning)
10. **AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks**. _Chin-Lun Fu, Zih-Ching Chen, Yun-Ru Lee, Hung-yi Lee_.**NAACL'2022**.[paper](https://www.aminer.cn/pub/62708f615aee126c0fa69008/adapterbias-parameter-efficient-token-dependent-representation-shift-for-adapters-in-nlp-tasks)
11. **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**. _Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao_.**arXiv'2022**.[paper](https://www.aminer.cn/pub/6344dede90e50fcafd24d1cc/sparseadapter-an-easy-approach-for-improving-the-parameter-efficiency-of-adapters)
12. **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning**. _Yi-Lin Sung, Jaemin Cho, Mohit Bansal_. **NIPS'2022**.[paper](https://www.aminer.cn/pub/62a94e065aee126c0f9c02cd/lst-ladder-side-tuning-for-parameter-and-memory-efficient-transfer-learning)
13. **MerA: Merging Pretrained Adapters For Few-Shot Learning**. _Shwai He, Run-Ze Fan, Liang Ding, Li Shen, Tianyi Zhou, Dacheng Tao_. **CoRR'2023**. [paper](https://aminer.cn/pub/64f00ff43fda6d7f06ecec7d/mera-merging-pretrained-adapters-for-few-shot-learning)

### CV

### VLM

### Gragh




## [Partial](#content)

### Most Influential






## [Reparameters](#content)

### Most Influential



### Recent SOTA




## [Hybrid](#content)

### Most Influential


