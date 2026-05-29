# 雷达脉冲分选资源汇总

<p align="center">
  <b>简体中文</b> |
  <a href="./README.en.md">English</a>
</p>

本仓库持续整理 **Radar Pulse Deinterleaving / Radar Signal Sorting（雷达脉冲分选 / 雷达信号分选）** 相关的开源资源，重点关注任务定义、PDW 数据集、开源实现、代表性方法、评价指标和可复现实验。

Radar pulse deinterleaving 的目标是将混叠的脉冲流按照不同雷达辐射源进行划分。在典型的 **PDW（Pulse Description Word，脉冲描述字）** 场景中，每个脉冲通常由 **TOA（Time of Arrival，到达时间）**、**PRI（Pulse Repetition Interval，脉冲重复间隔）**、**RF/CF（Radio Frequency / Carrier Frequency，射频/载频）**、**PW（Pulse Width，脉宽）**、**PA（Pulse Amplitude，脉冲幅度）** 以及 **DOA/AOA（Direction / Angle of Arrival，到达方向/到达角）** 等参数描述。

目前雷达脉冲分选相关资源比较分散，不同项目对 `deinterleaving`、`sorting`、`clustering`、`sequence labeling` 等概念的使用并不完全一致。因此，本仓库尝试对相关资源进行人工整理，并标注其任务类型、数据可用性、监督方式以及是否符合严格的无监督分选定义。

---

## Highlights / 快速结论

* **标准 benchmark 首选：** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) 及其 [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)。
* **严格无监督分选方法：** K-means、GMM、DBSCAN、HDBSCAN、hierarchical clustering 等最符合“将混叠脉冲划分为不同辐射源簇”的定义。
* **深度学习方法需谨慎区分：** 很多 neural deinterleaving 方法虽然完成的是分选任务，但训练阶段使用了标签，因此不属于严格无监督聚类。
* **数据集现状：** 公开可用的真实雷达分选数据非常有限，当前开源研究主要依赖合成 PDW 数据。
* **本仓库定位：** 本项目不是简单罗列链接，而是尽量区分每个资源的任务类型、监督方式、数据可用性以及是否符合严格雷达脉冲分选定义。

> 注：本仓库中的 ⭐ 推荐程度是整理者基于任务匹配度、开源程度、复现价值和数据说明完整度给出的主观推荐，不代表 GitHub stars。

---

## 目录 / Table of Contents

* [1. 任务概述 / Overview](#1-任务概述--overview)

  * [1.1 什么是雷达脉冲分选？](#11-什么是雷达脉冲分选)
  * [1.2 常见输入特征](#12-常见输入特征)
  * [1.3 常见评价指标](#13-常见评价指标)
* [2. 方法分类与严格定义 / Method Taxonomy and Task Fit](#2-方法分类与严格定义--method-taxonomy-and-task-fit)

  * [2.1 传统 PRI-based 方法](#21-传统-pri-based-方法)
  * [2.2 无监督聚类方法](#22-无监督聚类方法)
  * [2.3 表征学习 + 聚类方法](#23-表征学习--聚类方法)
  * [2.4 监督序列标注与分割方法](#24-监督序列标注与分割方法)
  * [2.5 哪些方法符合严格分选定义？](#25-哪些方法符合严格分选定义)
* [3. 数据集资源 / Datasets](#3-数据集资源--datasets)

  * [3.1 数据集总览](#32-数据集总览)
  * [3.2 重点数据集说明](#33-重点数据集说明)
* [4. 开源方法与实现 / Methods and Implementations](#4-开源方法与实现--methods-and-implementations)

  * [4.1 方法与代码总览](#42-方法与代码总览)
  * [4.2 重点方法说明](#43-重点方法说明)
* [5. 推荐实验设置 / Recommended Experimental Setup](#5-推荐实验设置--recommended-experimental-setup)

  * [5.1 主 benchmark](#51-主-benchmark)
  * [5.2 基线方法](#52-基线方法)
  * [5.3 建议流程](#53-建议流程)
* [6. 推荐阅读与入门资源 / Recommended Reading and Starting Points](#6-推荐阅读与入门资源--recommended-reading-and-starting-points)
* [7. 说明 / Notes](#7-说明--notes)
* [8. 引用与贡献 / Citation and Contribution](#8-引用与贡献--citation-and-contribution)

---

## 1. 任务概述 / Overview

### 1.1 什么是雷达脉冲分选？

雷达脉冲分选，也称为 **radar signal sorting**，是指在复杂电磁环境中，将来自多个雷达辐射源的混叠脉冲序列划分到各自对应的辐射源。

<p align="center">
  <img src="./asset\figures\fig1.png" width="90%">
</p>

<p align="center">
  <b>图 1.</b> 雷达脉冲分选任务示意图：多个辐射源脉冲序列经过雷达告警设备接收后形成混叠脉冲序列，分选算法需要将其重新划分为不同辐射源对应的脉冲簇。
</p>

给定一个混叠脉冲序列：

```text
P = {p1, p2, ..., pN}
```

其中每个脉冲 `pi` 通常可以表示为一个 PDW 特征向量：

```text
pi = [TOA, RF/CF, PW, PA, DOA/AOA, ...]
```

分选任务的目标是输出一个划分结果：

```text
C = {C1, C2, ..., CK}
```

其中每个簇 `Ck` 对应一个雷达辐射源 emitter。

从最严格的意义上讲，雷达脉冲分选可以看作一个 **无监督聚类问题（unsupervised clustering problem）**，原因包括：

* 辐射源数量通常是未知的；
* 辐射源身份不是固定类别；
* 同一辐射源产生的脉冲应该被分到同一类；
* 不同辐射源产生的脉冲应该被分开。

不过，许多现代方法会将该任务建模为监督学习、半监督学习、度量学习、序列标注、语义分割或实例分割问题。这些方法虽然也可以完成脉冲分选任务，但并不一定属于纯粹的无监督聚类方法。

---

### 1.2 常见输入特征

| Feature 特征 | Meaning 含义                                                   | Usage 用途               |
| ---------- | ------------------------------------------------------------ | ---------------------- |
| TOA        | Time of Arrival，到达时间                                         | 用于计算 PRI 和分析时间规律       |
| PRI / DTOA | Pulse Repetition Interval / Difference of TOA，脉冲重复间隔 / 到达时间差 | 传统分选方法的核心特征            |
| RF / CF    | Radio Frequency / Carrier Frequency，射频 / 载频                  | 用于区分固定频率或频率捷变雷达        |
| PW         | Pulse Width，脉宽                                               | 可辅助区分不同波形参数的辐射源        |
| PA / AMP   | Pulse Amplitude，脉冲幅度                                         | 可作为辅助特征，但容易受传播路径和接收机影响 |
| DOA / AOA  | Direction / Angle of Arrival，到达方向 / 到达角                      | 若可获得，是很强的空间区分特征        |

---

### 1.3 常见评价指标

| Metric 指标    | Description 说明                                                |
| ------------ | ------------------------------------------------------------- |
| V-measure    | Homogeneity 和 completeness 的调和平均                              |
| Homogeneity  | 每个预测簇中是否主要只包含同一真实辐射源的脉冲                                       |
| Completeness | 同一真实辐射源的脉冲是否被完整分到同一预测簇中                                       |
| ARI          | Adjusted Rand Index，调整兰德指数                                    |
| AMI          | Adjusted Mutual Information，调整互信息                             |
| Pairwise F1  | 将任意两个脉冲是否属于同一辐射源视为二分类问题后的 F1                                  |
| MCC          | Matthews Correlation Coefficient，用于 pairwise matching 的相关系数指标 |

---

## 2. 方法分类与严格定义 / Method Taxonomy and Task Fit

雷达脉冲分选方法大致可以分为 **传统 PRI-based 方法、无监督聚类方法、表征学习 + 聚类方法、监督序列标注与分割方法** 等几类。

需要注意的是，不同论文或代码仓库虽然都会使用 `deinterleaving`、`sorting`、`pulse clustering` 等关键词，但它们的任务设定并不完全相同。有些方法是严格意义上的无监督聚类，有些方法则使用了监督标签、模板先验或深度学习序列标注。

---

### 2.1 传统 PRI-based 方法

这类方法主要依赖脉冲到达时间中的周期性或准周期性规律，通常以 TOA / DTOA / PRI 为核心特征。

| Method 方法         | Main idea 核心思想                          | Advantages 优点   | Limitations 局限   |
| ----------------- | --------------------------------------- | --------------- | ---------------- |
| PRI histogram     | 使用直方图估计主要 PRI 值                         | 简单、可解释          | 对丢脉冲、假脉冲和密集环境敏感  |
| CDIF              | Cumulative Difference Histogram，累积差值直方图 | 对稳定 PRI 模式有效    | 对复杂 PRI 调制的鲁棒性较弱 |
| SDIF              | Sequential Difference Histogram，序列差值直方图 | 比普通直方图更好地利用时间顺序 | 仍受 PRI 捷变和噪声影响   |
| PRI transform     | 对 TOA 序列进行变换以揭示 PRI 结构                  | 经典且研究较多         | 在高度交叠环境下性能下降     |
| Sequence matching | 根据 PRI 模板匹配脉冲列                          | 适合已知模式的辐射源      | 需要先验知识或模板        |

这类方法接近传统雷达信号分选定义，但通常依赖 PRI 规律或先验模板，不一定适用于复杂、密集、频率捷变的现代电磁环境。

---

### 2.2 无监督聚类方法

这类方法直接对 PDW 原始特征或变换后的特征进行聚类，是最符合“将混叠脉冲划分为不同辐射源簇”这一定义的方法类型。

| Method 方法                    | Main idea 核心思想      | 是否严格无监督 | 备注                         |
| ---------------------------- | ------------------- | ------- | -------------------------- |
| K-means                      | 将脉冲聚为 K 个簇          | 是       | 需要预先给定或估计辐射源数量 K           |
| GMM                          | 使用高斯混合模型建模脉冲分布      | 是       | 通常需要模型选择确定 K               |
| DBSCAN                       | 基于密度聚类，并可处理噪声点      | 是       | 不需要预先设定 K，但参数敏感            |
| HDBSCAN                      | 层次密度聚类              | 是       | 适合未知簇数和变密度场景，常作为强 baseline |
| Hierarchical clustering      | 构建聚类树，再根据准则剪枝       | 是       | 剪枝准则会显著影响结果                |
| Spectral clustering          | 基于图相似度进行聚类          | 是       | 图构建方式非常关键                  |
| Sparse subspace clustering   | 假设同一辐射源脉冲位于某种子空间结构中 | 是       | 计算成本通常较高                   |
| Optimal transport clustering | 基于分布距离进行聚类          | 是       | 对复杂 PDW 分布有潜力              |

这一类方法最符合严格定义：**在不使用辐射源标签进行训练的情况下，将混叠脉冲划分为不同 emitter clusters。**

---

### 2.3 表征学习 + 聚类方法

这类方法通常先学习一个更适合分选的特征表示，再在特征空间中进行聚类。

| Method 方法                         | Main idea 核心思想                | Supervision level 监督程度 | 备注           |
| --------------------------------- | ----------------------------- | ---------------------- | ------------ |
| Autoencoder + clustering          | 学习压缩的 PDW embedding，再进行聚类     | 无监督或弱监督                | 适合特征重叠较强的场景  |
| Contrastive learning + clustering | 将相似脉冲或相似序列在特征空间中拉近            | 自监督、弱监督或监督均可能          | 需要明确正负样本构造方式 |
| Transformer encoder + HDBSCAN     | 学习具有上下文信息的脉冲 embedding，再聚类    | 常见为监督度量学习，也可设计为自监督     | 适合长序列建模      |
| Triplet-loss metric learning      | 使同一辐射源脉冲的 embedding 更接近，不同源更远 | 训练阶段有监督，推理阶段聚类         | 不能简单归为纯无监督方法 |

这类方法最终仍然输出辐射源簇，因此可以完成分选任务，但训练过程不一定是纯无监督的。

---

### 2.4 监督序列标注与分割方法

一些近期工作会将雷达脉冲分选建模为监督式序列标注、语义分割或实例分割任务。

| Method 方法                     | Main idea 核心思想          | Comment 说明      |
| ----------------------------- | ----------------------- | --------------- |
| RNN / LSTM / GRU              | 建模脉冲序列中的时间依赖关系          | 通常需要有标签训练数据     |
| TCN                           | 使用时间卷积进行长序列建模           | 对长序列较高效         |
| Transformer                   | 建模脉冲之间的长距离依赖关系          | 表达能力强，但对数据量要求高  |
| GCN                           | 构建脉冲关系图，对节点或边进行分类       | 适合建模脉冲间关系       |
| U-Net / semantic segmentation | 将脉冲序列转换为类似图像的表示，再进行语义分割 | 需要密集标签          |
| Mask R-CNN / SOLOv2           | 将不同辐射源视为图像中的不同实例        | 更接近实例分割，而不是传统聚类 |

这些方法可以解决分选任务，但它们 **不严格属于无监督聚类式分选方法**。

---

### 2.5 哪些方法符合严格分选定义？

> **严格定义：** 给定混叠脉冲序列，在不使用辐射源标签进行训练的情况下，将脉冲聚类为若干组，每组对应一个辐射源。

| Method Family 方法族                       | Matches the task? 是否符合分选任务 | Strictly unsupervised? 是否严格无监督 | Recommendation 推荐程度 | Comment 说明                     |
| --------------------------------------- | -------------------------- | ------------------------------ | ------------------- | ------------------------------ |
| DBSCAN / HDBSCAN                        | 是                          | 是                              | ⭐⭐⭐⭐⭐               | 适合未知 K，是当前最推荐的无监督聚类 baseline   |
| K-means / GMM                           | 是                          | 是                              | ⭐⭐⭐⭐                | 方法简单、易复现，但通常需要 K 或模型选择         |
| Hierarchical clustering                 | 是                          | 是                              | ⭐⭐⭐⭐                | 可解释性较好，但剪枝准则重要                 |
| PRI histogram / CDIF / SDIF             | 是                          | 基本是                            | ⭐⭐⭐                 | 传统方法，适合 PRI 结构明显的场景            |
| Autoencoder + clustering                | 是                          | 通常是                            | ⭐⭐⭐⭐                | 适合表征学习方向，需确认训练目标是否使用标签         |
| Contrastive learning + clustering       | 是                          | 视情况而定                          | ⭐⭐⭐⭐                | 可设计为自监督，也可能是监督                 |
| Transformer + triplet loss + clustering | 是                          | 否                              | ⭐⭐⭐                 | 可作为监督表征学习方法或上限对照               |
| TCN / RNN sequence labeling             | 是                          | 否                              | ⭐⭐⭐                 | 通常属于监督学习                       |
| GCN node/edge classification            | 是                          | 否 / 半监督                        | ⭐⭐⭐                 | 适合关系建模，但任务设定需仔细区分              |
| U-Net / semantic segmentation           | 相关                         | 否                              | ⭐⭐                  | 需要密集标签，更接近分割任务                 |
| Mask R-CNN / SOLOv2                     | 相关                         | 否                              | ⭐⭐                  | 属于实例分割式建模                      |
| RadSeg-style activity segmentation      | 相关                         | 否                              | ⭐⭐                  | 检测雷达活动，不是标准 emitter clustering |

---

## 3. 数据集资源 / Datasets

本节主要整理雷达脉冲分选相关的数据集、仿真数据和相关任务数据。推荐程度主要根据 **任务匹配度、数据是否开源、是否包含 ground-truth labels、是否适合作为 benchmark、数据说明是否清晰** 等因素综合判断。

> 说明：这里的星级是本仓库的整理推荐度，不代表 GitHub stars。

---



### 3.1 数据集总览

| Dataset 数据集                                | Task Fit 任务匹配度           | Data Type 数据类型                     | Labels 标签                     | Open Source? 是否开源 | Recommendation 推荐度 | Links 链接                                                                                                                                                                             | Notes 备注                     |
| ------------------------------------------ | ------------------------ | ---------------------------------- | ----------------------------- | ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| Turing Synthetic Radar Dataset, TSRD       | 标准 PDW 脉冲分选              | 合成 PDW 序列                          | 有 emitter ground-truth labels | 是                 | ⭐⭐⭐⭐⭐              | [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | 当前最推荐作为主 benchmark           |
| radar_data_Kmeans data                     | 雷达脉冲分选                   | 仿真 PDW / 项目数据                      | 仓库内数据说明有限                     | 部分，仓库内包含          | ⭐⭐⭐                | [GitHub](https://github.com/zda2019/radar_data_Kmeans)                                                                                                                               | 适合 K-means 工程实现和小型实验         |
| Stream-ConAEnet data                       | 雷达脉冲分选                   | `.mat` 脉冲特征数据                      | 可能包含标签，但字段说明较少                | 部分，仓库内包含          | ⭐⭐⭐                | [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)                                                                                             | 表征学习方向可参考，复现前需检查数据格式         |
| HMC-RFN simulation data                    | PRI-based deinterleaving | 仿真 TOA / PRI 数据                    | 仿真生成标签                        | 代码可生成             | ⭐⭐⭐                | [GitHub](https://github.com/xm980426/HMC-RFN)                                                                                                                                        | 适合 PRI 时间结构建模实验              |
| 2nd-EBDSC data                             | 竞赛式信号提取 / 分选             | PDW 序列                             | 训练 / 验证集有标签                   | 部分，通过网盘等方式提供      | ⭐⭐⭐                | [GitHub](https://github.com/framist/2nd-EBDSC)                                                                                                                                       | 更接近监督式或模板辅助任务                |
| EW Signal Intelligence demo data           | 教学型分选 demo               | CSV 脉冲数据                           | Demo labels / tracks          | 是                 | ⭐⭐                 | [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)                                                                                                          | 适合理解基本流程，不适合作为研究 benchmark   |
| RadSeg                                     | 雷达活动分割                   | 信号序列 / sample-wise segmentation 数据 | 有 sample-wise annotations     | 是                 | ⭐⭐                 | [GitHub](https://github.com/abcxyzi/radseg)                                                                                                                                          | 相关任务，不是标准 emitter clustering |
| Real measured radar deinterleaving dataset | 真实雷达分选                   | 真实 PDW / IQ                        | 通常不可获得                        | 公开资源很少            | ⭐                  | 暂无稳定公开 benchmark                                                                                                                                                                     | 真实数据难公开、难标注，目前不适合作为开源复现实验基础  |

---

### 3.2 重点数据集说明

#### 3.2.1 Turing Synthetic Radar Dataset / TSRD

**推荐程度：** ⭐⭐⭐⭐⭐
**适合用途：** 标准 benchmark、无监督聚类 baseline、深度表征学习、监督 / 半监督方法评估
**数据类型：** 合成 PDW 序列
**标签情况：** 提供 ground-truth emitter labels
**相关代码：** Turing Deinterleaving Challenge

TSRD 是目前最适合作为雷达脉冲分选主 benchmark 的公开数据集。它与 Turing Deinterleaving Challenge 配套使用，提供了标准任务定义、数据加载方式、baseline 和评价协议。

整理备注：如果目标是研究 **无监督雷达脉冲分选**，建议优先从该数据集开始，并使用 HDBSCAN、DBSCAN、K-means、hierarchical clustering 等方法建立 baseline。

**链接：** [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

---

#### 3.2.2 radar_data_Kmeans data

**推荐程度：** ⭐⭐⭐
**适合用途：** K-means 分选入门、工程实现参考、嵌入式部署参考
**数据类型：** 仿真 PDW / 项目数据
**标签情况：** 数据和评价说明不如标准 benchmark 完整

该项目包含 K-means 雷达脉冲分选相关的数据与仿真脚本，更适合用于理解传统聚类式分选和工程实现流程。

整理备注：适合作为补充实验或工程参考，不建议单独作为论文主 benchmark。

**链接：** [GitHub](https://github.com/zda2019/radar_data_Kmeans)

---

#### 3.2.3 Stream-ConAEnet data

**推荐程度：** ⭐⭐⭐
**适合用途：** 对比自编码器、表征学习、流式分选实验
**数据类型：** `.mat` 脉冲特征数据
**标签情况：** 仓库中包含数据文件和模型权重，但字段含义和数据说明较有限

该数据更适合用于研究深度表征学习与聚类结合的分选方法。由于数据说明不够完整，复现前需要先检查 `.mat` 文件字段含义和标签定义。

**链接：** [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

---

#### 3.2.4 2nd-EBDSC data

**推荐程度：** ⭐⭐⭐
**适合用途：** 竞赛式混叠信号提取、监督序列标注、模板辅助任务
**数据类型：** PDW 序列
**标签情况：** 训练 / 验证数据包含标签，测试集情况依赖竞赛设置

该数据与雷达脉冲分选相关，但不是严格的无监督 emitter clustering 数据集。它更适合研究有监督或模板辅助的混叠信号提取任务。

**链接：** [GitHub](https://github.com/framist/2nd-EBDSC)

---

## 4. 开源方法与实现 / Methods and Implementations

本节主要整理雷达脉冲分选相关的开源方法、代码实现和可复现实验框架。推荐程度主要根据 **是否符合分选任务、是否开源、是否容易复现、是否适合作为 baseline、方法代表性** 等因素综合判断。

> 说明：这里的星级是本仓库的整理推荐度，不代表 GitHub stars。

---


### 4.1 方法与代码总览

| Project / Method 项目或方法                  | Method Type 方法类型                               | Supervision 监督方式 | Strictly Unsupervised? 是否严格无监督 | Recommendation 推荐度 | Links 链接                                                                                 | Notes 备注                         |
| --------------------------------------- | ---------------------------------------------- | ---------------- | ------------------------------ | ------------------ | ---------------------------------------------------------------------------------------- | -------------------------------- |
| Turing HDBSCAN baseline                 | HDBSCAN raw PDW clustering                     | 无监督              | 是                              | ⭐⭐⭐⭐⭐              | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)       | 当前最推荐作为标准无监督 baseline            |
| radar_data_Kmeans                       | K-means clustering / APG / ZYNQ deployment     | 无监督聚类            | 是，但需要 K                        | ⭐⭐⭐⭐               | [GitHub](https://github.com/zda2019/radar_data_Kmeans)                                   | 适合传统聚类与工程部署参考                    |
| DBSCAN / HDBSCAN clustering             | Density-based clustering                       | 无监督              | 是                              | ⭐⭐⭐⭐⭐              | 可基于 Turing 框架自行实现                                                                        | 适合未知辐射源数量场景                      |
| K-means / GMM / hierarchical clustering | Classical clustering baselines                 | 无监督              | 是                              | ⭐⭐⭐⭐               | 可基于 Turing 框架自行实现                                                                        | 适合作为基础 baseline 组合               |
| Stream-ConAEnet                         | Contrastive autoencoder / streaming learning   | 无监督 / 弱监督情况需核实   | 部分符合                           | ⭐⭐⭐                | [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet) | 表征学习方向有参考价值，但数据说明有限              |
| HMC-RFN                                 | Hidden Markov Chains / Residual Fence Networks | 模型驱动 / 先验驱动      | 否                              | ⭐⭐⭐                | [GitHub](https://github.com/xm980426/HMC-RFN)                                            | 适合 PRI-based 时间结构建模              |
| 2nd-EBDSC solution                      | Wide-value embeddings / TCN / masking          | 监督 / 模板辅助        | 否                              | ⭐⭐⭐                | [GitHub](https://github.com/framist/2nd-EBDSC)                                           | 工程参考价值较高，但不是严格无监督聚类              |
| EW Signal Intelligence Demo             | PRI-based and feature-based grouping           | 基本无监督 / demo     | 基本符合                           | ⭐⭐                 | [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)              | 教学和快速原型参考                        |
| RadSeg-style segmentation               | Radar activity segmentation                    | 监督分割             | 否                              | ⭐⭐                 | [GitHub](https://github.com/abcxyzi/radseg)                                              | 相关任务，不是标准 PDW emitter clustering |

---

### 4.2 重点方法说明

#### 4.2.1 Turing HDBSCAN baseline

**推荐程度：** ⭐⭐⭐⭐⭐
**方法类型：** HDBSCAN on raw PDW features
**监督方式：** 无监督
**是否严格符合分选定义：** 是
**适合用途：** 标准 baseline、无监督聚类方法对比、后续深度表征学习方法的对照

Turing Deinterleaving Challenge 中的 HDBSCAN baseline 是目前最适合作为开源雷达脉冲分选 baseline 的方法之一。它不需要预设辐射源数量，能够直接在 PDW 特征空间中进行聚类。

整理备注：如果要搭建一个无监督分选实验框架，建议首先复现该 baseline，然后再加入 DBSCAN、K-means、GMM、hierarchical clustering 等对比方法。

**链接：** [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

---

#### 4.2.2 radar_data_Kmeans

**推荐程度：** ⭐⭐⭐⭐
**方法类型：** K-means clustering
**监督方式：** 无监督聚类
**是否严格符合分选定义：** 基本符合，但需要给定或估计 K
**适合用途：** 传统聚类分选入门、工程实现、嵌入式部署参考

该项目使用 K-means 聚类实现雷达脉冲分选，并包含 APG 优化、C++ 实现和 ZYNQ 部署相关内容。相比纯研究代码，它更偏工程实现。

整理备注：适合用来理解聚类式分选的基本流程，但由于 K-means 需要辐射源数量 K，在真实未知场景中需要配合 K 估计或模型选择方法。

**链接：** [GitHub](https://github.com/zda2019/radar_data_Kmeans)

---

#### 4.2.3 Stream-ConAEnet

**推荐程度：** ⭐⭐⭐
**方法类型：** Contrastive autoencoder + streaming learning
**监督方式：** 包含无监督、弱监督或带标签训练阶段
**是否严格符合分选定义：** 部分符合
**适合用途：** 表征学习、流式分选、深度聚类方向参考

该项目使用对比自编码器学习脉冲特征表示，再结合聚类或动态中心模块进行分选。它适合用来参考“深度表征学习 + 聚类”的思路。

整理备注：该项目有一定参考价值，但数据说明和训练细节需要仔细核查，不建议直接将其作为严格无监督方法引用。

**链接：** [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

---

#### 4.2.4 HMC-RFN

**推荐程度：** ⭐⭐⭐
**方法类型：** PRI-based temporal modeling
**监督方式：** 模型驱动 / 先验驱动
**是否严格符合分选定义：** 不属于纯无监督聚类
**适合用途：** PRI 结构建模、仿真实验、传统时序模型对比

HMC-RFN 使用 Hidden Markov Chains 和 Residual Fence Networks 来建模雷达脉冲分选中的时间结构。它不是纯粹的数据驱动聚类方法，而是更依赖 PRI 模式和先验建模。

**链接：** [GitHub](https://github.com/xm980426/HMC-RFN)

---

#### 4.2.5 2nd-EBDSC solution

**推荐程度：** ⭐⭐⭐
**方法类型：** Wide-value embeddings + TCN + masking
**监督方式：** 监督 / 模板辅助
**是否严格符合分选定义：** 否
**适合用途：** 序列建模、竞赛式信号提取、工程参考

该项目与混叠脉冲序列处理相关，但更接近监督式或模板辅助的信号提取任务，而不是严格意义上的无监督 emitter clustering。

整理备注：适合作为工程实现和深度序列建模参考，但在 README 中需要明确标注其任务设定与标准无监督分选不同。

**链接：** [GitHub](https://github.com/framist/2nd-EBDSC)

---

#### 4.2.6 EW Signal Intelligence Deinterleaving Demo

**推荐程度：** ⭐⭐
**方法类型：** PRI-based / feature-based grouping
**监督方式：** 基本无监督 / demo
**是否严格符合分选定义：** 概念上基本符合
**适合用途：** 教学、快速原型、理解基本分选流程

该项目提供了简单的 Python demo，适合用来理解 deinterleaving 的基本流程，但不适合作为研究级 benchmark。

**链接：** [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)

---

#### 4.2.7 RadSeg

**推荐程度：** ⭐⭐
**方法类型：** Radar activity segmentation
**监督方式：** 监督分割
**是否严格符合分选定义：** 否
**适合用途：** 相关任务参考、雷达活动检测与分割

RadSeg 关注的是雷达活动的 sample-wise segmentation，不是标准的 PDW emitter clustering。它可以作为相关任务参考，但不应与标准雷达脉冲分选数据集混为一类。

**链接：** [GitHub](https://github.com/abcxyzi/radseg)

---

## 5. 推荐实验设置 / Recommended Experimental Setup

如果研究目标是 **无监督雷达脉冲分选（unsupervised radar pulse deinterleaving）**，建议采用以下实验设置。

---

### 5.1 主 benchmark

建议将 **Turing Synthetic Radar Dataset** 作为主要 benchmark，因为它具备：

* 清晰的任务定义；
* 开源数据；
* ground-truth emitter labels；
* 标准聚类评价指标；
* 基线代码；
* 支持未知辐射源数量的实验设置。

---

### 5.2 基线方法

建议实现并比较以下 baseline：

| Category 类别             | Methods 方法                                                    |
| ----------------------- | ------------------------------------------------------------- |
| Raw-feature clustering  | K-means，GMM，DBSCAN，HDBSCAN，hierarchical clustering            |
| PRI-based methods       | PRI histogram，CDIF，SDIF，PRI transform                         |
| Hybrid methods          | RF/PW/DOA 粗聚类 + PRI refinement                                |
| Representation learning | Autoencoder + HDBSCAN，contrastive encoder + clustering        |
| Supervised upper bound  | Transformer metric learning，TCN，GCN，segmentation-based models |

---

### 5.3 建议流程

```text
Raw PDW sequence
      ↓
Feature normalization
      ↓
Optional temporal feature construction
      ↓
Unsupervised clustering or representation learning
      ↓
Cluster label assignment
      ↓
Evaluation using V-measure / ARI / AMI / pairwise F1
```

---


## 6. 推荐阅读与入门资源 / Recommended Reading and Starting Points

### 标准分选任务的最佳起点

* [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
* [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)

### 简单无监督聚类方法的最佳起点

* [radar_data_Kmeans](https://github.com/zda2019/radar_data_Kmeans)
* Turing Deinterleaving Challenge 中的 HDBSCAN baseline

### 表征学习方向的起点

* [Radar Pulse Sorting Based on Stream-ConAEnet](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

### PRI-based 时间建模方向的起点

* [HMC-RFN](https://github.com/xm980426/HMC-RFN)

### 相关但不严格属于分选的任务

* [2nd-EBDSC](https://github.com/framist/2nd-EBDSC)
* [RadSeg](https://github.com/abcxyzi/radseg)

---

## 7. 说明 / Notes

* 与通用 RF modulation recognition 或 RF fingerprinting 相比，雷达脉冲分选方向的开源资源仍然较少；
* 大多数公开数据集是合成数据，因为真实雷达 PDW 或 IQ 数据很难公开，也很难获得可靠 ground truth；
* 报告实验结果时，应明确说明方法属于无监督、监督、半监督，还是“监督表征学习后再聚类”；
* 如果研究目标是严格的无监督雷达脉冲分选，Turing TSRD + HDBSCAN / DBSCAN / K-means / hierarchical clustering 是最透明、最容易复现的起点；
* 本仓库中的推荐程度会随着项目更新、数据开放情况和复现记录变化而调整。

---

## 8. 引用与贡献 / Citation and Contribution

本资源列表由 **厦门大学信息学院 SMARTDSP 实验室** 整理与维护，旨在汇总雷达脉冲分选（Radar Pulse Deinterleaving / Radar Signal Sorting）相关的公开数据集、开源代码、代表性方法和可复现实验资源。

如果本仓库对你的学习、研究或项目开发有帮助，欢迎在论文、报告或项目文档中引用本仓库，并请同时引用相关方法、数据集和代码仓库的原始论文或项目页面。

### 维护说明

- 本仓库主要关注雷达脉冲分选、PDW deinterleaving、radar signal sorting 及其相关任务；
- 资源整理过程中会尽量标注任务类型、数据可用性、监督方式和是否符合严格无监督分选定义；
- 由于部分开源项目的数据说明、许可证、复现流程可能不完整，相关判断会随着项目更新持续修正；
- 本仓库中的推荐程度仅代表整理者基于任务匹配度、开源程度、复现价值和数据说明完整度给出的参考意见。

### 贡献方式

欢迎通过 Issue 或 Pull Request 补充和修正资源，包括但不限于：

- 新发布的雷达脉冲分选数据集；
- PRI-based、clustering-based 或 deep learning-based 分选方法的开源实现；
- 可复现的 benchmark 结果；
- 数据集可用性、链接失效或许可证信息的修正；
- 关于某个方法是否严格无监督的补充说明。

建议提交资源时尽量包含以下信息：

```text
项目名称：
项目链接：
任务类型：
方法类型：
是否开源代码：
是否开源数据：
是否包含标签：
是否严格无监督：
推荐理由：
备注：

