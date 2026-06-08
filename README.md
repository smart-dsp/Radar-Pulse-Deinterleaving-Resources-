# 雷达脉冲分选资源汇总

<p align="center">
  <b>简体中文</b> |
  <a href="./README.en.md">English</a>
</p>

本仓库由 **厦门大学信息学院 SmartDSP 实验室** 整理，汇总 **雷达脉冲分选 / 雷达信号分选（Radar Pulse Deinterleaving / Radar Signal Sorting）** 相关的高价值公开资源，重点关注任务定义、PDW 数据集、可复现实验框架、代表性方法和评价指标。

本资源列表优先保留与雷达脉冲分选任务高度相关、具有明确任务定义、公开数据或代码、论文支撑或较强复现价值的资源。对于仅名称相关、文档不足、复现价值有限或不适合作为正式研究资源的项目，不纳入正文推荐。

雷达脉冲分选的目标是将混叠的脉冲流按照不同雷达辐射源进行划分。在典型的 **PDW（Pulse Description Word，脉冲描述字）** 场景中，每个脉冲通常由 **TOA（Time of Arrival，到达时间）**、**PRI（Pulse Repetition Interval，脉冲重复间隔）**、**RF/CF（Radio Frequency / Carrier Frequency，射频/载频）**、**PW（Pulse Width，脉宽）**、**PA（Pulse Amplitude，脉冲幅度）** 以及 **DOA/AOA（Direction / Angle of Arrival，到达方向 / 到达角）** 等参数描述。

---

## 快速结论

* **首选基准：** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) 与 [Turing Synthetic Radar Dataset / TSRD](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)。
* **首选基线：** Turing Challenge 中提供的 HDBSCAN 基线，可作为严格无监督雷达脉冲分选方法的起点。
* **严格无监督分选：** DBSCAN、HDBSCAN、层次聚类、K-means、GMM 等传统聚类方法最符合“未知辐射源数量下的辐射源聚类”定义。
* **深度学习方法：** Transformer 度量学习、深度对比聚类、语义分割、图卷积网络等方法具有研究价值，但多数依赖标签训练，更适合作为代表性方法或性能上限参考。
* **公开数据现状：** 真实雷达 PDW / IQ 分选数据公开较少，目前可复现实验主要依赖合成 PDW 数据。

---

## 目录

* [1. 任务概述](#1-任务概述)
  * [1.1 什么是雷达脉冲分选？](#11-什么是雷达脉冲分选)
  * [1.2 常见输入特征](#12-常见输入特征)
  * [1.3 常见评价指标](#13-常见评价指标)
* [2. 方法分类与任务边界](#2-方法分类与任务边界)
  * [2.1 方法类型总览](#21-方法类型总览)
  * [2.2 方法与严格无监督分选的关系](#22-方法与严格无监督分选的关系)
* [3. 数据集与基准](#3-数据集与基准)
  * [3.1 数据集总览](#31-数据集总览)
  * [3.2 Turing Synthetic Radar Dataset / TSRD](#32-turing-synthetic-radar-dataset--tsrd)
  * [3.3 RadSeg](#33-radseg)
* [4. 代表性方法与论文](#4-代表性方法与论文)
  * [4.1 论文总览](#41-论文总览)
  * [4.2 方法简介](#42-方法简介)
  * [4.3 论文与方法详细说明](#43-论文与方法详细说明)
* [5. 推荐实验设置](#5-推荐实验设置)
  * [5.1 主基准](#51-主基准)
  * [5.2 建议基线](#52-建议基线)
  * [5.3 建议流程](#53-建议流程)
* [6. 非核心资源说明](#6-非核心资源说明)
* [7. 说明](#7-说明)
* [8. 引用与贡献](#8-引用与贡献)

---

## 1. 任务概述

### 1.1 什么是雷达脉冲分选？

雷达脉冲分选，也称为 **雷达脉冲解交织（radar pulse deinterleaving）** 或 **雷达信号分选（radar signal sorting）**，是指在复杂电磁环境中，将来自多个雷达辐射源的混叠脉冲序列划分到各自对应的辐射源。

<p align="center">
  <img src="./assets/figures/fig1.png" width="90%">
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

其中每个簇 `Ck` 对应一个雷达辐射源。

从严格意义上讲，雷达脉冲分选可以看作一个 **无监督聚类问题**：

* 辐射源数量通常未知；
* 辐射源身份不是固定类别；
* 同一辐射源产生的脉冲应被分到同一类；
* 不同辐射源产生的脉冲应被分开。

同时，许多现代方法也会将该任务建模为监督学习、半监督学习、度量学习、序列标注或语义分割问题。这些方法可以完成分选式输出，但不一定属于严格的无监督聚类式分选方法。

---

### 1.2 常见输入特征

| 特征 | 含义 | 用途 |
| --- | --- | --- |
| TOA | 到达时间（Time of Arrival） | 用于计算 PRI 和分析时间规律 |
| PRI / DTOA | 脉冲重复间隔 / 到达时间差 | 传统分选方法的核心特征 |
| RF / CF | 射频 / 载频 | 用于区分固定频率或频率捷变雷达 |
| PW | 脉宽 | 可辅助区分不同波形参数的辐射源 |
| PA / AMP | 脉冲幅度 | 可作为辅助特征，但容易受传播路径和接收机影响 |
| DOA / AOA | 到达方向 / 到达角 | 若可获得，是很强的空间区分特征 |

---

### 1.3 常见评价指标

| 指标 | 说明 |
| --- | --- |
| V-measure | 同质性和完整性的调和平均 |
| Homogeneity | 每个预测簇中是否主要只包含同一真实辐射源的脉冲 |
| Completeness | 同一真实辐射源的脉冲是否被完整分到同一预测簇中 |
| ARI | 调整兰德指数（Adjusted Rand Index） |
| AMI | 调整互信息（Adjusted Mutual Information） |
| Pairwise F1 | 将任意两个脉冲是否属于同一辐射源视为二分类问题后的 F1 |
| MCC | Matthews 相关系数，可用于成对匹配评价 |

---

## 2. 方法分类与任务边界

本节用于说明雷达脉冲分选中常见的技术路线，并区分不同方法与标准 **雷达脉冲分选 / 辐射源聚类** 定义之间的关系。方法分类并不等同于开源资源列表；后续资源表仅收录具有明确论文、数据或项目链接的内容。

### 2.1 方法类型总览

| 方法类型 | 任务匹配度 | 说明 |
| --- | --- | --- |
| 基于 PRI 的传统方法 | 高 | 主要利用 TOA / PRI 周期结构，如 PRI 直方图、CDIF、SDIF、PRI 变换等 |
| 传统聚类方法 | 高 | K-means、GMM、DBSCAN、HDBSCAN、层次聚类等，可直接用于 PDW 特征聚类 |
| 基于密度的聚类基线 | 高 | HDBSCAN / DBSCAN 适合未知簇数场景，是当前无监督分选实验中常用的基础对照 |
| 度量学习 + 聚类 | 中到高 | 通过监督或弱监督方式学习脉冲嵌入表示，再在特征空间中聚类 |
| 最优传输聚类 | 中到高 | 先进行过分割，再利用分布距离或最优传输距离进行簇合并 |
| 对比学习 / 表征学习 | 中 | 通过对比学习、自编码器或表征学习提升特征可分性，具体任务设定需要结合论文判断 |
| 序列标注 / 语义分割 | 中 | 将脉冲分选建模为序列标注或语义分割任务，通常依赖有标签训练数据 |
| 雷达活动分割 | 相关任务 | 面向雷达活动检测或分割，不等同于标准 PDW 辐射源聚类 |

### 2.2 方法与严格无监督分选的关系

| 方法类型 | 是否严格无监督 | 说明 |
| --- | --- | --- |
| DBSCAN / HDBSCAN | 是 | 不需要预设辐射源类别标签，适合未知辐射源数量场景 |
| K-means / GMM | 是 | 属于无监督聚类，但通常需要预设或估计簇数 K |
| 层次聚类 | 是 | 适合构建脉冲相似性层次结构，剪枝准则会影响结果 |
| PRI 直方图 / CDIF / SDIF / PRI 变换 | 基本符合 | 传统模型驱动方法，适合 PRI 结构明显的脉冲序列 |
| 最优传输聚类 | 基本符合 | 以无监督聚类和簇合并为主，适合复杂 PDW 分布建模 |
| 度量学习 + 聚类 | 否 | 训练阶段通常使用标签，推理阶段可通过聚类输出辐射源分组 |
| 对比学习 + 聚类 | 取决于具体设定 | 自监督设定可接近无监督分选，监督对比学习则不属于严格无监督方法 |
| 语义分割 / 序列标注 | 否 | 通常依赖密集标签训练，属于监督式分选建模 |
| 雷达活动分割 | 否 | 主要解决雷达活动检测或分割问题，不是标准辐射源聚类 |

---

## 3. 数据集与基准

### 3.1 数据集总览

| 数据集 | 任务匹配度 | 数据类型 | 标签 | 是否开源 | 推荐度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| [Turing Synthetic Radar Dataset / TSRD](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) | 标准 PDW 脉冲分选 | 合成 PDW 脉冲序列 | 辐射源标签 | 是，需接受 Hugging Face 访问条件 | ⭐⭐⭐⭐⭐ | 当前最推荐作为主基准 |
| [RadSeg](https://github.com/abcxyzi/radseg) | 相关任务：雷达活动分割 | 复基带 IQ 序列 + 逐采样点掩码 | 分割掩码 | 是 | ⭐⭐ | 正式数据集，但不是标准 PDW 辐射源聚类 |

---

### 3.2 Turing Synthetic Radar Dataset / TSRD

**推荐度：** ⭐⭐⭐⭐⭐  
**任务：** 雷达脉冲分选 / PDW 辐射源聚类  
**数据类型：** 合成 PDW 脉冲序列  
**标签：** 辐射源标签  
**项目：** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

TSRD 是目前最适合作为雷达脉冲分选主基准的公开数据集之一。它面向混叠雷达脉冲序列，提供 PDW 序列、辐射源标签、stare / scan 两种接收模式和配套评价框架。

**推荐理由：**

* 任务定义明确，直接面向雷达脉冲分选；
* 数据规模较大，适合无监督、半监督和监督方法评估；
* 提供真实辐射源标签，可计算 V-measure、ARI、AMI、同质性、完整性等指标；
* 与 Turing Deinterleaving Challenge 配套，便于复现基线；
* 适合作为新方法的主基准。

**使用说明：**

* 数据为合成数据，不等同于真实雷达实测 PDW；
* 数据集文件较大，下载和处理需要一定计算与存储资源；
* 使用监督模型训练时，应明确区分其与严格无监督分选任务之间的差异。

---

### 3.3 RadSeg

**推荐度：** ⭐⭐  
**任务：** 雷达脉冲活动分割  
**数据类型：** 复基带 IQ 采样  
**标签：** 逐通道分割掩码  
**仓库：** [RadSeg](https://github.com/abcxyzi/radseg)

RadSeg 是面向雷达活动检测与分割任务的数据集，提供 IQ 序列和逐采样点分割掩码。该数据集与雷达信号分析相关，但其任务目标不是标准的 PDW 辐射源聚类。

**保留原因：**

* 数据集较正式，有论文支撑；
* 数据和任务说明较完整；
* 对基于原始 IQ 信号或分割建模的研究有参考价值。

**任务边界：**

* RadSeg 解决的是雷达脉冲活动分割，不是 PDW 辐射源聚类；
* 标注形式是分割掩码，而不是辐射源级脉冲标签；
* 不适合与 Turing TSRD 这类标准分选数据集直接对比。

---

## 4. 代表性方法与论文

本节整理具有明确论文出处的代表性方法。代码 / 项目列仅标注论文作者、数据集官方或任务官方提供的公开项目。

其中，**Turing Synthetic Radar Dataset / Turing Deinterleaving Challenge** 同时具备论文、数据集、基线代码和评价框架，是当前最适合作为可复现实验起点的资源。因此，本节对该资源及其 HDBSCAN 基线单独展开说明。

### 4.1 论文总览

| 方法 / 方向 | 代表论文 | 论文链接 | 代码 / 项目 | 任务匹配度 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 基准 / 数据集 | The Turing Synthetic Radar Dataset: A Dataset for Pulse Deinterleaving | [arXiv](https://arxiv.org/abs/2602.03856) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | 高 | 提供数据、标签、评价指标与官方基线，适合作为主基准 |
| HDBSCAN 基线 | Turing Deinterleaving Challenge baseline | [Project](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | 高 | 官方可复现无监督基线，直接基于 PDW 特征进行密度聚类 |
| Transformer 度量学习 | Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning | [arXiv](https://arxiv.org/abs/2503.13476) | 仅有论文，无开源代码 | 高 | 使用 Transformer + triplet loss 学习脉冲嵌入，推理阶段用于辐射源聚类 |
| 最优传输聚类 | Deinterleaving RADAR Emitters with Optimal Transport Distances | [arXiv](https://arxiv.org/abs/2312.11178) | 仅有论文，无开源代码 | 高 | 无监督聚类 + 最优传输距离簇合并，适合复杂 PDW 分布下的辐射源分选 |
| 更新过程混合模型 | Deinterleaving of Mixtures of Renewal Processes | [DOI](https://doi.org/10.1109/TSP.2018.2886149) | 仅有论文，无开源代码 | 中到高 | 从随机过程混合模型角度建模脉冲序列分选 |
| 离散更新过程混合模型 | Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures | [arXiv](https://arxiv.org/abs/2402.09166) / [DOI](https://doi.org/10.1109/TSP.2024.3464753) | 仅有论文，无开源代码 | 中到高 | 面向电子支援场景的离散更新马尔可夫链分选方法 |
| 深度对比聚类 | Deep Contrastive Clustering for Signal Deinterleaving | [DOI](https://doi.org/10.1109/TAES.2023.3322971) | 仅有论文，无开源代码 | 中到高 | 支撑对比学习 + 聚类方向 |
| 语义分割式分选 | A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network | [arXiv](https://arxiv.org/abs/2110.13706) / [DOI](https://doi.org/10.1109/TSP.2022.3229630) | 仅有论文，无开源代码 | 中 | 将分选建模为序列语义标注 / 分割任务，通常需要标签训练 |
| 图像分割式分选 | Image Segmentation for Radar Signal Deinterleaving Using Deep Learning | [DOI](https://doi.org/10.1109/TAES.2022.3188225) | 仅有论文，无开源代码 | 中 | 将雷达脉冲序列转换为图像表示，再使用图像分割方式完成分选 |
| Sep-RefineNet | Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation | [Paper](https://www.mdpi.com/2076-3417/13/4/2726) | 仅有论文，无开源代码 | 中 | 构造频率特征矩阵，通过语义分割网络完成脉冲流位置解码 |
| Deep ToA mask | Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving | [DOI](https://doi.org/10.1109/TAES.2022.3193948) | 仅有论文，无开源代码 | 中到高 | 使用 ToA 掩码和递归方式逐步分离混叠脉冲序列 |
| GCN 半监督分选 | Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning | [DOI](https://doi.org/10.1109/LSP.2024.3519884) | 仅有论文，无开源代码 | 中 | 半监督图学习式雷达信号分选，不属于严格无监督聚类 |

---

### 4.2 方法简介

#### 4.2.1 Turing Synthetic Radar Dataset / Turing Deinterleaving Challenge

Turing Synthetic Radar Dataset / TSRD 是当前最适合作为雷达脉冲分选主基准的公开数据集之一。该数据集以 PDW 脉冲序列为基本输入，提供辐射源标签，可用于评估算法是否能够把混叠脉冲流重新划分为不同辐射源对应的脉冲簇。

与一般只提供数据文件的资源不同，Turing Deinterleaving Challenge 同时提供了数据使用说明、评价流程和 HDBSCAN 基线，因此更适合作为可复现实验平台。对于新方法，可以在同一数据划分、同一评价协议和同一指标体系下，与官方基线或自实现聚类方法进行比较。

**可复现实验价值：**

* 数据集、任务定义和评价指标相对完整；
* 提供 stare / scan 等不同接收模式，便于测试算法在不同观测条件下的稳定性；
* 提供辐射源标签，可计算 V-measure、ARI、AMI、同质性、完整性、Pairwise F1 等聚类指标；
* 官方项目包含 HDBSCAN 基线，可作为无监督方法的复现起点；
* 适合扩展 DBSCAN、K-means、GMM、层次聚类、表征学习 + 聚类等方法。

**建议使用方式：**

1. 先复现 Turing Challenge 中的 HDBSCAN 基线，确认数据读取、标准化、聚类和评价流程正确；
2. 再加入 DBSCAN、K-means、GMM、层次聚类等基础聚类方法，建立自己的基线表；
3. 若研究深度表征学习，可在 PDW 嵌入表示上继续使用 HDBSCAN 或层次聚类进行簇划分；
4. 汇报结果时应同时给出接收模式、输入特征、是否使用标签训练、是否预设辐射源数量以及主要聚类指标。

#### 4.2.2 HDBSCAN 基线

HDBSCAN 是 Turing Deinterleaving Challenge 中提供的主要无监督基线。它直接在 PDW 特征空间中进行密度聚类，不需要预先指定辐射源数量，因此与“未知辐射源数量下的无监督分选”任务设定较为一致。

相比 K-means 和 GMM，HDBSCAN 的优势在于不需要提前给定簇数，并且能够将低密度或不稳定样本识别为噪声点；相比普通 DBSCAN，HDBSCAN 对不同密度簇的适应性更强。对于雷达脉冲分选任务，这一点比较重要，因为不同辐射源的脉冲数量、参数分布和局部密度可能存在较大差异。

**复现和扩展建议：**

* 直接使用原始 PDW 特征时，应先进行特征标准化，避免 TOA、RF、PW、PA 等量纲差异影响距离度量；
* 对比实验中可加入 DBSCAN、K-means、GMM 和层次聚类，区分“未知簇数方法”和“需要给定 K 的方法”；
* 若使用深度模型学习嵌入表示，建议仍保留 HDBSCAN 作为统一聚类后端，便于判断性能提升来自表征学习还是聚类器本身；
* 汇报 HDBSCAN 结果时，应记录关键超参数和噪声点比例，并使用与 Turing Challenge 一致的评价指标。

#### 4.2.3 Transformer 度量学习

“Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning” 将雷达脉冲分选定义为将混叠脉冲序列按辐射源分离的问题，并强调单个脉冲序列中的辐射源数量未知。该方法使用 Transformer 编码 PDW 序列，并通过 triplet loss 学习脉冲嵌入表示，使同一辐射源的脉冲在特征空间中更接近，不同辐射源的脉冲更远。该方向适合作为监督度量学习 + 聚类的代表方法。

#### 4.2.4 最优传输聚类

“Deinterleaving RADAR Emitters with Optimal Transport Distances” 提出一种无监督分选方法。其基本思想是先对脉冲进行过分割，降低不同辐射源被错误合并的风险；随后考虑复杂辐射源可能对应多个簇，使用基于最优传输距离的层次聚类进行簇合并。该方法适合支撑“无监督聚类 + 簇合并”的方法方向。

#### 4.2.5 更新过程混合模型方法

“Deinterleaving of Mixtures of Renewal Processes” 和 “Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures” 都从随机过程混合模型角度处理分选问题。这类方法不是简单在 PDW 空间做聚类，而是利用脉冲到达时间、符号序列和更新过程 / 马尔可夫链结构对混叠脉冲序列进行建模。它们适合作为统计建模类方法参考。

#### 4.2.6 深度对比聚类

“Deep Contrastive Clustering for Signal Deinterleaving” 是对比学习 + 聚类方向的代表论文。该方向的核心思想是通过对比学习增强信号或脉冲表示的可分性，再利用聚类或伪标签机制完成分选。该方法适合放在深度表征学习式分选方法中，而不是开源框架或基准部分。

#### 4.2.7 语义分割式分选

“A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network” 将分选任务转换为语义分割 / 序列标注问题。其输入主要围绕 DTOA 等时间差序列，通过神经网络为脉冲流中的脉冲赋予语义标签，从而完成分选。该类方法适合复杂 PRI 调制环境，但通常依赖有标签训练数据，因此不属于严格无监督聚类式分选。

#### 4.2.8 图像分割式分选

“Image Segmentation for Radar Signal Deinterleaving Using Deep Learning” 将雷达脉冲序列变换为图像化表示，再使用深度图像分割模型完成信号分选。它的价值在于把脉冲分选从传统 PRI 搜索问题转化为二维图像分割问题，但任务设定和输入表示不同于直接 PDW 聚类。

#### 4.2.9 Sep-RefineNet

Sep-RefineNet 是语义分割式分选方法。它通过构造频率特征矩阵来表示不同雷达信号脉冲流的语义结构，再使用 Sep-RefineNet 对矩阵进行像素级分割，最后通过位置解码和验证还原原始脉冲流中的脉冲位置。该方法适合放在分割式分选方法中，而不应归为无监督聚类基线。

#### 4.2.10 Deep ToA mask 递归分选

“Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving” 使用 ToA 掩码和递归分离思想处理混叠雷达脉冲序列。它更强调从到达时间结构中逐步提取不同辐射源对应的脉冲子序列，适合放在深度递归分选 / 基于掩码的分选类别中。

#### 4.2.11 GCN 半监督雷达信号分选

“Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning” 使用图卷积网络和半监督学习建模雷达信号分选问题。该方向适合描述脉冲之间关系图、伪标签和半监督学习在信号分选中的应用，但它不是严格无监督 PDW 辐射源聚类方法。

---

### 4.3 论文与方法详细说明

本节只对上表中已经列出的论文和项目进行展开说明，不额外加入新的论文。对于没有独立论文的项目资源，例如 Turing Challenge 的 HDBSCAN 基线，本节按“项目基线”而不是“论文方法”处理。

#### 4.3.1 The Turing Synthetic Radar Dataset: A Dataset for Pulse Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 数据集 / 基准 / 挑战赛  
**发表形式：** arXiv 预印本，2026  
**方法归属：** 标准 PDW 脉冲分选数据集与评价基准  
**是否严格无监督：** 数据集本身不限定方法；可用于无监督、半监督和监督实验  
**代码 / 数据：** 有公开项目和 Hugging Face 数据集  

</td>
<td width="72%" valign="top">

**论文：** The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving  
**作者：** Edward Gunn, Adam Hosford, Robert Jones, Leo Zeitler, Ian Groves, Victoria Nockles  
**链接：** [arXiv](https://arxiv.org/abs/2602.03856) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

**Highlights**

* 面向雷达脉冲解交织任务构建，是当前少见的公开大规模 PDW 分选数据集；
* 数据包含 stare 和 scan 两类接收配置，可用于分析不同接收模式下的分选性能；
* 提供辐射源标签和配套挑战赛评价流程，便于使用 V-measure、ARI、AMI 等聚类指标进行统一对比；
* 与 Turing Deinterleaving Challenge 配套，提供官方基线和示例流程；
* 适合作为新方法的主基准，而不是只作为补充数据来源。

</td>
</tr>
</table>

**中文摘要：**  
该论文介绍了 Turing Synthetic Radar Dataset / TSRD，这是一个用于雷达脉冲解交织研究的合成 PDW 数据集。数据集面向电子战和信号情报场景中“多个未知辐射源脉冲混叠后如何重新归属到各个辐射源”的问题，提供大规模脉冲列、辐射源标签和不同接收配置。论文同时配套 Turing Deinterleaving Challenge，鼓励研究者在统一的数据、指标和评价流程下比较不同分选方法。对于雷达脉冲分选研究而言，该资源的主要价值不在于提出某个单一算法，而在于提供了一个相对标准化、可复现的公开实验平台。

**适合放置位置：** 数据集与基准 / 主推荐资源。  
**局限性：** 数据为合成数据，仍不能完全替代真实电子支援系统中的实测 PDW；如果使用标签训练模型，需要在论文中明确说明其与严格无监督分选任务的区别。

---

#### 4.3.2 Turing Deinterleaving Challenge HDBSCAN Baseline

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 官方项目基线  
**发表形式：** 无独立论文，依托 Turing Deinterleaving Challenge  
**方法归属：** 无监督密度聚类基线  
**是否严格无监督：** 是  
**代码 / 数据：** 有官方 GitHub 示例  

</td>
<td width="72%" valign="top">

**项目：** Turing Deinterleaving Challenge baseline  
**链接：** [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

**Highlights**

* 直接在原始 PDW 特征上使用 HDBSCAN 聚类；
* 不需要预设辐射源数量，适合未知辐射源数量的分选设定；
* 可作为 TSRD 上最直接、最透明的无监督基线；
* 适合与 DBSCAN、K-means、GMM、层次聚类以及表征学习 + 聚类方法进行对比；
* 便于检验数据读取、标准化、聚类评价流程是否正确。

</td>
</tr>
</table>

**中文摘要：**  
HDBSCAN 基线是 Turing Deinterleaving Challenge 提供的官方无监督参考方法。该方法不依赖深度学习训练，也不需要辐射源类别标签，而是直接在 PDW 特征空间中进行密度聚类。由于 HDBSCAN 能够在一定程度上适应不同密度簇，并可将部分样本视为噪声点，因此比需要预设簇数的 K-means 或 GMM 更贴近未知辐射源数量场景。对于后续方法研究而言，HDBSCAN 基线可以作为最基础的可复现对照，用来判断改进是否真正来自新的特征构造、表征学习或聚类策略。

**适合放置位置：** 推荐基线 / 官方无监督基线。  
**局限性：** 直接在原始 PDW 上聚类时容易受特征尺度、参数重叠和复杂工作模式影响；如果不进行合理标准化，距离度量可能被 TOA、RF 等大尺度特征主导。

---

#### 4.3.3 Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表形式：** arXiv 预印本，2025  
**方法归属：** Transformer 度量学习 / 监督表征学习 + 聚类  
**是否严格无监督：** 否，训练阶段使用标签构造 triplet loss  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning  
**作者：** Edward Gunn, Adam Hosford, Daniel Mannion, Jarrod Williams, Varun Chhabra, Victoria Nockles  
**链接：** [arXiv](https://arxiv.org/abs/2503.13476)

**Highlights**

* 将雷达脉冲分选建模为未知辐射源数量下的脉冲归属问题；
* 使用 Transformer 编码 PDW 序列，利用上下文信息学习脉冲表示；
* 通过 triplet loss 约束同源脉冲更近、异源脉冲更远；
* 推理阶段可在学习到的嵌入空间中进行聚类或相似性分组；
* 适合作为“监督度量学习 + 分选式输出”的代表方法。

</td>
</tr>
</table>

**中文摘要：**  
该论文针对雷达脉冲列中包含多个未知辐射源的场景，提出一种基于 Transformer 的深度度量学习方法。方法使用 Transformer 对 PDW 序列进行建模，希望利用脉冲之间的上下文关系获得更有判别性的嵌入表示。训练阶段采用 triplet loss，使来自同一辐射源的脉冲在特征空间中更加接近，而来自不同辐射源的脉冲距离更远。该方法的重点不是直接设计传统 PRI 搜索规则，而是学习一个适合后续聚类或相似性判断的脉冲嵌入空间。

**适合放置位置：** 监督表征学习 / 度量学习方法。  
**局限性：** 训练阶段依赖标签，因此不能与 HDBSCAN、DBSCAN 等严格无监督方法简单等价比较；更适合作为监督学习性能参考或表征学习上限。

---

#### 4.3.4 Deinterleaving RADAR Emitters with Optimal Transport Distances

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Aerospace and Electronic Systems，2024  
**方法归属：** 无监督聚类 / 最优传输距离 / 层次簇合并  
**是否严格无监督：** 基本符合  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Deinterleaving RADAR Emitters With Optimal Transport Distances  
**作者：** Manon Mottier, Gilles Chardon, Frédéric Pascal  
**链接：** [arXiv](https://arxiv.org/abs/2312.11178) / [DOI](https://doi.org/10.1109/TAES.2024.3367287)

**Highlights**

* 面向未知辐射源数量的雷达信号分选问题；
* 先通过聚类进行过分割，尽量避免不同辐射源被过早合并；
* 再使用最优传输距离度量簇间分布差异；
* 通过层次聚类合并可能属于同一辐射源的多个簇；
* 适合复杂辐射源行为和多模态参数分布场景。

</td>
</tr>
</table>

**中文摘要：**  
该论文提出了一种基于聚类和最优传输距离的无监督雷达辐射源分选方法。方法首先对接收脉冲进行初始聚类，并约束不同辐射源的脉冲尽量不要出现在同一个初始簇中。由于一个复杂辐射源可能因工作模式变化或参数分布差异而被拆分为多个簇，论文进一步使用最优传输距离衡量簇与簇之间的整体分布相似性，并通过层次聚类完成簇合并。该方法的思路适合“先保证纯度、再合并碎片簇”的无监督分选框架。

**适合放置位置：** 无监督聚类方法 / 簇合并方法。  
**局限性：** 初始过分割质量会影响后续合并效果；最优传输距离计算和层次合并会引入额外计算开销。

---

#### 4.3.5 Deinterleaving of Mixtures of Renewal Processes

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Signal Processing，2019  
**方法归属：** 统计建模 / 更新过程混合模型 / 到达时间建模  
**是否严格无监督：** 基本符合，属于模型驱动的源分离方法  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Deinterleaving of Mixtures of Renewal Processes  
**作者：** Jeremy Young, Anders Høst-Madsen, Eva-Marie Nosal  
**链接：** [DOI](https://doi.org/10.1109/TSP.2018.2886149)

**Highlights**

* 从多源脉冲到达时间序列出发，将分选问题建模为更新过程混合分离；
* 主要依赖 impulse spacing / 脉冲间隔信息，而不是多维 PDW 特征聚类；
* 强调在只有时间信息时如何进行源分离；
* 具有较强理论解释性，适合作为统计建模类分选方法参考；
* 可用于理解传统 PRI / TOA 分选问题的概率建模基础。

</td>
</tr>
</table>

**中文摘要：**  
该论文研究多源脉冲序列的解交织问题，将来自不同源的脉冲间隔建模为不同的更新过程。与直接使用 RF、PW、PA 等多维 PDW 特征聚类的方法不同，该方法重点关注仅基于脉冲到达间隔进行源分离的问题。论文从多输入单输出系统中的 impulsive source separation 出发，建立更新过程混合模型，并利用脉冲间隔统计规律对不同来源的事件进行划分。对于雷达分选研究而言，该论文的价值在于提供了一个更偏理论和统计建模的视角。

**适合放置位置：** 统计建模类方法 / TOA-PRI 理论方法。  
**局限性：** 方法主要依赖时间间隔建模，对复杂参数捷变、多参数耦合或高重叠 PDW 分布的适应性需要结合具体场景分析。

---

#### 4.3.6 Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Signal Processing，2024  
**方法归属：** 离散更新过程 / 马尔可夫链混合模型 / 惩罚似然优化  
**是否严格无监督：** 基本符合  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures  
**作者：** Jean Pinsolle, Olivier Goudet, Cyrille Enderli, Sylvain Lamprier, Jin-Kao Hao  
**链接：** [arXiv](https://arxiv.org/abs/2402.09166) / [DOI](https://doi.org/10.1109/TSP.2024.3464753)

**Highlights**

* 将混叠序列建模为多个离散更新马尔可夫链的混合；
* 使用惩罚似然分数进行分选优化；
* 同时利用符号序列信息和到达时间信息；
* 给出大样本条件下恢复真实划分的理论分析；
* 面向 RESM / 电子支援测量场景验证方法有效性。

</td>
</tr>
</table>

**中文摘要：**  
该论文提出一种面向离散更新过程混合的解交织方法，并将其应用于雷达电子支援测量场景。方法将混叠观测看作多个离散更新马尔可夫链的叠加，通过最大化惩罚似然分数来恢复不同过程对应的划分。与只利用到达时间间隔的模型相比，该方法同时考虑不同符号的序列信息和到达时间信息，并给出理论分析说明在一定条件下可以恢复真实划分。该方法适合作为“随机过程建模 + 理论分析 + 电子支援应用”的代表工作。

**适合放置位置：** 统计建模类方法 / 离散序列分选方法。  
**局限性：** 模型假设和似然形式对实际复杂雷达场景的适配需要进一步结合数据验证；相较普通聚类方法，理解和实现门槛更高。

---

#### 4.3.7 Deep Contrastive Clustering for Signal Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Aerospace and Electronic Systems，2024  
**方法归属：** 自监督对比学习 / 深度聚类 / 表征学习  
**是否严格无监督：** 更接近自监督 / 无先验信息设定，具体实现需看伪标签构造  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Deep Contrastive Clustering for Signal Deinterleaving  
**作者：** Shuyuan Yang, Xinyi Zhao, Huiling Liu, Chen Yang, Tongqing Peng, Rundong Li, Feng Zhang  
**链接：** [DOI](https://doi.org/10.1109/TAES.2023.3322971)

**Highlights**

* 提出 DCCA（Deep Contrastive Clustering Algorithm）用于雷达信号分选；
* 使用自监督对比学习框架学习更具判别性的信号表示；
* 通过增强信号的自定义伪标签构造监督信息；
* 目标是在缺少辐射源先验信息的情况下完成信号分选；
* 适合作为对比学习 + 聚类方向的代表论文。

</td>
</tr>
</table>

**中文摘要：**  
该论文针对复杂电磁环境下雷达信号分选困难的问题，提出深度对比聚类算法 DCCA。方法首先构建对比式自监督深度注意力网络，通过对增强信号生成自定义伪标签来学习信号表示，使同类或相似信号在特征空间中更加聚集，不同信号更加分离。随后结合聚类机制完成信号分选。该方法的主要价值在于将对比学习用于雷达信号解交织，使模型能够在缺少明确辐射源先验信息的情况下学习可分表示。

**适合放置位置：** 深度表征学习 / 对比学习分选方法。  
**局限性：** 是否严格无监督取决于伪标签和数据增强的设计；如果伪标签构造依赖较强假设，则需要在综述或实验中说明。

---

#### 4.3.8 A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Signal Processing，2022  
**方法归属：** 序列建模 / 语义分割式分选 / 监督学习  
**是否严格无监督：** 否  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network  
**作者：** Wang Chao, Sun Liting, Liu Zhangmeng, Huang Zhitao  
**链接：** [arXiv](https://arxiv.org/abs/2110.13706) / [DOI](https://doi.org/10.1109/TSP.2022.3229630)

**Highlights**

* 提出 SSD（semantic segmentation deinterleaving）思想；
* 将 DTOA 序列输入神经网络进行序列建模；
* 将每个脉冲按照语义类别进行标注，从而完成分选；
* 不需要传统方法中的多轮 PRI 搜索与合并；
* 对复杂 PRI 调制、脉冲丢失和噪声脉冲具有一定鲁棒性。

</td>
</tr>
</table>

**中文摘要：**  
该论文将雷达信号分选问题转换为语义分割 / 序列标注问题，提出语义分割式分选方法 SSD。方法以脉冲流的到达时间差 DTOA 为主要输入，通过序列建模神经网络学习不同雷达信号类型在序列中的语义模式，并为脉冲流中的每个脉冲分配语义标签，从而实现分选。与传统 PRI 搜索方法相比，该方法不需要显式搜索 PRI 或 PRI 周期，也不需要对多脉冲周期信号进行多轮搜索和合并。论文还指出，循环神经网络在该序列建模任务中相较卷积神经网络更有优势。

**适合放置位置：** 监督式序列标注 / 语义分割式分选。  
**局限性：** 训练阶段依赖标签，不属于严格无监督分选；输出类别与训练数据标签空间相关，泛化到未知辐射源时需要额外设计。

---

#### 4.3.9 Image Segmentation for Radar Signal Deinterleaving Using Deep Learning

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Aerospace and Electronic Systems，2023  
**方法归属：** 图像化表示 / 深度图像分割 / 监督学习  
**是否严格无监督：** 否  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Image Segmentation for Radar Signal Deinterleaving Using Deep Learning  
**作者：** Mustafa Atahan Nuhoglu, Yasar Kemal Alp, Mehmet Ege Can Ulusoy, Hakan Ali Çirpan  
**链接：** [DOI](https://doi.org/10.1109/TAES.2022.3188225)

**Highlights**

* 将雷达信号分选问题转化为图像分割问题；
* 通过预处理将脉冲序列转换为二维图像或矩阵表示；
* 使用深度分割网络提取图像结构中的 PRI / TOA 规律；
* 适合借鉴 U-Net 等图像分割网络结构；
* 提供了从传统序列分选到视觉分割建模的思路迁移。

</td>
</tr>
</table>

**中文摘要：**  
该论文尝试将雷达信号解交织任务转化为图像分割任务。其基本思路是先根据脉冲到达时间、PRI 或相关变换构造二维图像表示，使不同雷达信号的时间规律在图像空间中呈现出可分结构；随后利用深度图像分割模型对这些结构进行像素级或区域级划分，从而实现对不同辐射源脉冲的分离。该方法的意义在于把雷达分选从传统的一维 PRI 搜索问题扩展到二维图像分割问题，可以借鉴计算机视觉中成熟的分割网络和训练策略。

**适合放置位置：** 图像分割式分选 / 深度学习监督方法。  
**局限性：** 需要设计脉冲序列到图像的转换方式；训练通常依赖标签；分割结果需要进一步映射回原始脉冲序列。

---

#### 4.3.10 Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** Applied Sciences，2023  
**方法归属：** 频率特征矩阵 / 语义分割网络 / 位置解码  
**是否严格无监督：** 否  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation  
**作者：** Yongjiang Mao, Wenjuan Ren, Xipeng Li, Zhanpeng Yang, Wei Cao  
**链接：** [Paper](https://www.mdpi.com/2076-3417/13/4/2726) / [DOI](https://doi.org/10.3390/app13042726)

**Highlights**

* 构造 FCM（frequency characteristic matrix）编码雷达脉冲流的 PRI 语义特征；
* 使用 Sep-RefineNet 对 FCM 进行像素级语义分割；
* 通过位置解码和验证将分割结果映射回原始脉冲流；
* 减少传统算法对人工阈值判断和脉冲序列搜索的依赖；
* 在混叠脉冲和脉冲缺失条件下具有一定抗噪能力。

</td>
</tr>
</table>

**中文摘要：**  
该论文提出 Sep-RefineNet，一种基于语义分割的雷达信号解交织方法。方法首先构造频率特征矩阵 FCM，用于编码不同雷达脉冲流的 PRI 语义结构；然后使用 Sep-RefineNet 语义分割网络对该矩阵进行像素级分割；最后通过位置解码和验证得到原始脉冲流中的脉冲位置，从而完成分选。该方法试图减少传统分选算法对人工阈值、PRI 搜索和经验规则的依赖，并提升在混叠脉冲和脉冲缺失条件下的鲁棒性。

**适合放置位置：** 语义分割式分选 / 频率矩阵编码方法。  
**局限性：** 属于监督式分割建模，输入依赖特定 FCM 构造流程；不适合作为严格无监督 PDW 聚类基线。

---

#### 4.3.11 Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Transactions on Aerospace and Electronic Systems，2023  
**方法归属：** ToA 掩码预测 / 递归分离 / 深度学习  
**是否严格无监督：** 通常不是，训练阶段依赖仿真或标签监督  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving  
**作者：** Haoran Xiang, Furao Shen, Jian Zhao  
**链接：** [DOI](https://doi.org/10.1109/TAES.2022.3193948)

**Highlights**

* 提出 Deep ToA Mask / Recursive Deinterleaving Network；
* 将 ToA 脉冲列映射到适合分离的特征空间；
* 估计每个雷达辐射源对应的 ToA 系数掩码；
* 采用递归方式逐步从混叠脉冲列中分离辐射源；
* 适合讨论复杂混叠、PRI 抖动和脉冲缺失条件下的深度递归分选。

</td>
</tr>
</table>

**中文摘要：**  
该论文针对复杂电磁环境中多种模式雷达信号密集混叠的问题，提出一种基于 Deep ToA Mask 的递归雷达脉冲分选方法。方法通过递归分选网络将 ToA 脉冲列映射到特征空间，并结合局部和全局上下文信息估计不同雷达辐射源对应的 ToA 系数掩码。与一次性输出全部簇标签的方法不同，该方法采用递归方式逐步分离脉冲子序列，从而降低强混叠场景下的分选难度。论文强调该方法可处理参数重叠、脉冲缺失和 PRI 抖动等复杂情况。

**适合放置位置：** 深度递归分选 / ToA 掩码方法。  
**局限性：** 递归分离容易受到前序错误影响；对于多模式辐射源，可能需要额外的调制类型或 PRI 信息辅助。

---

#### 4.3.12 Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning

<table>
<tr>
<td width="28%" valign="top">

**资源类型：** 代表性论文  
**发表期刊：** IEEE Signal Processing Letters，2025  
**方法归属：** 图卷积网络 / 半监督学习 / 关系建模  
**是否严格无监督：** 否  
**代码 / 数据：** 未发现官方开源代码  

</td>
<td width="72%" valign="top">

**论文：** Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning  
**作者：** Ziying Li, Xiongjun Fu, Jian Dong, Min Xie  
**链接：** [DOI](https://doi.org/10.1109/LSP.2024.3519884)

**Highlights**

* 面向标注脉冲难以大量获取的问题，引入半监督学习框架；
* 使用图卷积网络建模脉冲样本之间的关系；
* 将脉冲看作图节点，将相似性或邻接关系建模为边；
* 利用少量标注样本和大量未标注样本提升分选性能；
* 适合作为半监督关系建模类分选方法代表。

</td>
</tr>
</table>

**中文摘要：**  
该论文关注实际雷达侦察系统中标注脉冲难以大量获取的问题，提出一种结合图卷积网络和半监督学习的雷达信号分选方法。方法将脉冲样本建模为图中的节点，通过节点特征和边关系传播信息，使少量标注样本的监督信号能够扩展到未标注样本。与完全监督的深度学习方法相比，该方法更强调在标签稀缺条件下利用样本之间的结构关系；与传统无监督聚类相比，它仍然依赖部分标签，因此属于半监督分选方法。

**适合放置位置：** 半监督学习 / 图神经网络分选方法。  
**局限性：** 不是严格无监督方法；图构建方式、边权设计和少量标签质量会显著影响最终性能。

---

## 5. 推荐实验设置

如果研究目标是 **无监督雷达脉冲分选**，建议以 TSRD 和 Turing Deinterleaving Challenge 为主实验平台，并围绕 HDBSCAN 基线构建可复现对比。

### 5.1 主基准

建议将 **Turing Synthetic Radar Dataset / TSRD** 作为主要基准。该数据集提供 PDW 脉冲序列、辐射源标签和配套评价流程，适合比较传统聚类、表征学习和监督式上限模型。

实验设置中建议明确以下内容：

* 使用的接收模式，例如 stare 或 scan；
* 输入特征集合，例如 TOA、CF/RF、PW、PA、AoA；
* 是否使用辐射源标签参与训练；
* 是否预设辐射源数量 K；
* 聚类方法、距离度量和主要超参数；
* 是否对 TOA、RF、PW、PA 等特征进行标准化或变换。

### 5.2 建议基线

| 类别 | 方法 | 用途 |
| --- | --- | --- |
| 官方基线 | HDBSCAN | 复现 Turing Challenge 的基础结果 |
| 原始特征聚类 | DBSCAN，K-means，GMM，层次聚类 | 评估传统聚类方法在原始 PDW 特征上的表现 |
| 基于 PRI 的方法 | PRI 直方图，CDIF，SDIF，PRI 变换 | 对比传统时间结构分选方法 |
| 混合方法 | RF/PW/DOA 粗聚类 + PRI 精细分选 | 验证多特征粗分组与 PRI 精细分选的组合效果 |
| 表征学习 | 自编码器 / 对比编码器 + HDBSCAN | 评估学习型嵌入表示是否提升聚类可分性 |
| 监督式上限 | Transformer 度量学习，GCN，TCN，序列标注模型 | 作为监督或半监督方法的性能参考 |

### 5.3 建议流程

```text
原始 PDW 序列
      ↓
特征选择与标准化
      ↓
可选的时序特征构造
      ↓
聚类或表征学习
      ↓
簇标签分配
      ↓
使用 V-measure / ARI / AMI / Pairwise F1 进行评价
```

对于无监督方法，建议首先完成 `原始 PDW 特征 + HDBSCAN` 的复现，再逐步加入新的特征构造、聚类策略或表征学习模块。这样可以避免直接比较复杂模型时无法判断性能提升来源。

---

## 6. 非核心资源说明

为保证本资源列表质量，以下资源不纳入核心推荐。部分资源可能与雷达信号处理相关，但由于任务不匹配、文档不足、复现价值有限或项目质量不适合作为正式研究资源，因此不放入主表。

| 资源 | 处理方式 | 原因 |
| --- | --- | --- |
| [Stream-ConAEnet](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet) | 不纳入正文推荐 | 项目说明中明确属于本科毕设内容，数据说明、运行流程、评价协议和论文支撑不足 |
| [EW Signal Intelligence Deinterleaving Demo](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence) | 不纳入正文推荐 | 教学型小 demo，适合理解概念，但不适合作为研究级基准或代表性方法 |
| [radar_data_Kmeans](https://github.com/zda2019/radar_data_Kmeans) | 不纳入核心资源 | 与雷达分选相关，但文档、评价协议和数据说明较弱，可作为工程实现线索 |
| [HMC-RFN GitHub repository](https://github.com/xm980426/HMC-RFN) | 不纳入核心资源 | 有 MATLAB 代码，但 README 和复现说明较少，可作为论文配套代码线索 |
| [2nd-EBDSC repository](https://github.com/framist/2nd-EBDSC) | 不纳入核心资源 | 工程内容较完整，但任务更接近竞赛式监督序列建模或模板辅助信号提取，不是标准无监督 PDW 辐射源聚类 |

---

## 7. 说明

* 与通用 RF modulation recognition 或 RF fingerprinting 相比，雷达脉冲分选方向的高质量开源资源仍然较少；
* 大多数公开数据集是合成数据，因为真实雷达 PDW 或 IQ 数据很难公开，也很难获得可靠真值标签；
* 报告实验结果时，应明确说明方法属于无监督、监督、半监督，还是“监督表征学习后再聚类”；
* 如果研究目标是严格的无监督雷达脉冲分选，TSRD + HDBSCAN / DBSCAN / K-means / 层次聚类是当前最透明、最容易复现的起点；
* 本仓库中的星级是整理者基于任务匹配度、资源质量、复现价值和数据说明完整度给出的主观评价，不代表 GitHub stars。

---

## 8. 引用与贡献

本资源列表由 **厦门大学信息学院 SmartDSP 实验室** 整理与维护，旨在为雷达脉冲分选、雷达信号分选、PDW 数据处理以及相关电磁信号智能处理研究提供一个相对系统、可复现、便于扩展的开源资源索引。

如果本仓库对你的研究或项目有帮助，欢迎在论文、报告或项目中引用本资源列表，并尽量同时引用对应的原始论文、数据集和代码仓库。

### 引用格式

如果你使用了本仓库整理的资源列表，可以采用如下方式引用：

```bibtex
@misc{smartdsp_radar_deinterleaving_resources,
  title        = {Radar Pulse Deinterleaving Resources},
  author       = {SmartDSP Lab, School of Informatics, Xiamen University},
  year         = {2026},
  howpublished = {\url{https://github.com/your-repository-url}},
  note         = {A curated resource list for radar pulse deinterleaving and radar signal sorting}
}
```

其中 `https://github.com/your-repository-url` 请替换为本仓库的实际 GitHub 地址。

### 贡献方式

欢迎研究者和开发者通过 issue 或 pull request 的方式补充和修正本仓库内容。可以贡献的内容包括但不限于：

* 新发布的雷达脉冲分选数据集；
* 基于 PRI、聚类或深度学习的分选方法开源实现；
* 可复现的基准结果；
* 公开论文、代码和数据集链接；
* 对已有资源的数据可用性、任务定义或推荐程度的修正；
* 关于某个方法是否严格符合无监督雷达脉冲分选定义的补充说明。

在贡献新资源时，建议尽量说明以下信息：

```text
资源名称：
资源链接：
任务类型：
方法类型：
是否开源代码：
是否开源数据：
是否包含标签：
是否严格无监督：
推荐理由：
备注：
```

本仓库将持续更新雷达脉冲分选相关公开资源，也欢迎相关方向的研究者共同完善。
