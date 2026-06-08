# Radar Pulse Deinterleaving Resources

<p align="center">
  <a href="./README.md">简体中文</a> |
  <b>English</b>
</p>

This repository is curated by **SmartDSP Lab, School of Informatics, Xiamen University**. It collects high-value public resources related to **Radar Pulse Deinterleaving / Radar Signal Sorting**, with a focus on task definitions, PDW datasets, reproducible experimental frameworks, representative methods, and evaluation metrics.

This resource list prioritizes resources that are highly relevant to radar pulse deinterleaving, have clear task definitions, public data or code, paper support, or strong reproducibility value. Projects that are only loosely related by name, lack documentation, have limited reproducibility value, or are not suitable as formal research resources are not included in the main recommendations.

The goal of radar pulse deinterleaving is to partition an interleaved pulse stream according to different radar emitters. In a typical **PDW（Pulse Description Word）** setting, each pulse is usually described by parameters such as **TOA（Time of Arrival）**, **PRI（Pulse Repetition Interval）**, **RF/CF（Radio Frequency / Carrier Frequency）**, **PW（Pulse Width）**, **PA（Pulse Amplitude）**, and **DOA/AOA（Direction / Angle of Arrival）**.

---

## Quick Takeaways

* **Recommended primary benchmark:** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) and [Turing Synthetic Radar Dataset / TSRD](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset).
* **Recommended baseline:** The HDBSCAN baseline provided in the Turing Challenge, which can serve as a starting point for strictly unsupervised radar pulse deinterleaving.
* **Strictly unsupervised deinterleaving:** Traditional clustering methods such as DBSCAN, HDBSCAN, hierarchical clustering, K-means, and GMM best match the definition of emitter clustering when the number of emitters is unknown.
* **Deep learning methods:** Transformer-based metric learning, deep contrastive clustering, semantic segmentation, graph convolutional networks, and related methods are valuable research directions, but many rely on labeled training data. They are better treated as representative methods or upper-bound references.
* **Public data status:** Publicly available real radar PDW / IQ deinterleaving datasets are scarce. Most reproducible experiments currently rely on synthetic PDW data.

---

## Table of Contents

* [1. Task Overview](#1-task-overview)

  * [1.1 What Is Radar Pulse Deinterleaving?](#11-what-is-radar-pulse-deinterleaving)
  * [1.2 Common Input Features](#12-common-input-features)
  * [1.3 Common Evaluation Metrics](#13-common-evaluation-metrics)
* [2. Method Taxonomy and Task Boundaries](#2-method-taxonomy-and-task-boundaries)

  * [2.1 Overview of Method Types](#21-overview-of-method-types)
  * [2.2 Relationship Between Methods and Strictly Unsupervised Deinterleaving](#22-relationship-between-methods-and-strictly-unsupervised-deinterleaving)
* [3. Datasets and Benchmarks](#3-datasets-and-benchmarks)

  * [3.1 Dataset Overview](#31-dataset-overview)
  * [3.2 Turing Synthetic Radar Dataset / TSRD](#32-turing-synthetic-radar-dataset--tsrd)
  * [3.3 RadSeg](#33-radseg)
* [4. Representative Methods and Papers](#4-representative-methods-and-papers)

  * [4.1 Paper Overview](#41-paper-overview)
  * [4.2 Method Summaries](#42-method-summaries)
  * [4.3 Detailed Notes on Papers and Methods](#43-detailed-notes-on-papers-and-methods)
* [5. Recommended Experimental Setup](#5-recommended-experimental-setup)

  * [5.1 Primary Benchmark](#51-primary-benchmark)
  * [5.2 Recommended Baselines](#52-recommended-baselines)
  * [5.3 Recommended Pipeline](#53-recommended-pipeline)
* [6. Notes on Non-Core Resources](#6-notes-on-non-core-resources)
* [7. Notes](#7-notes)
* [8. Citation and Contribution](#8-citation-and-contribution)

---

## 1. Task Overview

### 1.1 What Is Radar Pulse Deinterleaving?

Radar pulse deinterleaving, also known as **radar pulse deinterleaving** or **radar signal sorting**, refers to the task of separating an interleaved pulse sequence collected in a complex electromagnetic environment into pulse groups corresponding to different radar emitters.

<p align="center">
  <img src="./assets/figures/fig1.png" width="90%">
</p>

<p align="center">
  <b>Figure 1.</b> Illustration of radar pulse deinterleaving: pulse sequences from multiple emitters are received by radar warning equipment and form an interleaved pulse stream. A deinterleaving algorithm needs to regroup the pulses into clusters corresponding to different emitters.
</p>

Given an interleaved pulse sequence:

```text
P = {p1, p2, ..., pN}
```

each pulse `pi` can usually be represented as a PDW feature vector:

```text
pi = [TOA, RF/CF, PW, PA, DOA/AOA, ...]
```

The goal of deinterleaving is to output a partition:

```text
C = {C1, C2, ..., CK}
```

where each cluster `Ck` corresponds to one radar emitter.

Strictly speaking, radar pulse deinterleaving can be regarded as an **unsupervised clustering problem**:

* the number of emitters is usually unknown;
* emitter identities are not fixed categories;
* pulses generated by the same emitter should be assigned to the same cluster;
* pulses generated by different emitters should be separated.

At the same time, many modern methods formulate the task as supervised learning, semi-supervised learning, metric learning, sequence labeling, or semantic segmentation. These methods may produce deinterleaving-style outputs, but they are not necessarily strictly unsupervised clustering-based deinterleaving methods.

---

### 1.2 Common Input Features

| Feature    | Meaning                                                   | Usage                                                                     |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| TOA        | Time of Arrival                                           | Used to compute PRI and analyze temporal patterns                         |
| PRI / DTOA | Pulse Repetition Interval / Difference of Time of Arrival | Core features in traditional deinterleaving methods                       |
| RF / CF    | Radio Frequency / Carrier Frequency                       | Used to distinguish fixed-frequency or frequency-agile radars             |
| PW         | Pulse Width                                               | Helps distinguish emitters with different waveform parameters             |
| PA / AMP   | Pulse Amplitude                                           | Auxiliary feature, but easily affected by propagation paths and receivers |
| DOA / AOA  | Direction / Angle of Arrival                              | A strong spatial discriminative feature when available                    |

---

### 1.3 Common Evaluation Metrics

| Metric       | Description                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| V-measure    | Harmonic mean of homogeneity and completeness                                                        |
| Homogeneity  | Whether each predicted cluster mainly contains pulses from a single true emitter                     |
| Completeness | Whether pulses from the same true emitter are assigned to the same predicted cluster                 |
| ARI          | Adjusted Rand Index                                                                                  |
| AMI          | Adjusted Mutual Information                                                                          |
| Pairwise F1  | F1 score obtained by treating whether any two pulses belong to the same emitter as a binary decision |
| MCC          | Matthews Correlation Coefficient, usable for pairwise matching evaluation                            |

---

## 2. Method Taxonomy and Task Boundaries

This section describes common technical routes in radar pulse deinterleaving and clarifies the relationship between different methods and the standard definition of **radar pulse deinterleaving / emitter clustering**. The taxonomy is not identical to the list of open-source resources; the subsequent resource tables only include items with clear papers, data, or project links.

### 2.1 Overview of Method Types

| Method Type                                    | Task Relevance | Description                                                                                                                                     |
| ---------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PRI-based traditional methods                  | High           | Mainly exploit TOA / PRI periodic structures, such as PRI histograms, CDIF, SDIF, and PRI transform                                             |
| Traditional clustering methods                 | High           | K-means, GMM, DBSCAN, HDBSCAN, hierarchical clustering, etc.; can be directly applied to PDW feature clustering                                 |
| Density-based clustering baselines             | High           | HDBSCAN / DBSCAN are suitable for unknown-cluster-number scenarios and are commonly used as basic unsupervised deinterleaving baselines         |
| Metric learning + clustering                   | Medium to high | Learn pulse embeddings through supervised or weakly supervised training, then cluster in the embedding space                                    |
| Optimal transport clustering                   | Medium to high | First over-segment pulses, then merge clusters using distributional distances or optimal transport distances                                    |
| Contrastive learning / representation learning | Medium         | Use contrastive learning, autoencoders, or representation learning to improve feature separability; the exact task setting depends on the paper |
| Sequence labeling / semantic segmentation      | Medium         | Formulate pulse deinterleaving as sequence labeling or semantic segmentation, usually requiring labeled training data                           |
| Radar activity segmentation                    | Related task   | Focuses on radar activity detection or segmentation; not equivalent to standard PDW emitter clustering                                          |

### 2.2 Relationship Between Methods and Strictly Unsupervised Deinterleaving

| Method Type                                 | Strictly Unsupervised? | Description                                                                                                                             |
| ------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| DBSCAN / HDBSCAN                            | Yes                    | Do not require predefined emitter labels and are suitable for unknown numbers of emitters                                               |
| K-means / GMM                               | Yes                    | Unsupervised clustering methods, but usually require the number of clusters K to be preset or estimated                                 |
| Hierarchical clustering                     | Yes                    | Suitable for constructing pulse similarity hierarchies; pruning criteria affect final results                                           |
| PRI histogram / CDIF / SDIF / PRI transform | Basically yes          | Traditional model-driven methods, suitable for pulse sequences with clear PRI structures                                                |
| Optimal transport clustering                | Basically yes          | Mainly based on unsupervised clustering and cluster merging, suitable for modeling complex PDW distributions                            |
| Metric learning + clustering                | No                     | Training usually uses labels, although inference can produce emitter groups via clustering                                              |
| Contrastive learning + clustering           | Depends on the setting | Self-supervised settings may approach unsupervised deinterleaving, whereas supervised contrastive learning is not strictly unsupervised |
| Semantic segmentation / sequence labeling   | No                     | Usually relies on dense labels during training and belongs to supervised deinterleaving modeling                                        |
| Radar activity segmentation                 | No                     | Mainly solves radar activity detection or segmentation, not standard emitter clustering                                                 |

---

## 3. Datasets and Benchmarks

### 3.1 Dataset Overview

| Dataset                                                                                                                       | Task Relevance                            | Data Type                                        | Labels             | Open Source?                                           | Recommendation | Notes                                                   |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------ | ------------------ | ------------------------------------------------------ | -------------- | ------------------------------------------------------- |
| [Turing Synthetic Radar Dataset / TSRD](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) | Standard PDW pulse deinterleaving         | Synthetic PDW pulse sequences                    | Emitter labels     | Yes, requires accepting Hugging Face access conditions | ⭐⭐⭐⭐⭐          | Currently the most recommended primary benchmark        |
| [RadSeg](https://github.com/abcxyzi/radseg)                                                                                   | Related task: radar activity segmentation | Complex baseband IQ sequences + per-sample masks | Segmentation masks | Yes                                                    | ⭐⭐             | Formal dataset, but not standard PDW emitter clustering |

---

### 3.2 Turing Synthetic Radar Dataset / TSRD

**Recommendation:** ⭐⭐⭐⭐⭐  
**Task:** Radar pulse deinterleaving / PDW emitter clustering  
**Data type:** Synthetic PDW pulse sequences  
**Labels:** Emitter labels  
**Project:** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

TSRD is one of the most suitable public datasets for use as a primary benchmark in radar pulse deinterleaving. It targets interleaved radar pulse sequences and provides PDW sequences, emitter labels, two receiver modes — stare and scan — and a corresponding evaluation framework.

**Reasons for recommendation:**

* clear task definition directly targeting radar pulse deinterleaving;
* relatively large-scale data suitable for evaluating unsupervised, semi-supervised, and supervised methods;
* provides ground-truth emitter labels, enabling metrics such as V-measure, ARI, AMI, homogeneity, and completeness;
* paired with the Turing Deinterleaving Challenge, making baseline reproduction convenient;
* suitable as the primary benchmark for new methods.

**Usage notes:**

* the data is synthetic and is not equivalent to real measured radar PDW data;
* the dataset files are large, and downloading and processing them require sufficient computing and storage resources;
* when using supervised models, the difference between supervised training and strictly unsupervised deinterleaving should be clearly stated.

---

### 3.3 RadSeg

**Recommendation:** ⭐⭐  
**Task:** Radar pulse activity segmentation  
**Data type:** Complex baseband IQ samples  
**Labels:** Per-channel segmentation masks  
**Repository:** [RadSeg](https://github.com/abcxyzi/radseg)

RadSeg is a dataset for radar activity detection and segmentation. It provides IQ sequences and per-sample segmentation masks. The dataset is related to radar signal analysis, but its task objective is not standard PDW emitter clustering.

**Reasons for inclusion:**

* relatively formal dataset with paper support;
* data and task descriptions are relatively complete;
* useful for research based on raw IQ signals or segmentation-style modeling.

**Task boundary:**

* RadSeg addresses radar pulse activity segmentation, not PDW emitter clustering;
* its annotation format is segmentation masks rather than emitter-level pulse labels;
* it is not suitable for direct comparison with standard deinterleaving datasets such as Turing TSRD.

---

## 4. Representative Methods and Papers

This section collects representative methods with clear paper sources. The code / project column only marks public projects provided by paper authors, dataset providers, or official task organizers.

Among them, **Turing Synthetic Radar Dataset / Turing Deinterleaving Challenge** includes a paper, dataset, baseline code, and evaluation framework, making it currently the most suitable starting point for reproducible experiments. Therefore, this section separately describes this resource and its HDBSCAN baseline.

### 4.1 Paper Overview

| Method / Direction                         | Representative Paper                                                                                | Paper Link                                                                                                                                  | Code / Project                                                                     | Task Relevance | Description                                                                                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Benchmark / dataset                        | The Turing Synthetic Radar Dataset: A Dataset for Pulse Deinterleaving                              | [arXiv](https://arxiv.org/abs/2602.03856) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | High           | Provides data, labels, evaluation metrics, and official baselines; suitable as the primary benchmark                                            |
| HDBSCAN baseline                           | Turing Deinterleaving Challenge baseline                                                            | [Project](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)                                                         | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | High           | Official reproducible unsupervised baseline based directly on density clustering over PDW features                                              |
| Transformer metric learning                | Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning                              | [arXiv](https://arxiv.org/abs/2503.13476)                                                                                                   | Paper only; no open-source code                                                    | High           | Uses Transformer + triplet loss to learn pulse embeddings; used for emitter clustering during inference                                         |
| Optimal transport clustering               | Deinterleaving RADAR Emitters with Optimal Transport Distances                                      | [arXiv](https://arxiv.org/abs/2312.11178)                                                                                                   | Paper only; no open-source code                                                    | High           | Unsupervised clustering + cluster merging with optimal transport distances; suitable for emitter deinterleaving under complex PDW distributions |
| Renewal process mixture model              | Deinterleaving of Mixtures of Renewal Processes                                                     | [DOI](https://doi.org/10.1109/TSP.2018.2886149)                                                                                             | Paper only; no open-source code                                                    | Medium to high | Models pulse sequence deinterleaving from the perspective of stochastic process mixtures                                                        |
| Discrete renewal process mixture model     | Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures | [arXiv](https://arxiv.org/abs/2402.09166) / [DOI](https://doi.org/10.1109/TSP.2024.3464753)                                                 | Paper only; no open-source code                                                    | Medium to high | Discrete renewal Markov-chain deinterleaving method for electronic support scenarios                                                            |
| Deep contrastive clustering                | Deep Contrastive Clustering for Signal Deinterleaving                                               | [DOI](https://doi.org/10.1109/TAES.2023.3322971)                                                                                            | Paper only; no open-source code                                                    | Medium to high | Supports the contrastive learning + clustering direction                                                                                        |
| Semantic-segmentation-based deinterleaving | A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network             | [arXiv](https://arxiv.org/abs/2110.13706) / [DOI](https://doi.org/10.1109/TSP.2022.3229630)                                                 | Paper only; no open-source code                                                    | Medium         | Formulates deinterleaving as sequence semantic labeling / segmentation, usually requiring labeled training data                                 |
| Image-segmentation-based deinterleaving    | Image Segmentation for Radar Signal Deinterleaving Using Deep Learning                              | [DOI](https://doi.org/10.1109/TAES.2022.3188225)                                                                                            | Paper only; no open-source code                                                    | Medium         | Converts radar pulse sequences into image representations and performs deinterleaving through image segmentation                                |
| Sep-RefineNet                              | Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation             | [Paper](https://www.mdpi.com/2076-3417/13/4/2726)                                                                                           | Paper only; no open-source code                                                    | Medium         | Constructs a frequency feature matrix and uses a semantic segmentation network to decode pulse positions in the pulse stream                    |
| Deep ToA mask                              | Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving                                            | [DOI](https://doi.org/10.1109/TAES.2022.3193948)                                                                                            | Paper only; no open-source code                                                    | Medium to high | Uses ToA masks and recursive separation to gradually separate interleaved pulse sequences                                                       |
| GCN semi-supervised sorting                | Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning                   | [DOI](https://doi.org/10.1109/LSP.2024.3519884)                                                                                             | Paper only; no open-source code                                                    | Medium         | Semi-supervised graph-learning-based radar signal sorting; not strictly unsupervised clustering                                                 |

---

### 4.2 Method Summaries

#### 4.2.1 Turing Synthetic Radar Dataset / Turing Deinterleaving Challenge

Turing Synthetic Radar Dataset / TSRD is one of the most suitable public datasets for use as a primary benchmark in radar pulse deinterleaving. The dataset uses PDW pulse sequences as the basic input and provides emitter labels, enabling evaluation of whether algorithms can regroup interleaved pulse streams into clusters corresponding to different emitters.

Unlike resources that only provide data files, the Turing Deinterleaving Challenge also provides usage instructions, an evaluation pipeline, and an HDBSCAN baseline. It is therefore more suitable as a reproducible experimental platform. New methods can be compared with the official baseline or self-implemented clustering methods under the same data split, evaluation protocol, and metric system.

**Reproducibility value:**

* the dataset, task definition, and evaluation metrics are relatively complete;
* stare / scan receiving modes are provided, making it possible to test algorithm stability under different observation conditions;
* emitter labels are provided, enabling V-measure, ARI, AMI, homogeneity, completeness, Pairwise F1, and other clustering metrics;
* the official project includes an HDBSCAN baseline, which can serve as a starting point for unsupervised method reproduction;
* suitable for extending DBSCAN, K-means, GMM, hierarchical clustering, representation learning + clustering, and related methods.

**Recommended usage:**

1. First reproduce the HDBSCAN baseline in the Turing Challenge to verify that data loading, standardization, clustering, and evaluation are correct.
2. Then add basic clustering methods such as DBSCAN, K-means, GMM, and hierarchical clustering to build a baseline table.
3. If studying deep representation learning, continue to use HDBSCAN or hierarchical clustering on the learned PDW embeddings for cluster assignment.
4. When reporting results, specify the receiver mode, input features, whether labels are used for training, whether the number of emitters is preset, and the main clustering metrics.

#### 4.2.2 HDBSCAN Baseline

HDBSCAN is the main unsupervised baseline provided in the Turing Deinterleaving Challenge. It performs density clustering directly in the PDW feature space and does not require the number of emitters to be specified in advance. Therefore, it is relatively consistent with the task setting of “unsupervised deinterleaving with an unknown number of emitters.”

Compared with K-means and GMM, HDBSCAN does not require a predefined number of clusters and can identify low-density or unstable samples as noise. Compared with standard DBSCAN, HDBSCAN is more adaptive to clusters with different densities. This is important for radar pulse deinterleaving because different emitters may have substantially different pulse counts, parameter distributions, and local densities.

**Reproduction and extension suggestions:**

* When using raw PDW features, perform feature standardization first to avoid distance metrics being dominated by differences in scale among TOA, RF, PW, PA, and other features.
* Include DBSCAN, K-means, GMM, and hierarchical clustering in comparative experiments to distinguish methods that handle unknown cluster numbers from methods that require a given K.
* If using a deep model to learn embeddings, keep HDBSCAN as a unified clustering backend so that performance gains can be attributed to representation learning rather than the clustering algorithm itself.
* When reporting HDBSCAN results, record key hyperparameters and the noise-point ratio, and use evaluation metrics consistent with the Turing Challenge.

#### 4.2.3 Transformer Metric Learning

“Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning” defines radar pulse deinterleaving as the problem of separating interleaved pulse sequences according to emitters, emphasizing that the number of emitters in a pulse sequence is unknown. The method uses a Transformer to encode PDW sequences and applies triplet loss to learn pulse embeddings, making pulses from the same emitter closer in the feature space and pulses from different emitters farther apart. This direction is suitable as a representative method for supervised metric learning + clustering.

#### 4.2.4 Optimal Transport Clustering

“Deinterleaving RADAR Emitters with Optimal Transport Distances” proposes an unsupervised deinterleaving method. The basic idea is to first over-segment pulses to reduce the risk of incorrectly merging different emitters; then, considering that a complex emitter may correspond to multiple clusters, it uses optimal transport distances for hierarchical cluster merging. This method supports the direction of “unsupervised clustering + cluster merging.”

#### 4.2.5 Renewal Process Mixture Model Methods

“Deinterleaving of Mixtures of Renewal Processes” and “Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures” both address deinterleaving from the perspective of stochastic process mixture modeling. These methods do not simply cluster in the PDW space; instead, they model interleaved pulse sequences using pulse arrival times, symbolic sequences, renewal processes, or Markov-chain structures. They are suitable references for statistical modeling methods.

#### 4.2.6 Deep Contrastive Clustering

“Deep Contrastive Clustering for Signal Deinterleaving” is a representative paper in the contrastive learning + clustering direction. The core idea is to enhance the separability of signal or pulse representations through contrastive learning, and then perform deinterleaving using clustering or pseudo-label mechanisms. This method is better placed under deep representation learning methods for deinterleaving rather than under open-source frameworks or benchmark resources.

#### 4.2.7 Semantic-Segmentation-Based Deinterleaving

“A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network” converts deinterleaving into a semantic segmentation / sequence labeling problem. Its input mainly revolves around DTOA and related time-difference sequences. A neural network assigns semantic labels to pulses in a pulse stream to complete deinterleaving. This type of method is suitable for complex PRI modulation environments, but it usually relies on labeled training data and therefore is not strictly unsupervised clustering-based deinterleaving.

#### 4.2.8 Image-Segmentation-Based Deinterleaving

“Image Segmentation for Radar Signal Deinterleaving Using Deep Learning” transforms radar pulse sequences into image-like representations and then uses deep image segmentation models for signal deinterleaving. Its value lies in transforming pulse deinterleaving from a traditional PRI search problem into a two-dimensional image segmentation problem. However, its task setting and input representation differ from direct PDW clustering.

#### 4.2.9 Sep-RefineNet

Sep-RefineNet is a semantic-segmentation-based deinterleaving method. It constructs a frequency feature matrix to represent the semantic structure of pulse streams from different radar signals, uses Sep-RefineNet to perform pixel-level segmentation on the matrix, and then recovers pulse positions in the original pulse stream through position decoding and verification. This method belongs under segmentation-style deinterleaving and should not be categorized as an unsupervised clustering baseline.

#### 4.2.10 Deep ToA Mask Recursive Deinterleaving

“Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving” uses ToA masks and recursive separation to process interleaved radar pulse sequences. It emphasizes gradually extracting pulse subsequences corresponding to different emitters from arrival-time structures. It is suitable for the category of deep recursive deinterleaving / mask-based deinterleaving.

#### 4.2.11 GCN Semi-Supervised Radar Signal Sorting

“Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning” uses graph convolutional networks and semi-supervised learning to model radar signal sorting. This direction is suitable for describing the use of pulse relationship graphs, pseudo-labels, and semi-supervised learning in signal sorting. However, it is not a strictly unsupervised PDW emitter clustering method.

---

### 4.3 Detailed Notes on Papers and Methods

This section expands only on the papers and projects already listed in the table above. It does not add additional papers. For project resources without independent papers, such as the HDBSCAN baseline in the Turing Challenge, they are treated as “project baselines” rather than “paper methods.”

#### 4.3.1 The Turing Synthetic Radar Dataset: A Dataset for Pulse Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Dataset / benchmark / challenge  
**Publication type:** arXiv preprint, 2026  
**Method category:** Standard PDW pulse deinterleaving dataset and evaluation benchmark  
**Strictly unsupervised?** The dataset itself does not restrict methods; it can be used for unsupervised, semi-supervised, and supervised experiments  
**Code / data:** Public project and Hugging Face dataset available  

</td>
<td width="72%" valign="top">

**Paper:** The Turing Synthetic Radar Dataset: A Dataset for Pulse Deinterleaving
**Authors:** Edward Gunn, Adam Hosford, Robert Jones, Leo Zeitler, Ian Groves, Victoria Nockles
**Links:** [arXiv](https://arxiv.org/abs/2602.03856) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

**Highlights**

* Built for radar pulse deinterleaving and one of the few public large-scale PDW deinterleaving datasets;
* includes stare and scan receiving configurations, enabling performance analysis under different receiving modes;
* provides emitter labels and a paired challenge evaluation pipeline, making unified comparison with V-measure, ARI, AMI, and other clustering metrics convenient;
* paired with the Turing Deinterleaving Challenge, which provides official baselines and example workflows;
* suitable as a primary benchmark for new methods rather than merely a supplementary data source.

</td>
</tr>
</table>

**Summary:**
This paper introduces the Turing Synthetic Radar Dataset / TSRD, a synthetic PDW dataset for radar pulse deinterleaving research. The dataset targets the problem of “how to reassign interleaved pulses from multiple unknown emitters back to their corresponding emitters” in electronic warfare and signal intelligence scenarios. It provides large-scale pulse sequences, emitter labels, and different receiving configurations. The paper is paired with the Turing Deinterleaving Challenge, encouraging researchers to compare deinterleaving methods under unified data, metrics, and evaluation procedures. For radar pulse deinterleaving research, the main value of this resource lies not in proposing a single algorithm, but in providing a relatively standardized and reproducible public experimental platform.

**Recommended placement:** Datasets and benchmarks / primary recommended resource.
**Limitations:** The data is synthetic and cannot fully replace real measured PDW data from electronic support systems. If labels are used to train a model, the difference from strictly unsupervised deinterleaving must be clearly stated.

---

#### 4.3.2 Turing Deinterleaving Challenge HDBSCAN Baseline

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Official project baseline  
**Publication type:** No independent paper; part of the Turing Deinterleaving Challenge  
**Method category:** Unsupervised density clustering baseline  
**Strictly unsupervised?** Yes  
**Code / data:** Official GitHub examples available

</td>
<td width="72%" valign="top">

**Project:** Turing Deinterleaving Challenge baseline
**Link:** [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

**Highlights**

* Uses HDBSCAN clustering directly on raw PDW features;
* does not require a preset number of emitters, making it suitable for unknown-emitter-number settings;
* can serve as the most direct and transparent unsupervised baseline on TSRD;
* suitable for comparison with DBSCAN, K-means, GMM, hierarchical clustering, and representation learning + clustering methods;
* useful for verifying data loading, standardization, clustering, and evaluation pipelines.

</td>
</tr>
</table>

**Summary:**
The HDBSCAN baseline is the official unsupervised reference method provided by the Turing Deinterleaving Challenge. It does not rely on deep learning training or emitter class labels. Instead, it performs density clustering directly in the PDW feature space. Since HDBSCAN can adapt to clusters with different densities to some extent and can treat some samples as noise, it is closer to the unknown-emitter-number setting than K-means or GMM, which require a preset number of clusters. For subsequent method research, the HDBSCAN baseline can serve as the most basic reproducible reference for judging whether improvements truly come from new feature construction, representation learning, or clustering strategies.

**Recommended placement:** Recommended baselines / official unsupervised baseline.
**Limitations:** Direct clustering on raw PDW features can be affected by feature scale, parameter overlap, and complex operating modes. Without proper standardization, distance metrics may be dominated by large-scale features such as TOA or RF.

---

#### 4.3.3 Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Publication type:** arXiv preprint, 2025  
**Method category:** Transformer metric learning / supervised representation learning + clustering  
**Strictly unsupervised?** No; labels are used during training to construct triplet loss  
**Code / data:** No official open-source code found  

</td>
<td width="72%" valign="top">

**Paper:** Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning
**Authors:** Edward Gunn, Adam Hosford, Daniel Mannion, Jarrod Williams, Varun Chhabra, Victoria Nockles
**Link:** [arXiv](https://arxiv.org/abs/2503.13476)

**Highlights**

* Models radar pulse deinterleaving as pulse assignment under an unknown number of emitters;
* uses a Transformer to encode PDW sequences and learn contextual pulse representations;
* uses triplet loss to make same-emitter pulses closer and different-emitter pulses farther apart;
* at inference time, clustering or similarity grouping can be performed in the learned embedding space;
* suitable as a representative method for “supervised metric learning + deinterleaving-style output.”

</td>
</tr>
</table>

**Summary:**
This paper proposes a Transformer-based deep metric learning method for radar pulse sequences containing multiple unknown emitters. The method uses a Transformer to model PDW sequences and aims to obtain more discriminative embeddings by leveraging contextual relationships among pulses. During training, triplet loss is used so that pulses from the same emitter become closer in the feature space, while pulses from different emitters become farther apart. The focus is not on designing traditional PRI search rules, but on learning a pulse embedding space suitable for subsequent clustering or similarity-based grouping.

**Recommended placement:** Supervised representation learning / metric learning methods.
**Limitations:** The training stage relies on labels, so it should not be directly treated as equivalent to strictly unsupervised methods such as HDBSCAN or DBSCAN. It is more suitable as a supervised learning reference or representation-learning upper bound.

---

#### 4.3.4 Deinterleaving RADAR Emitters with Optimal Transport Distances

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Aerospace and Electronic Systems, 2024  
**Method category:** Unsupervised clustering / optimal transport distance / hierarchical cluster merging  
**Strictly unsupervised?** Basically yes  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Deinterleaving RADAR Emitters With Optimal Transport Distances
**Authors:** Manon Mottier, Gilles Chardon, Frédéric Pascal
**Links:** [arXiv](https://arxiv.org/abs/2312.11178) / [DOI](https://doi.org/10.1109/TAES.2024.3367287)

**Highlights**

* Targets radar signal deinterleaving with an unknown number of emitters;
* first performs over-segmentation through clustering to avoid premature merging of different emitters;
* then uses optimal transport distances to measure distributional differences between clusters;
* merges clusters that may belong to the same emitter through hierarchical clustering;
* suitable for complex emitter behavior and multimodal parameter distributions.

</td>
</tr>
</table>

**Summary:**
This paper proposes an unsupervised radar emitter deinterleaving method based on clustering and optimal transport distances. The method first performs initial clustering on received pulses and tries to avoid assigning pulses from different emitters to the same initial cluster. Since a complex emitter may be split into multiple clusters due to operating-mode changes or parameter distribution differences, the paper further uses optimal transport distances to measure overall distributional similarity between clusters and completes cluster merging through hierarchical clustering. The method is suitable for an unsupervised deinterleaving framework based on “first ensuring purity, then merging fragmented clusters.”

**Recommended placement:** Unsupervised clustering methods / cluster merging methods.
**Limitations:** The quality of initial over-segmentation affects subsequent merging; optimal transport distance computation and hierarchical merging introduce additional computational cost.

---

#### 4.3.5 Deinterleaving of Mixtures of Renewal Processes

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Signal Processing, 2019  
**Method category:** Statistical modeling / renewal process mixture model / arrival-time modeling  
**Strictly unsupervised?** Basically yes; a model-driven source separation method  
**Code / data:** No official open-source code found  

</td>
<td width="72%" valign="top">

**Paper:** Deinterleaving of Mixtures of Renewal Processes
**Authors:** Jeremy Young, Anders Høst-Madsen, Eva-Marie Nosal
**Link:** [DOI](https://doi.org/10.1109/TSP.2018.2886149)

**Highlights**

* Models deinterleaving from multi-source pulse arrival time sequences as separation of renewal process mixtures;
* mainly relies on impulse spacing / pulse interval information rather than multi-dimensional PDW feature clustering;
* emphasizes how to perform source separation using only timing information;
* has strong theoretical interpretability and is suitable as a reference for statistical modeling approaches;
* useful for understanding the probabilistic modeling basis of traditional PRI / TOA deinterleaving problems.

</td>
</tr>
</table>

**Summary:**
This paper studies the deinterleaving of multi-source pulse sequences by modeling pulse intervals from different sources as different renewal processes. Unlike methods that directly cluster multi-dimensional PDW features such as RF, PW, and PA, this method focuses on source separation based only on pulse arrival intervals. Starting from impulsive source separation in multi-input single-output systems, the paper builds a renewal process mixture model and uses pulse-interval statistics to partition events from different sources. For radar deinterleaving research, the value of this paper lies in providing a more theoretical and statistical modeling perspective.

**Recommended placement:** Statistical modeling methods / TOA-PRI theoretical methods.
**Limitations:** The method mainly relies on interval modeling. Its adaptability to complex parameter agility, multi-parameter coupling, or highly overlapping PDW distributions needs to be analyzed according to the specific scenario.

---

#### 4.3.6 Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Signal Processing, 2024  
**Method category:** Discrete renewal process / Markov-chain mixture model / penalized likelihood optimization  
**Strictly unsupervised?** Basically yes  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Deinterleaving of Discrete Renewal Process Mixtures with Application to Electronic Support Measures
**Authors:** Jean Pinsolle, Olivier Goudet, Cyrille Enderli, Sylvain Lamprier, Jin-Kao Hao
**Links:** [arXiv](https://arxiv.org/abs/2402.09166) / [DOI](https://doi.org/10.1109/TSP.2024.3464753)

**Highlights**

* Models an interleaved sequence as a mixture of multiple discrete renewal Markov chains;
* uses a penalized likelihood score for deinterleaving optimization;
* jointly exploits symbolic sequence information and arrival-time information;
* provides theoretical analysis showing recovery of the true partition under large-sample conditions;
* validates the method in RESM / electronic support measurement scenarios.

</td>
</tr>
</table>

**Summary:**
This paper proposes a method for deinterleaving mixtures of discrete renewal processes and applies it to radar electronic support measurement scenarios. The method treats interleaved observations as the superposition of multiple discrete renewal Markov chains and recovers the corresponding partitions by maximizing a penalized likelihood score. Compared with models that only use arrival-time intervals, this method jointly considers symbolic sequence information and arrival-time information. It also provides theoretical analysis showing that the true partition can be recovered under certain conditions. The method is suitable as a representative work for “stochastic process modeling + theoretical analysis + electronic support applications.”

**Recommended placement:** Statistical modeling methods / discrete sequence deinterleaving methods.
**Limitations:** The model assumptions and likelihood form need to be validated against real complex radar scenarios. Compared with ordinary clustering methods, the method has a higher barrier to understanding and implementation.

---

#### 4.3.7 Deep Contrastive Clustering for Signal Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Aerospace and Electronic Systems, 2024  
**Method category:** Self-supervised contrastive learning / deep clustering / representation learning  
**Strictly unsupervised?** Closer to a self-supervised / no-prior-information setting, depending on the pseudo-label construction  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Deep Contrastive Clustering for Signal Deinterleaving
**Authors:** Shuyuan Yang, Xinyi Zhao, Huiling Liu, Chen Yang, Tongqing Peng, Rundong Li, Feng Zhang
**Link:** [DOI](https://doi.org/10.1109/TAES.2023.3322971)

**Highlights**

* Proposes DCCA（Deep Contrastive Clustering Algorithm）for radar signal deinterleaving;
* uses a self-supervised contrastive learning framework to learn more discriminative signal representations;
* constructs supervisory information through customized pseudo-labels for augmented signals;
* aims to complete signal deinterleaving without prior information about emitters;
* suitable as a representative paper for contrastive learning + clustering.

</td>
</tr>
</table>

**Summary:**
This paper proposes a Deep Contrastive Clustering Algorithm, DCCA, to address radar signal deinterleaving in complex electromagnetic environments. The method first builds a contrastive self-supervised deep attention network and learns signal representations by generating customized pseudo-labels for augmented signals, making similar signals more compact and different signals more separated in the feature space. It then combines the learned representations with clustering mechanisms to complete signal deinterleaving. The main value of this method is that it applies contrastive learning to radar signal deinterleaving, enabling the model to learn separable representations without explicit prior information about emitters.

**Recommended placement:** Deep representation learning / contrastive learning deinterleaving methods.
**Limitations:** Whether it is strictly unsupervised depends on the design of pseudo-labels and data augmentation. If pseudo-label construction relies on strong assumptions, this should be explained in surveys or experiments.

---

#### 4.3.8 A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Signal Processing, 2022  
**Method category:** Sequence modeling / semantic-segmentation-based deinterleaving / supervised learning  
**Strictly unsupervised?** No  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** A Radar Signal Deinterleaving Method Based on Semantic Segmentation with Neural Network
**Authors:** Wang Chao, Sun Liting, Liu Zhangmeng, Huang Zhitao
**Links:** [arXiv](https://arxiv.org/abs/2110.13706) / [DOI](https://doi.org/10.1109/TSP.2022.3229630)

**Highlights**

* Proposes SSD（semantic segmentation deinterleaving）;
* feeds DTOA sequences into a neural network for sequence modeling;
* assigns semantic labels to pulses to complete deinterleaving;
* avoids repeated PRI search and merging in traditional methods;
* has certain robustness to complex PRI modulation, missing pulses, and noise pulses.

</td>
</tr>
</table>

**Summary:**
This paper converts radar signal deinterleaving into a semantic segmentation / sequence labeling problem and proposes a semantic segmentation deinterleaving method, SSD. The method mainly uses DTOA sequences from pulse streams as input. A sequence modeling neural network learns semantic patterns of different radar signals in the sequence and assigns semantic labels to each pulse, thereby completing deinterleaving. Compared with traditional PRI search methods, it does not require explicit PRI or PRI-period search, nor repeated search and merging of multi-pulse-period signals. The paper also notes that recurrent neural networks are more advantageous than convolutional neural networks for this sequence modeling task.

**Recommended placement:** Supervised sequence labeling / semantic-segmentation-based deinterleaving.
**Limitations:** The training stage relies on labels and is not strictly unsupervised. The output categories are tied to the training label space, and additional design is needed for generalization to unknown emitters.

---

#### 4.3.9 Image Segmentation for Radar Signal Deinterleaving Using Deep Learning

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper
**Published in:** IEEE Transactions on Aerospace and Electronic Systems, 2023
**Method category:** Image-like representation / deep image segmentation / supervised learning
**Strictly unsupervised?** No
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Image Segmentation for Radar Signal Deinterleaving Using Deep Learning
**Authors:** Mustafa Atahan Nuhoglu, Yasar Kemal Alp, Mehmet Ege Can Ulusoy, Hakan Ali Çirpan
**Link:** [DOI](https://doi.org/10.1109/TAES.2022.3188225)

**Highlights**

* Converts radar signal deinterleaving into an image segmentation problem;
* transforms pulse sequences into two-dimensional image or matrix representations through preprocessing;
* uses deep segmentation networks to extract PRI / TOA structures in the image;
* suitable for borrowing image segmentation architectures such as U-Net;
* provides a way to transfer traditional sequence deinterleaving into visual segmentation modeling.

</td>
</tr>
</table>

**Summary:**
This paper attempts to convert radar signal deinterleaving into an image segmentation task. The basic idea is to first construct a two-dimensional image representation from pulse arrival times, PRI, or related transformations so that temporal patterns of different radar signals appear as separable structures in the image space. Deep image segmentation models are then used to segment these structures at the pixel or region level, thereby separating pulses from different emitters. The significance of this method is that it extends radar deinterleaving from a traditional one-dimensional PRI search problem to a two-dimensional image segmentation problem, allowing the use of mature segmentation networks and training strategies from computer vision.

**Recommended placement:** Image-segmentation-based deinterleaving / supervised deep learning methods.
**Limitations:** Requires designing a transformation from pulse sequences to images; training usually relies on labels; segmentation results need to be mapped back to the original pulse sequence.

---

#### 4.3.10 Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** Applied Sciences, 2023  
**Method category:** Frequency characteristic matrix / semantic segmentation network / position decoding  
**Strictly unsupervised?** No  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Sep-RefineNet: A Deinterleaving Method for Radar Signals Based on Semantic Segmentation
**Authors:** Yongjiang Mao, Wenjuan Ren, Xipeng Li, Zhanpeng Yang, Wei Cao
**Links:** [Paper](https://www.mdpi.com/2076-3417/13/4/2726) / [DOI](https://doi.org/10.3390/app13042726)

**Highlights**

* Constructs an FCM（frequency characteristic matrix）to encode PRI semantic features of radar pulse streams;
* uses Sep-RefineNet for pixel-level semantic segmentation of the FCM;
* maps segmentation results back to the original pulse stream through position decoding and verification;
* reduces dependence on manual thresholding and pulse sequence search in traditional algorithms;
* has certain noise robustness under interleaved pulses and missing-pulse conditions.

</td>
</tr>
</table>

**Summary:**
This paper proposes Sep-RefineNet, a radar signal deinterleaving method based on semantic segmentation. The method first constructs a frequency characteristic matrix, FCM, to encode the PRI semantic structure of different radar pulse streams. It then uses the Sep-RefineNet semantic segmentation network to perform pixel-level segmentation on the matrix. Finally, it obtains pulse positions in the original pulse stream through position decoding and verification. The method aims to reduce the dependence of traditional deinterleaving algorithms on manual thresholds, PRI search, and empirical rules, while improving robustness under interleaved pulses and missing-pulse conditions.

**Recommended placement:** Semantic-segmentation-based deinterleaving / frequency matrix encoding methods.
**Limitations:** It is a supervised segmentation method and depends on a specific FCM construction pipeline. It is not suitable as a strictly unsupervised PDW clustering baseline.

---

#### 4.3.11 Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper  
**Published in:** IEEE Transactions on Aerospace and Electronic Systems, 2023  
**Method category:** ToA mask prediction / recursive separation / deep learning  
**Strictly unsupervised?** Usually no; training relies on simulation or labeled supervision  
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Deep ToA Mask-Based Recursive Radar Pulse Deinterleaving
**Authors:** Haoran Xiang, Furao Shen, Jian Zhao
**Link:** [DOI](https://doi.org/10.1109/TAES.2022.3193948)

**Highlights**

* Proposes Deep ToA Mask / Recursive Deinterleaving Network;
* maps ToA pulse sequences into a feature space suitable for separation;
* estimates ToA coefficient masks corresponding to each radar emitter;
* uses recursion to gradually separate emitters from an interleaved pulse sequence;
* suitable for discussing deep recursive deinterleaving under complex interleaving, PRI jitter, and missing-pulse conditions.

</td>
</tr>
</table>

**Summary:**
This paper proposes a recursive radar pulse deinterleaving method based on Deep ToA Mask for densely interleaved multi-mode radar signals in complex electromagnetic environments. The method maps ToA pulse sequences into a feature space through a recursive deinterleaving network and estimates ToA coefficient masks corresponding to different radar emitters by combining local and global contextual information. Unlike methods that output all cluster labels at once, this method recursively separates pulse subsequences, thereby reducing the difficulty of deinterleaving in strongly interleaved scenarios. The paper emphasizes that the method can handle parameter overlap, missing pulses, and PRI jitter.

**Recommended placement:** Deep recursive deinterleaving / ToA mask methods.
**Limitations:** Recursive separation is vulnerable to errors propagated from earlier steps. For multi-mode emitters, additional modulation type or PRI information may be required.

---

#### 4.3.12 Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning

<table>
<tr>
<td width="28%" valign="top">

**Resource type:** Representative paper
**Published in:** IEEE Signal Processing Letters, 2025
**Method category:** Graph convolutional network / semi-supervised learning / relational modeling
**Strictly unsupervised?** No
**Code / data:** No official open-source code found

</td>
<td width="72%" valign="top">

**Paper:** Radar Signal Sorting via Graph Convolutional Network and Semi-Supervised Learning
**Authors:** Ziying Li, Xiongjun Fu, Jian Dong, Min Xie
**Link:** [DOI](https://doi.org/10.1109/LSP.2024.3519884)

**Highlights**

* Introduces a semi-supervised learning framework for the problem that large numbers of labeled pulses are difficult to obtain;
* uses graph convolutional networks to model relationships among pulse samples;
* treats pulses as graph nodes and models similarity or adjacency as edges;
* uses a small number of labeled samples and a large number of unlabeled samples to improve deinterleaving performance;
* suitable as a representative semi-supervised relational modeling method for radar signal sorting.

</td>
</tr>
</table>

**Summary:**
This paper addresses the difficulty of obtaining large amounts of labeled pulse data in practical radar reconnaissance systems and proposes a radar signal sorting method based on graph convolutional networks and semi-supervised learning. The method models pulse samples as graph nodes and propagates information through node features and edge relationships, allowing supervision from a small number of labeled samples to extend to unlabeled samples. Compared with fully supervised deep learning methods, it emphasizes the use of structural relationships among samples under label scarcity. Compared with traditional unsupervised clustering, it still relies on partial labels and therefore belongs to semi-supervised deinterleaving.

**Recommended placement:** Semi-supervised learning / graph neural network deinterleaving methods.
**Limitations:** It is not strictly unsupervised. Graph construction, edge-weight design, and the quality of the few labeled samples significantly affect final performance.

---

## 5. Recommended Experimental Setup

If the research goal is **unsupervised radar pulse deinterleaving**, TSRD and the Turing Deinterleaving Challenge are recommended as the primary experimental platform, with reproducible comparisons built around the HDBSCAN baseline.

### 5.1 Primary Benchmark

It is recommended to use **Turing Synthetic Radar Dataset / TSRD** as the primary benchmark. The dataset provides PDW pulse sequences, emitter labels, and a corresponding evaluation pipeline, making it suitable for comparing traditional clustering, representation learning, and supervised upper-bound models.

The experimental setup should clearly specify the following:

* the receiving mode used, such as stare or scan;
* the input feature set, such as TOA, CF/RF, PW, PA, and AoA;
* whether emitter labels are used during training;
* whether the number of emitters K is preset;
* the clustering method, distance metric, and main hyperparameters;
* whether features such as TOA, RF, PW, and PA are standardized or transformed.

### 5.2 Recommended Baselines

| Category                | Method                                                          | Usage                                                                                    |
| ----------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Official baseline       | HDBSCAN                                                         | Reproduce the basic result of the Turing Challenge                                       |
| Raw-feature clustering  | DBSCAN, K-means, GMM, hierarchical clustering                   | Evaluate traditional clustering methods on raw PDW features                              |
| PRI-based methods       | PRI histogram, CDIF, SDIF, PRI transform                        | Compare against traditional temporal-structure deinterleaving methods                    |
| Hybrid methods          | RF/PW/DOA coarse clustering + PRI fine deinterleaving           | Validate combinations of multi-feature coarse grouping and PRI-based fine deinterleaving |
| Representation learning | Autoencoder / contrastive encoder + HDBSCAN                     | Evaluate whether learned embeddings improve clustering separability                      |
| Supervised upper bounds | Transformer metric learning, GCN, TCN, sequence labeling models | Serve as performance references for supervised or semi-supervised methods                |

### 5.3 Recommended Pipeline

```text
Raw PDW sequence
      ↓
Feature selection and standardization
      ↓
Optional temporal feature construction
      ↓
Clustering or representation learning
      ↓
Cluster label assignment
      ↓
Evaluation with V-measure / ARI / AMI / Pairwise F1
```

For unsupervised methods, it is recommended to first reproduce `raw PDW features + HDBSCAN`, and then gradually add new feature construction, clustering strategies, or representation learning modules. This avoids the difficulty of directly comparing complex models without knowing where performance improvements come from.

---

## 6. Notes on Non-Core Resources

To maintain the quality of this resource list, the following resources are not included as core recommendations. Some may be related to radar signal processing, but due to task mismatch, insufficient documentation, limited reproducibility value, or project quality issues, they are not placed in the main tables.

| Resource                                                                                                        | Treatment                                | Reason                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Stream-ConAEnet](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)               | Not included in the main recommendations | The project description states that it is an undergraduate thesis project; data description, running workflow, evaluation protocol, and paper support are insufficient                                        |
| [EW Signal Intelligence Deinterleaving Demo](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence) | Not included in the main recommendations | Educational demo suitable for understanding concepts, but not suitable as a research-grade benchmark or representative method                                                                                 |
| [radar_data_Kmeans](https://github.com/zda2019/radar_data_Kmeans)                                               | Not included as a core resource          | Related to radar deinterleaving, but documentation, evaluation protocol, and data description are weak; may serve as an engineering implementation clue                                                       |
| [HMC-RFN GitHub repository](https://github.com/xm980426/HMC-RFN)                                                | Not included as a core resource          | MATLAB code is available, but the README and reproduction instructions are limited; may serve as a paper-code clue                                                                                            |
| [2nd-EBDSC repository](https://github.com/framist/2nd-EBDSC)                                                    | Not included as a core resource          | Engineering content is relatively complete, but the task is closer to competition-style supervised sequence modeling or template-assisted signal extraction, not standard unsupervised PDW emitter clustering |

---

## 7. Notes

* Compared with general RF modulation recognition or RF fingerprinting, high-quality open-source resources for radar pulse deinterleaving remain scarce.
* Most public datasets are synthetic, because real radar PDW or IQ data is difficult to release and reliable ground-truth labels are hard to obtain.
* When reporting experimental results, clearly state whether the method is unsupervised, supervised, semi-supervised, or “supervised representation learning followed by clustering.”
* If the research goal is strictly unsupervised radar pulse deinterleaving, TSRD + HDBSCAN / DBSCAN / K-means / hierarchical clustering is currently the most transparent and easiest-to-reproduce starting point.
* The star ratings in this repository are subjective assessments by the maintainers based on task relevance, resource quality, reproducibility value, and completeness of data descriptions. They do not represent GitHub stars.

---

## 8. Citation and Contribution

This resource list is curated and maintained by **SmartDSP Lab, School of Informatics, Xiamen University**. It aims to provide a systematic, reproducible, and extensible open-source resource index for radar pulse deinterleaving, radar signal sorting, PDW data processing, and related intelligent electromagnetic signal processing research.

If this repository is helpful to your research or project, you are welcome to cite this resource list in papers, reports, or projects, and please also cite the corresponding original papers, datasets, and code repositories whenever possible.

### Citation Format

If you use the resource list curated in this repository, you may cite it as follows:

```bibtex
@misc{smartdsp_radar_deinterleaving_resources,
  title        = {Radar Pulse Deinterleaving Resources},
  author       = {SmartDSP Lab, School of Informatics, Xiamen University},
  year         = {2026},
  howpublished = {\url{https://github.com/your-repository-url}},
  note         = {A curated resource list for radar pulse deinterleaving and radar signal sorting}
}
```

Please replace `https://github.com/your-repository-url` with the actual GitHub URL of this repository.

### How to Contribute

Researchers and developers are welcome to contribute corrections and additions through issues or pull requests. Contributions may include, but are not limited to:

* newly released radar pulse deinterleaving datasets;
* open-source implementations of PRI-based, clustering-based, or deep-learning-based deinterleaving methods;
* reproducible benchmark results;
* public paper, code, and dataset links;
* corrections to data availability, task definitions, or recommendation levels of existing resources;
* additional notes on whether a method strictly satisfies the definition of unsupervised radar pulse deinterleaving.

When contributing a new resource, please provide the following information whenever possible:

```text
Resource name:
Resource link:
Task type:
Method type:
Open-source code:
Open data:
Labels included:
Strictly unsupervised:
Reason for recommendation:
Notes:
```

This repository will continue to update public resources related to radar pulse deinterleaving. Researchers in related fields are also welcome to help improve it.
