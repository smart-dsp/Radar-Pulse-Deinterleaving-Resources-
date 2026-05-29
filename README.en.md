# Radar Pulse Deinterleaving Resources

<p align="center">
  <a href="./README.md">简体中文</a> |
  <b>English</b>
</p>

This repository continuously collects open-source resources related to **Radar Pulse Deinterleaving / Radar Signal Sorting**, with a focus on task definitions, PDW datasets, open-source implementations, representative methods, evaluation metrics, and reproducible experiments.

Radar pulse deinterleaving aims to separate an interleaved pulse stream into groups corresponding to different radar emitters. In a typical **PDW (Pulse Description Word)** setting, each pulse is usually described by parameters such as **TOA (Time of Arrival)**, **PRI (Pulse Repetition Interval)**, **RF/CF (Radio Frequency / Carrier Frequency)**, **PW (Pulse Width)**, **PA (Pulse Amplitude)**, and **DOA/AOA (Direction / Angle of Arrival)**.

Open-source resources for radar pulse deinterleaving are still relatively scattered. Different projects may use terms such as `deinterleaving`, `sorting`, `clustering`, and `sequence labeling`, but their task settings are not always the same. Therefore, this repository attempts to organize related resources manually and annotate their task type, data availability, supervision setting, and whether they match the strict definition of unsupervised radar pulse deinterleaving.

---

## Highlights

* **Recommended benchmark:** [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) and its [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset).
* **Strictly unsupervised deinterleaving methods:** K-means, GMM, DBSCAN, HDBSCAN, and hierarchical clustering best match the definition of grouping interleaved pulses into emitter-specific clusters.
* **Deep learning methods should be carefully distinguished:** Many neural deinterleaving methods solve the deinterleaving task, but they may use labels during training and therefore should not be treated as strictly unsupervised clustering methods.
* **Dataset status:** Publicly available real radar deinterleaving datasets are very limited. Most open research currently relies on synthetic PDW data.
* **Repository scope:** This project is not a simple link collection. It aims to distinguish resources by task type, supervision level, data availability, and whether they fit the strict definition of radar pulse deinterleaving.

> Note: The ⭐ recommendation level in this repository reflects the maintainer’s subjective assessment based on task relevance, openness, reproducibility, and data documentation. It does not represent GitHub stars.

---

## Table of Contents

* [1. Overview](#1-overview)

  * [1.1 What is radar pulse deinterleaving?](#11-what-is-radar-pulse-deinterleaving)
  * [1.2 Common input features](#12-common-input-features)
  * [1.3 Common evaluation metrics](#13-common-evaluation-metrics)
* [2. Method Taxonomy and Task Fit](#2-method-taxonomy-and-task-fit)

  * [2.1 Traditional PRI-based methods](#21-traditional-pri-based-methods)
  * [2.2 Unsupervised clustering methods](#22-unsupervised-clustering-methods)
  * [2.3 Representation learning + clustering methods](#23-representation-learning--clustering-methods)
  * [2.4 Supervised sequence labeling and segmentation methods](#24-supervised-sequence-labeling-and-segmentation-methods)
  * [2.5 Which methods match the strict definition?](#25-which-methods-match-the-strict-definition)
* [3. Datasets](#3-datasets)

  * [3.1 Dataset recommendation levels](#31-dataset-recommendation-levels)
  * [3.2 Dataset summary](#32-dataset-summary)
  * [3.3 Key dataset notes](#33-key-dataset-notes)
* [4. Methods and Implementations](#4-methods-and-implementations)

  * [4.1 Method recommendation levels](#41-method-recommendation-levels)
  * [4.2 Method and code summary](#42-method-and-code-summary)
  * [4.3 Key method notes](#43-key-method-notes)
* [5. Recommended Experimental Setup](#5-recommended-experimental-setup)

  * [5.1 Primary benchmark](#51-primary-benchmark)
  * [5.2 Baseline methods](#52-baseline-methods)
  * [5.3 Suggested pipeline](#53-suggested-pipeline)
* [6. Recommended Reading and Starting Points](#6-recommended-reading-and-starting-points)
* [7. Notes](#7-notes)
* [8. Citation and Contribution](#8-citation-and-contribution)

---

## 1. Overview

### 1.1 What is radar pulse deinterleaving?

Radar pulse deinterleaving, also known as **radar signal sorting**, is the task of assigning pulses from a mixed electromagnetic environment to their corresponding radar emitters.

<p align="center">
  <img src="./fig1.png" width="90%">
</p>

<p align="center">
  <b>Figure 1.</b> Illustration of radar pulse deinterleaving: pulse sequences from multiple radar emitters are received as an interleaved pulse stream by a radar warning receiver, and the deinterleaving algorithm aims to separate them into emitter-specific pulse clusters.
</p>

Given an interleaved pulse sequence:

```text
P = {p1, p2, ..., pN}
```

where each pulse `pi` is usually represented as a PDW feature vector:

```text
pi = [TOA, RF/CF, PW, PA, DOA/AOA, ...]
```

the goal is to output a partition:

```text
C = {C1, C2, ..., CK}
```

where each cluster `Ck` corresponds to one radar emitter.

In the strictest sense, radar pulse deinterleaving can be regarded as an **unsupervised clustering problem** because:

* the number of emitters is usually unknown;
* emitter identities are not fixed classes;
* pulses from the same emitter should be grouped together;
* pulses from different emitters should be separated.

However, many modern methods formulate this task as supervised learning, semi-supervised learning, metric learning, sequence labeling, semantic segmentation, or instance segmentation. These methods may still solve the deinterleaving task, but they are not always purely unsupervised clustering methods.

---

### 1.2 Common input features

| Feature    | Meaning                                       | Usage                                                                             |
| ---------- | --------------------------------------------- | --------------------------------------------------------------------------------- |
| TOA        | Time of Arrival                               | Used to derive PRI and analyze temporal patterns                                  |
| PRI / DTOA | Pulse Repetition Interval / Difference of TOA | Core feature for traditional deinterleaving methods                               |
| RF / CF    | Radio Frequency / Carrier Frequency           | Useful for distinguishing fixed-frequency or frequency-agile radars               |
| PW         | Pulse Width                                   | Helps distinguish emitters with different waveform settings                       |
| PA / AMP   | Pulse Amplitude                               | Useful as an auxiliary feature, but sensitive to propagation and receiver effects |
| DOA / AOA  | Direction / Angle of Arrival                  | A strong spatial feature when available                                           |

---

### 1.3 Common evaluation metrics

| Metric       | Description                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| V-measure    | Harmonic mean of homogeneity and completeness                                                           |
| Homogeneity  | Whether each predicted cluster mainly contains pulses from only one true emitter                        |
| Completeness | Whether pulses from the same true emitter are assigned to the same predicted cluster                    |
| ARI          | Adjusted Rand Index                                                                                     |
| AMI          | Adjusted Mutual Information                                                                             |
| Pairwise F1  | F1 score when treating whether two pulses belong to the same emitter as a binary classification problem |
| MCC          | Matthews Correlation Coefficient for pairwise matching                                                  |

---

## 2. Method Taxonomy and Task Fit

Radar pulse deinterleaving methods can be roughly divided into **traditional PRI-based methods**, **unsupervised clustering methods**, **representation learning + clustering methods**, and **supervised sequence labeling or segmentation methods**.

It should be noted that different papers and code repositories may use terms such as `deinterleaving`, `sorting`, and `pulse clustering`, but their task settings can differ significantly. Some methods are strictly unsupervised clustering methods, while others rely on supervised labels, templates, priors, or neural sequence labeling.

---

### 2.1 Traditional PRI-based methods

These methods mainly rely on periodic or quasi-periodic patterns in pulse arrival times, usually using TOA / DTOA / PRI as core features.

| Method            | Main idea                                       | Advantages                                          | Limitations                                                          |
| ----------------- | ----------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------- |
| PRI histogram     | Estimate dominant PRI values using histograms   | Simple and interpretable                            | Sensitive to missing pulses, spurious pulses, and dense environments |
| CDIF              | Cumulative Difference Histogram                 | Effective for stable PRI patterns                   | Less robust to complex PRI modulation                                |
| SDIF              | Sequential Difference Histogram                 | Better use of temporal order than simple histograms | Still affected by PRI agility and noise                              |
| PRI transform     | Transform TOA sequences to reveal PRI structure | Classical and widely studied                        | Performance may degrade in highly interleaved scenarios              |
| Sequence matching | Match pulse trains according to PRI templates   | Suitable for emitters with known patterns           | Requires prior knowledge or templates                                |

These methods are close to the classical radar signal sorting definition, but they usually depend on PRI regularities or prior templates and may not be suitable for complex, dense, and frequency-agile electromagnetic environments.

---

### 2.2 Unsupervised clustering methods

These methods directly cluster raw or transformed PDW features. They best match the definition of separating interleaved pulses into emitter-specific clusters.

| Method                       | Main idea                                                       | Strictly unsupervised? | Notes                                                               |
| ---------------------------- | --------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------- |
| K-means                      | Cluster pulses into K groups                                    | Yes                    | Requires the number of emitters K to be known or estimated          |
| GMM                          | Model pulse distributions using Gaussian mixtures               | Yes                    | Usually requires model selection for K                              |
| DBSCAN                       | Density-based clustering with noise handling                    | Yes                    | Does not require predefined K, but is parameter-sensitive           |
| HDBSCAN                      | Hierarchical density-based clustering                           | Yes                    | Suitable for unknown cluster numbers and variable-density scenarios |
| Hierarchical clustering      | Build a cluster tree and cut it according to a criterion        | Yes                    | The cutting criterion strongly affects results                      |
| Spectral clustering          | Cluster based on graph similarity                               | Yes                    | Graph construction is critical                                      |
| Sparse subspace clustering   | Assume pulses from the same emitter lie in a subspace structure | Yes                    | Usually computationally heavier                                     |
| Optimal transport clustering | Cluster based on distributional distances                       | Yes                    | Promising for complex PDW distributions                             |

This category best fits the strict definition: **separating mixed pulses into emitter clusters without using emitter labels during training**.

---

### 2.3 Representation learning + clustering methods

These methods usually learn a more suitable feature representation first and then perform clustering in the learned embedding space.

| Method                            | Main idea                                                                                | Supervision level                                            | Notes                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Autoencoder + clustering          | Learn compressed PDW embeddings and then cluster them                                    | Unsupervised or weakly supervised                            | Suitable when raw features overlap heavily                 |
| Contrastive learning + clustering | Pull similar pulses or sequences closer in feature space                                 | Self-supervised, weakly supervised, or supervised            | Requires a clear definition of positive and negative pairs |
| Transformer encoder + HDBSCAN     | Learn contextual pulse embeddings and then cluster them                                  | Often supervised metric learning, but can be self-supervised | Suitable for long-sequence modeling                        |
| Triplet-loss metric learning      | Make embeddings of same-emitter pulses closer and different-emitter pulses farther apart | Supervised during training, clustering during inference      | Should not be simply regarded as purely unsupervised       |

These methods still output emitter clusters and can therefore solve the deinterleaving task, but their training process may not be purely unsupervised.

---

### 2.4 Supervised sequence labeling and segmentation methods

Some recent works formulate radar pulse deinterleaving as a supervised sequence labeling, semantic segmentation, or instance segmentation task.

| Method                        | Main idea                                                                                 | Comment                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| RNN / LSTM / GRU              | Model temporal dependencies in pulse sequences                                            | Usually requires labeled training data                      |
| TCN                           | Use temporal convolution for long-sequence modeling                                       | Efficient for long sequences                                |
| Transformer                   | Model long-range dependencies among pulses                                                | Powerful but data-hungry                                    |
| GCN                           | Build pulse relation graphs and classify nodes or edges                                   | Suitable for relational modeling                            |
| U-Net / semantic segmentation | Convert pulse sequences into image-like representations and perform semantic segmentation | Requires dense labels                                       |
| Mask R-CNN / SOLOv2           | Treat different emitters as different instances in an image-like representation           | Closer to instance segmentation than traditional clustering |

These methods can solve the deinterleaving task, but they **do not strictly belong to unsupervised clustering-based deinterleaving methods**.

---

### 2.5 Which methods match the strict definition?

> **Strict definition:** Given an interleaved pulse sequence, cluster the pulses into groups without using emitter labels during training, where each group corresponds to one emitter.

| Method Family                           | Matches the task? | Strictly unsupervised? | Recommendation | Comment                                                                        |
| --------------------------------------- | ----------------- | ---------------------- | -------------- | ------------------------------------------------------------------------------ |
| DBSCAN / HDBSCAN                        | Yes               | Yes                    | ⭐⭐⭐⭐⭐          | Strong unsupervised baselines for unknown K                                    |
| K-means / GMM                           | Yes               | Yes                    | ⭐⭐⭐⭐           | Simple and reproducible, but usually requires K or model selection             |
| Hierarchical clustering                 | Yes               | Yes                    | ⭐⭐⭐⭐           | Interpretable, but the cutting criterion is important                          |
| PRI histogram / CDIF / SDIF             | Yes               | Mostly                 | ⭐⭐⭐            | Classical methods for clear PRI structures                                     |
| Autoencoder + clustering                | Yes               | Usually                | ⭐⭐⭐⭐           | Useful for representation learning, but the training objective must be checked |
| Contrastive learning + clustering       | Yes               | Depends                | ⭐⭐⭐⭐           | Can be self-supervised or supervised depending on pair construction            |
| Transformer + triplet loss + clustering | Yes               | No                     | ⭐⭐⭐            | Useful as supervised representation learning or upper-bound comparison         |
| TCN / RNN sequence labeling             | Yes               | No                     | ⭐⭐⭐            | Usually supervised learning                                                    |
| GCN node/edge classification            | Yes               | No / semi-supervised   | ⭐⭐⭐            | Useful for relational modeling, but the task setting must be distinguished     |
| U-Net / semantic segmentation           | Related           | No                     | ⭐⭐             | Requires dense labels and is closer to segmentation                            |
| Mask R-CNN / SOLOv2                     | Related           | No                     | ⭐⭐             | Instance-segmentation formulation                                              |
| RadSeg-style activity segmentation      | Related           | No                     | ⭐⭐             | Detects radar activity, not standard emitter clustering                        |

---

## 3. Datasets

This section summarizes datasets, simulation data, and related-task data for radar pulse deinterleaving. The recommendation level is based on **task relevance, data openness, availability of ground-truth labels, suitability as a benchmark, and clarity of data documentation**.

> Note: The star rating here is the recommendation level of this repository, not GitHub stars.

---

### 3.1 Dataset recommendation levels

| Recommendation | Meaning                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| ⭐⭐⭐⭐⭐          | Strongly recommended; suitable as a primary benchmark or core dataset        |
| ⭐⭐⭐⭐           | Recommended; suitable for reproduction, comparison, or method validation     |
| ⭐⭐⭐            | Useful reference, but limited by documentation, scale, or task setting       |
| ⭐⭐             | Related-task data or more suitable for teaching / demo / auxiliary reference |
| ⭐              | Supplementary only; not recommended as a main experimental dataset           |

---

### 3.2 Dataset summary

| Dataset                                    | Task Fit                                      | Data Type                                       | Labels                                                 | Open Source?               | Recommendation | Links                                                                                                                                                                                | Notes                                                                                                          |
| ------------------------------------------ | --------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------ | -------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Turing Synthetic Radar Dataset, TSRD       | Standard PDW pulse deinterleaving             | Synthetic PDW sequences                         | Emitter ground-truth labels                            | Yes                        | ⭐⭐⭐⭐⭐          | [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) | Currently the most recommended primary benchmark                                                               |
| radar_data_Kmeans data                     | Radar pulse sorting                           | Simulated PDW / project data                    | Limited data documentation                             | Partly, included in repo   | ⭐⭐⭐            | [GitHub](https://github.com/zda2019/radar_data_Kmeans)                                                                                                                               | Suitable for K-means engineering implementation and small experiments                                          |
| Stream-ConAEnet data                       | Radar pulse sorting                           | `.mat` pulse feature data                       | May contain labels, but field descriptions are limited | Partly, included in repo   | ⭐⭐⭐            | [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)                                                                                             | Useful for representation learning; data format should be checked before reproduction                          |
| HMC-RFN simulation data                    | PRI-based deinterleaving                      | Simulated TOA / PRI data                        | Simulation-generated labels                            | Can be generated from code | ⭐⭐⭐            | [GitHub](https://github.com/xm980426/HMC-RFN)                                                                                                                                        | Suitable for PRI temporal-structure modeling                                                                   |
| 2nd-EBDSC data                             | Competition-style signal extraction / sorting | PDW sequences                                   | Labels for training / validation data                  | Partly, via cloud links    | ⭐⭐⭐            | [GitHub](https://github.com/framist/2nd-EBDSC)                                                                                                                                       | Closer to supervised or template-assisted tasks                                                                |
| EW Signal Intelligence demo data           | Educational deinterleaving demo               | CSV pulse data                                  | Demo labels / tracks                                   | Yes                        | ⭐⭐             | [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)                                                                                                          | Good for understanding the basic workflow, not suitable as a research benchmark                                |
| RadSeg                                     | Radar activity segmentation                   | Signal sequence / sample-wise segmentation data | Sample-wise annotations                                | Yes                        | ⭐⭐             | [GitHub](https://github.com/abcxyzi/radseg)                                                                                                                                          | Related task, not standard emitter clustering                                                                  |
| Real measured radar deinterleaving dataset | Real radar deinterleaving                     | Real PDW / IQ                                   | Usually unavailable                                    | Very limited public access | ⭐              | No stable public benchmark yet                                                                                                                                                       | Real data is difficult to release and label, so it is not currently suitable as a public reproducibility basis |

---

### 3.3 Key dataset notes

#### 3.3.1 Turing Synthetic Radar Dataset / TSRD

**Recommendation:** ⭐⭐⭐⭐⭐
**Suitable for:** standard benchmark, unsupervised clustering baselines, deep representation learning, supervised / semi-supervised evaluation
**Data type:** synthetic PDW sequences
**Labels:** ground-truth emitter labels
**Related code:** Turing Deinterleaving Challenge

TSRD is currently one of the most suitable public datasets for radar pulse deinterleaving. It is used together with the Turing Deinterleaving Challenge and provides a clear task definition, data loading interface, baselines, and evaluation protocol.

Maintainer note: If the goal is to study **unsupervised radar pulse deinterleaving**, this dataset is the recommended starting point. HDBSCAN, DBSCAN, K-means, and hierarchical clustering can be used to build initial baselines.

**Links:** [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) / [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

---

#### 3.3.2 radar_data_Kmeans data

**Recommendation:** ⭐⭐⭐
**Suitable for:** K-means deinterleaving introduction, engineering reference, embedded deployment reference
**Data type:** simulated PDW / project data
**Labels:** data and evaluation descriptions are less complete than standard benchmarks

This project contains data and simulation scripts related to K-means-based radar pulse sorting. It is more suitable for understanding traditional clustering-based sorting and engineering workflows.

Maintainer note: Suitable as supplementary experimental or engineering reference. It is not recommended as the only main benchmark for research papers.

**Link:** [GitHub](https://github.com/zda2019/radar_data_Kmeans)

---

#### 3.3.3 Stream-ConAEnet data

**Recommendation:** ⭐⭐⭐
**Suitable for:** contrastive autoencoder, representation learning, streaming deinterleaving experiments
**Data type:** `.mat` pulse feature data
**Labels:** data files and model weights are included, but field definitions and data documentation are limited

This data is more suitable for studying deep representation learning combined with clustering. Since the data documentation is limited, the meaning of `.mat` fields and label definitions should be checked before reproduction.

**Link:** [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

---

#### 3.3.4 2nd-EBDSC data

**Recommendation:** ⭐⭐⭐
**Suitable for:** competition-style interleaved signal extraction, supervised sequence labeling, template-assisted tasks
**Data type:** PDW sequences
**Labels:** training / validation data contain labels; test data depends on the competition setting

This data is related to radar pulse deinterleaving, but it is not a strict unsupervised emitter-clustering dataset. It is more suitable for supervised or template-assisted interleaved signal extraction tasks.

**Link:** [GitHub](https://github.com/framist/2nd-EBDSC)

---

## 4. Methods and Implementations

This section summarizes open-source methods, code implementations, and reproducible frameworks related to radar pulse deinterleaving. The recommendation level is based on **task relevance, code availability, reproducibility, suitability as a baseline, and method representativeness**.

> Note: The star rating here is the recommendation level of this repository, not GitHub stars.

---

### 4.1 Method recommendation levels

| Recommendation | Meaning                                                                               |
| -------------- | ------------------------------------------------------------------------------------- |
| ⭐⭐⭐⭐⭐          | Strongly recommended; suitable as a main baseline or primary reproduction target      |
| ⭐⭐⭐⭐           | Recommended; representative and suitable for comparison or extension                  |
| ⭐⭐⭐            | Useful reference, but limited by task setting, data documentation, or reproducibility |
| ⭐⭐             | Related task or teaching demo; suitable as auxiliary reference                        |
| ⭐              | Supplementary only; not recommended as a main method                                  |

---

### 4.2 Method and code summary

| Project / Method                        | Method Type                                    | Supervision                                                 | Strictly Unsupervised? | Recommendation | Links                                                                                    | Notes                                                                    |
| --------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------- | ---------------------- | -------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Turing HDBSCAN baseline                 | HDBSCAN raw PDW clustering                     | Unsupervised                                                | Yes                    | ⭐⭐⭐⭐⭐          | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)       | Currently the most recommended standard unsupervised baseline            |
| radar_data_Kmeans                       | K-means clustering / APG / ZYNQ deployment     | Unsupervised clustering                                     | Yes, but requires K    | ⭐⭐⭐⭐           | [GitHub](https://github.com/zda2019/radar_data_Kmeans)                                   | Useful for classical clustering and engineering deployment               |
| DBSCAN / HDBSCAN clustering             | Density-based clustering                       | Unsupervised                                                | Yes                    | ⭐⭐⭐⭐⭐          | Can be implemented based on the Turing framework                                         | Suitable for unknown emitter numbers                                     |
| K-means / GMM / hierarchical clustering | Classical clustering baselines                 | Unsupervised                                                | Yes                    | ⭐⭐⭐⭐           | Can be implemented based on the Turing framework                                         | Suitable as a basic baseline group                                       |
| Stream-ConAEnet                         | Contrastive autoencoder / streaming learning   | Unsupervised / weakly supervised setting should be verified | Partly fits            | ⭐⭐⭐            | [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet) | Useful for representation learning, but data documentation is limited    |
| HMC-RFN                                 | Hidden Markov Chains / Residual Fence Networks | Model-driven / prior-driven                                 | No                     | ⭐⭐⭐            | [GitHub](https://github.com/xm980426/HMC-RFN)                                            | Suitable for PRI-based temporal-structure modeling                       |
| 2nd-EBDSC solution                      | Wide-value embeddings / TCN / masking          | Supervised / template-assisted                              | No                     | ⭐⭐⭐            | [GitHub](https://github.com/framist/2nd-EBDSC)                                           | Valuable engineering reference, but not strictly unsupervised clustering |
| EW Signal Intelligence Demo             | PRI-based and feature-based grouping           | Mostly unsupervised / demo                                  | Mostly fits            | ⭐⭐             | [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)              | Useful for teaching and quick prototyping                                |
| RadSeg-style segmentation               | Radar activity segmentation                    | Supervised segmentation                                     | No                     | ⭐⭐             | [GitHub](https://github.com/abcxyzi/radseg)                                              | Related task, not standard PDW emitter clustering                        |

---

### 4.3 Key method notes

#### 4.3.1 Turing HDBSCAN baseline

**Recommendation:** ⭐⭐⭐⭐⭐
**Method type:** HDBSCAN on raw PDW features
**Supervision:** unsupervised
**Strict task fit:** yes
**Suitable for:** standard baseline, unsupervised clustering comparison, baseline for later representation learning methods

The HDBSCAN baseline in the Turing Deinterleaving Challenge is one of the most suitable open-source baselines for radar pulse deinterleaving. It does not require a predefined number of emitters and can directly cluster pulses in the PDW feature space.

Maintainer note: To build an unsupervised deinterleaving experimental framework, it is recommended to first reproduce this baseline and then add DBSCAN, K-means, GMM, and hierarchical clustering for comparison.

**Link:** [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)

---

#### 4.3.2 radar_data_Kmeans

**Recommendation:** ⭐⭐⭐⭐
**Method type:** K-means clustering
**Supervision:** unsupervised clustering
**Strict task fit:** mostly yes, but requires predefined or estimated K
**Suitable for:** introduction to classical clustering-based sorting, engineering implementation, embedded deployment reference

This project implements radar pulse sorting using K-means clustering and includes APG optimization, C++ implementation, and ZYNQ deployment. Compared with purely research-oriented code, it is more engineering-focused.

Maintainer note: Suitable for understanding the basic workflow of clustering-based deinterleaving. Since K-means requires the number of emitters K, it should be combined with K estimation or model selection in realistic unknown-emitter scenarios.

**Link:** [GitHub](https://github.com/zda2019/radar_data_Kmeans)

---

#### 4.3.3 Stream-ConAEnet

**Recommendation:** ⭐⭐⭐
**Method type:** contrastive autoencoder + streaming learning
**Supervision:** may include unsupervised, weakly supervised, or label-based training stages; code inspection is needed
**Strict task fit:** partly fits
**Suitable for:** representation learning, streaming deinterleaving, deep clustering reference

This project learns pulse representations using a contrastive autoencoder and then performs deinterleaving with clustering or dynamic center modules. It is useful as a reference for the idea of “deep representation learning + clustering”.

Maintainer note: This project is valuable as a reference, but its data documentation and training details should be carefully checked. It is not recommended to directly cite it as a strictly unsupervised method without verification.

**Link:** [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

---

#### 4.3.4 HMC-RFN

**Recommendation:** ⭐⭐⭐
**Method type:** PRI-based temporal modeling
**Supervision:** model-driven / prior-driven
**Strict task fit:** not pure unsupervised clustering
**Suitable for:** PRI structure modeling, simulation experiments, comparison with classical temporal models

HMC-RFN uses Hidden Markov Chains and Residual Fence Networks to model temporal structures in radar pulse deinterleaving. It is not a purely data-driven clustering method and relies more on PRI patterns and prior modeling.

**Link:** [GitHub](https://github.com/xm980426/HMC-RFN)

---

#### 4.3.5 2nd-EBDSC solution

**Recommendation:** ⭐⭐⭐
**Method type:** wide-value embeddings + TCN + masking
**Supervision:** supervised / template-assisted
**Strict task fit:** no
**Suitable for:** sequence modeling, competition-style signal extraction, engineering reference

This project is related to interleaved pulse sequence processing, but it is closer to supervised or template-assisted signal extraction than strict unsupervised emitter clustering.

Maintainer note: It is useful as an engineering and deep sequence modeling reference, but its task setting should be clearly distinguished from standard unsupervised radar pulse deinterleaving.

**Link:** [GitHub](https://github.com/framist/2nd-EBDSC)

---

#### 4.3.6 EW Signal Intelligence Deinterleaving Demo

**Recommendation:** ⭐⭐
**Method type:** PRI-based / feature-based grouping
**Supervision:** mostly unsupervised / demo
**Strict task fit:** conceptually mostly fits
**Suitable for:** teaching, quick prototyping, understanding the basic deinterleaving workflow

This project provides a simple Python demo for understanding the basic deinterleaving workflow, but it is not suitable as a research-level benchmark.

**Link:** [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)

---

#### 4.3.7 RadSeg

**Recommendation:** ⭐⭐
**Method type:** radar activity segmentation
**Supervision:** supervised segmentation
**Strict task fit:** no
**Suitable for:** related-task reference, radar activity detection and segmentation

RadSeg focuses on sample-wise radar activity segmentation. It is not standard PDW emitter clustering. It can be used as a related-task reference but should not be mixed with standard radar pulse deinterleaving datasets.

**Link:** [GitHub](https://github.com/abcxyzi/radseg)

---

## 5. Recommended Experimental Setup

If the research goal is **unsupervised radar pulse deinterleaving**, the following setup is recommended.

---

### 5.1 Primary benchmark

Use **Turing Synthetic Radar Dataset** as the primary benchmark because it provides:

* clear task definition;
* open data;
* ground-truth emitter labels;
* standard clustering metrics;
* baseline code;
* support for unknown numbers of emitters.

---

### 5.2 Baseline methods

The following baselines are recommended:

| Category                | Methods                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| Raw-feature clustering  | K-means, GMM, DBSCAN, HDBSCAN, hierarchical clustering           |
| PRI-based methods       | PRI histogram, CDIF, SDIF, PRI transform                         |
| Hybrid methods          | RF/PW/DOA coarse clustering + PRI refinement                     |
| Representation learning | Autoencoder + HDBSCAN, contrastive encoder + clustering          |
| Supervised upper bound  | Transformer metric learning, TCN, GCN, segmentation-based models |

---

### 5.3 Suggested pipeline

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

## 6. Recommended Reading and Starting Points

### Best starting point for standard deinterleaving

* [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
* [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)

### Best starting point for simple unsupervised clustering

* [radar_data_Kmeans](https://github.com/zda2019/radar_data_Kmeans)
* HDBSCAN baseline in the Turing Deinterleaving Challenge

### Starting point for representation learning

* [Radar Pulse Sorting Based on Stream-ConAEnet](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

### Starting point for PRI-based temporal modeling

* [HMC-RFN](https://github.com/xm980426/HMC-RFN)

### Related but not strict deinterleaving tasks

* [2nd-EBDSC](https://github.com/framist/2nd-EBDSC)
* [RadSeg](https://github.com/abcxyzi/radseg)

---

## 7. Notes

* Compared with general RF modulation recognition or RF fingerprinting, open-source resources for radar pulse deinterleaving are still limited.
* Most public datasets are synthetic because real radar PDW or IQ data is difficult to release and difficult to label reliably.
* When reporting experimental results, it is important to clearly state whether the method is unsupervised, supervised, semi-supervised, or supervised representation learning followed by clustering.
* For strict unsupervised radar pulse deinterleaving, Turing TSRD + HDBSCAN / DBSCAN / K-means / hierarchical clustering is currently the most transparent and reproducible starting point.
* The recommendation levels in this repository may be updated as projects evolve, data availability changes, or reproduction results become available.

---

## 8. Citation and Contribution

If you use this resource list, please cite the original papers, datasets, and code repositories whenever possible.

Contributions are welcome, including:

* newly released radar pulse deinterleaving datasets;
* open-source implementations of PRI-based, clustering-based, or deep learning-based deinterleaving methods;
* reproducible benchmark results;
* corrections to dataset availability information;
* notes on whether a method is strictly unsupervised.

When submitting a new resource, please include the following information if possible:

```text
Project name:
Project link:
Task type:
Method type:
Open-source code:
Open-source data:
Labels available:
Strictly unsupervised:
Reason for recommendation:
Notes:
```
