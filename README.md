# Radar Pulse Deinterleaving Resources

This repository collects relevant resources related to **Radar Pulse Deinterleaving / Radar Signal Sorting**, including task definitions, commonly used datasets, open-source implementations, representative methods, and suggested evaluation protocols.

Radar pulse deinterleaving aims to separate an interleaved pulse stream into groups corresponding to different radar emitters. In a typical Pulse Description Word (PDW) setting, each pulse is described by parameters such as **Time of Arrival (TOA)**, **Pulse Repetition Interval (PRI)**, **Radio Frequency / Carrier Frequency (RF/CF)**, **Pulse Width (PW)**, **Pulse Amplitude (PA)**, and **Direction / Angle of Arrival (DOA/AOA)**.

---

## 1. Overview

### 1.1 What is radar pulse deinterleaving?

Radar pulse deinterleaving, also called **radar signal sorting**, is the task of assigning pulses from a mixed electromagnetic environment to their corresponding emitters.

Given an interleaved pulse sequence:

```text
P = {p1, p2, ..., pN}
```

where each pulse `pi` is usually represented by a PDW feature vector:

```text
pi = [TOA, RF/CF, PW, PA, DOA/AOA, ...]
```

the goal is to output a partition:

```text
C = {C1, C2, ..., CK}
```

where each cluster `Ck` corresponds to one radar emitter.

In the strictest sense, this is an **unsupervised clustering problem** because:

- the number of emitters is usually unknown;
- the emitter identities are not fixed classes;
- pulses from the same emitter should be grouped together;
- pulses from different emitters should be separated.

However, many modern methods formulate the task as supervised learning, semi-supervised learning, metric learning, sequence labeling, semantic segmentation, or instance segmentation. These methods may still solve the deinterleaving task, but they are not always purely unsupervised clustering methods.

### 1.2 Common input features

| Feature | Meaning | Usage |
|---|---|---|
| TOA | Time of Arrival | Used to derive PRI and temporal patterns |
| PRI / DTOA | Pulse Repetition Interval / Difference of TOA | Core feature for traditional deinterleaving |
| RF / CF | Radio Frequency / Carrier Frequency | Useful for fixed-frequency and frequency-agile emitters |
| PW | Pulse Width | Helps distinguish emitters with different waveform settings |
| PA / AMP | Pulse Amplitude | Useful but sensitive to propagation and receiver effects |
| DOA / AOA | Direction / Angle of Arrival | Strong spatial cue when available |

### 1.3 Common evaluation metrics

| Metric | Description |
|---|---|
| V-measure | Harmonic mean of homogeneity and completeness |
| Homogeneity | Each predicted cluster contains pulses mainly from one emitter |
| Completeness | Pulses from the same emitter are assigned to the same cluster |
| ARI | Adjusted Rand Index |
| AMI | Adjusted Mutual Information |
| Pairwise F1 | Pairwise same-emitter classification performance |
| MCC | Matthews Correlation Coefficient for pairwise matching |

---

## 2. Method Taxonomy

Radar pulse deinterleaving methods can be roughly divided into the following categories.

### 2.1 Traditional PRI-based methods

These methods rely on temporal regularities in pulse arrival times.

| Method | Main idea | Advantages | Limitations |
|---|---|---|---|
| PRI histogram | Estimate dominant PRI values using histograms | Simple and interpretable | Sensitive to missing pulses, spurious pulses, and dense environments |
| CDIF | Cumulative Difference Histogram | Effective for stable PRI patterns | Less robust for complex PRI modulation |
| SDIF | Sequential Difference Histogram | Better temporal exploitation than simple histogram | Still limited by PRI agility and noise |
| PRI transform | Transform TOA sequence to reveal PRI structure | Classical and widely studied | Performance drops in highly interleaved scenarios |
| Sequence matching | Match pulse trains according to PRI templates | Useful when emitter patterns are known | Requires prior knowledge or templates |

These methods are close to the classical radar signal sorting definition, but they are not always general-purpose clustering algorithms.

### 2.2 Unsupervised clustering methods

These methods directly cluster PDW features or transformed features.

| Method | Main idea | Whether it matches strict unsupervised deinterleaving |
|---|---|---|
| K-means | Cluster pulses into K groups | Yes, but requires the number of emitters K |
| GMM | Model pulses using Gaussian mixtures | Yes, but usually requires model selection for K |
| DBSCAN | Density-based clustering with noise handling | Yes, no need to predefine K |
| HDBSCAN | Hierarchical density-based clustering | Yes, often stronger than DBSCAN for variable densities |
| Hierarchical clustering | Build cluster tree and cut into groups | Yes, but cut criterion is important |
| Spectral clustering | Cluster based on graph similarity | Yes, but graph construction is critical |
| Sparse subspace clustering | Assume pulses from each emitter lie in a subspace | Yes, but computationally heavier |
| Optimal transport clustering | Cluster based on distributional distances | Yes, promising for complex PDW distributions |

This category best matches the definition: **separate mixed pulses into emitter clusters without using emitter labels during training**.

### 2.3 Representation learning + clustering

These methods first learn a feature representation and then perform clustering.

| Method | Main idea | Supervision level |
|---|---|---|
| Autoencoder + clustering | Learn compressed PDW embeddings, then cluster | Unsupervised or weakly supervised |
| Contrastive learning + clustering | Pull similar pulses/sequences closer in embedding space | Self-supervised, weakly supervised, or supervised |
| Transformer encoder + HDBSCAN | Learn contextual pulse embeddings and cluster them | Often supervised metric learning, but can be self-supervised |
| Triplet-loss metric learning | Learn an embedding where same-emitter pulses are close | Supervised during training, unsupervised during inference |

These methods still produce emitter clusters, but their training process may not be purely unsupervised.

### 2.4 Supervised sequence labeling and segmentation methods

Some recent works formulate deinterleaving as a supervised labeling problem.

| Method | Main idea | Comment |
|---|---|---|
| RNN / LSTM / GRU | Model temporal dependencies in pulse sequences | Usually requires labeled training data |
| TCN | Use temporal convolution for pulse sequence labeling | Efficient for long sequences |
| Transformer | Capture long-range dependencies among pulses | Strong but data-hungry |
| GCN | Build pulse relationship graph and classify edges/nodes | Good for relational modeling |
| U-Net / semantic segmentation | Convert pulse sequence into image-like representation | Requires dense labels |
| Mask R-CNN / SOLOv2 | Treat emitters as instances in a pulse feature map | Closer to instance segmentation than clustering |

These methods can solve deinterleaving, but they do **not** strictly match the unsupervised clustering definition.

---

## 3. Open-Source Projects and Datasets

### 3.1 Summary

| Project | Task Type | Method | Dataset Open? | Strictly Unsupervised? | Links |
|---|---|---|---|---|---|
| Turing Deinterleaving Challenge | PDW deinterleaving | HDBSCAN baseline, challenge framework | Yes | Baseline yes | [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) |
| Turing Synthetic Radar Dataset | Synthetic PDW benchmark | Dataset and evaluation benchmark | Yes | N/A | [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset) |
| radar_data_Kmeans | Radar pulse sorting | K-means, APG optimization, ZYNQ deployment | Partly, in repo | Yes, but K is required | [GitHub](https://github.com/zda2019/radar_data_Kmeans) |
| Stream-ConAEnet | Radar pulse sorting | Contrastive autoencoder, streaming learning, clustering | Partly, in repo | Partly / unclear | [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet) |
| HMC-RFN | PRI-based deinterleaving | Hidden Markov Chains, Residual Fence Networks | Simulation code | No | [GitHub](https://github.com/xm980426/HMC-RFN) |
| 2nd-EBDSC | Competition signal extraction / sorting | Wide-value embeddings, TCN, masking | Partly, via cloud link | No | [GitHub](https://github.com/framist/2nd-EBDSC) |
| EW Signal Intelligence Demo | Educational deinterleaving demo | PRI-based and feature-based grouping | Sample data | Mostly yes | [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence) |
| RadSeg | Radar activity segmentation | Sample-wise segmentation | Yes | No | [GitHub](https://github.com/abcxyzi/radseg) |

---

## 4. Detailed Resources

### 4.1 Turing Deinterleaving Challenge

**Type:** PDW-based radar pulse deinterleaving benchmark  
**Dataset:** Turing Synthetic Radar Dataset  
**Method baseline:** HDBSCAN on raw PDW features  
**Task definition:** Partition an interleaved pulse sequence into emitter-specific pulse groups

The Turing Deinterleaving Challenge is one of the most standard open resources for radar pulse deinterleaving. It provides a large-scale synthetic PDW dataset, baseline code, evaluation tools, and a clear task formulation.

**Why it is important:**

- It directly matches the standard deinterleaving definition.
- It supports unknown numbers of emitters.
- It evaluates clustering results using metrics such as V-measure, ARI, AMI, homogeneity, and completeness.
- It can be used to compare traditional clustering methods, representation learning methods, and deep learning methods.

**Dataset characteristics:**

- Synthetic PDW sequences
- Multiple emitters per scenario
- Ground-truth emitter labels for evaluation
- PDW features such as TOA, RF/CF, PW, amplitude, and AOA
- Suitable for both supervised and unsupervised research

**Links:** [GitHub](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) / [Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)

---

### 4.2 radar_data_Kmeans

**Type:** Radar pulse sorting using K-means  
**Method:** K-means clustering, center-frequency filtering, APG optimization, C++ implementation, ZYNQ deployment  
**Dataset:** Simulation data and project data included in repository

This project implements radar pulse sorting using K-means clustering. It is close to the strict unsupervised clustering formulation because it groups pulses according to feature similarity.

**Main idea:**

1. Generate or load radar pulse sequence data.
2. Use filtering and feature preprocessing to estimate center-frequency-related information.
3. Apply K-means clustering to assign pulses to radar sources.
4. Optimize the computation using APG and C++ implementation.
5. Deploy the algorithm on ZYNQ hardware.

**Advantages:**

- Easy to understand.
- Engineering-oriented implementation.
- Useful for embedded or real-time radar sorting experiments.

**Limitations:**

- K-means requires the number of emitters to be known or estimated.
- The dataset and evaluation protocol are less standardized than Turing TSRD.
- Performance may degrade under dense, overlapping, or highly agile emitter conditions.

**Links:** [GitHub](https://github.com/zda2019/radar_data_Kmeans)

---

### 4.3 Radar Pulse Sorting Based on Stream-ConAEnet

**Type:** Deep representation learning for radar pulse sorting  
**Method:** Contrastive autoencoder, stream learning, dynamic center module, clustering  
**Dataset:** `standard_data.mat` and model weights are included, but dataset documentation is limited

This project studies radar pulse sorting using a contrastive autoencoder-based representation learning method. It aims to extract deep category-level features from pulse samples and then perform sorting or clustering.

**Main idea:**

1. Use ConAEnet to learn discriminative pulse representations.
2. Use contrastive learning to improve feature separability.
3. Apply a stream learning mechanism to handle non-stationary pulse streams.
4. Use a dynamic center module to support evolving classes or online sorting.

**Advantages:**

- More powerful than raw-feature clustering when PDW features overlap.
- Suitable for online or streaming deinterleaving studies.
- Good starting point for self-supervised or weakly supervised representation learning.

**Limitations:**

- The dataset description is not sufficiently detailed.
- It is not fully clear whether all training stages are label-free.
- Reproducibility may require additional inspection of the code and data format.

**Links:** [GitHub](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

---

### 4.4 HMC-RFN

**Type:** PRI-based radar signal deinterleaving  
**Method:** Hidden Markov Chains and Residual Fence Networks  
**Dataset:** Simulation scripts and MATLAB implementation

HMC-RFN models radar pulse deinterleaving using temporal PRI structure and prior assumptions. It is more model-driven than clustering-driven.

**Main idea:**

- Use TOA / PRI sequence characteristics.
- Model pulse train structure using Hidden Markov Chains.
- Use Residual Fence Networks to improve deinterleaving under jittered or complex PRI conditions.

**Advantages:**

- Useful for studying PRI-based temporal modeling.
- MATLAB implementation is relatively accessible.
- Suitable for fixed PRI, staggered PRI, sliding PRI, and jittered PRI experiments.

**Limitations:**

- Not a pure unsupervised clustering method.
- Relies on PRI structure and simulation assumptions.
- Less suitable for unknown, dense, highly agile emitter scenarios without prior modeling.

**Links:** [GitHub](https://github.com/xm980426/HMC-RFN)

---

### 4.5 2nd-EBDSC Electromagnetic Big Data Challenge Solution

**Type:** Competition solution for interleaved electromagnetic signal extraction  
**Method:** Wide-value embeddings, TCN, masking, hard sample mining  
**Dataset:** Training and validation data are described and partly available through cloud links

This repository contains a winning solution for an electromagnetic big data challenge. The task is related to extracting known and unknown signal patterns from interleaved pulse sequences.

**Main idea:**

1. Represent wide-range PDW numerical values using embeddings.
2. Use TCN to model long pulse sequences.
3. Use masking and hard sample mining to simulate more difficult interleaving cases.
4. Predict signal labels or extract signal patterns from mixed sequences.

**Advantages:**

- Strong engineering reference.
- Useful for sequence modeling and competition-style deinterleaving.
- Handles large-value PDW features in a neural network framework.

**Limitations:**

- Not a pure unsupervised clustering task.
- Uses known templates or labeled examples.
- More similar to supervised or open-set sequence labeling than classical clustering.

**Links:** [GitHub](https://github.com/framist/2nd-EBDSC)

---

### 4.6 EW Signal Intelligence Deinterleaving Demo

**Type:** Educational EW signal intelligence demo  
**Method:** PRI-based and feature-based grouping  
**Dataset:** Sample CSV data

This repository provides a simple Python demonstration for deinterleaving radar or communication pulses.

**Advantages:**

- Easy to read.
- Good for educational purposes.
- Useful as a minimal starting point for implementing DBSCAN, HDBSCAN, or custom clustering methods.

**Limitations:**

- Not a research-level benchmark.
- Limited dataset scale.
- Mostly suitable for learning and prototyping.

**Links:** [GitHub](https://github.com/hugodrak/deinterleaving_ew_signal_intelligence)

---

### 4.7 RadSeg

**Type:** Radar pulse activity segmentation  
**Method:** Sample-wise segmentation of radar activity  
**Dataset:** Open synthetic radar dataset

RadSeg is related to radar pulse analysis but does not strictly match PDW-based emitter deinterleaving. It focuses on segmenting radar activity in long signal sequences.

**Usefulness:**

- Useful for radar activity detection and segmentation.
- Useful if the input is raw or sampled signal sequences rather than PDW tables.
- Can inspire segmentation-based deinterleaving formulations.

**Limitations:**

- Not a standard emitter-clustering dataset.
- Requires sample-wise labels.
- More suitable for activity segmentation than pulse source sorting.

**Links:** [GitHub](https://github.com/abcxyzi/radseg)

---

## 5. Dataset Availability

| Dataset / Project | Open Source? | Data Type | Ground Truth Labels | Suitable for Strict Deinterleaving? |
|---|---|---|---|---|
| Turing Synthetic Radar Dataset | Yes | Synthetic PDW | Yes | Yes |
| radar_data_Kmeans | Partly | Simulated PDW / project data | Not fully standardized | Yes, for small experiments |
| Stream-ConAEnet data | Partly | `.mat` pulse feature data | Likely yes, but not well documented | Partly |
| HMC-RFN simulation | Code available | Simulated TOA / PRI | Generated labels | Partly |
| 2nd-EBDSC data | Partly | PDW sequences | Yes for train/validation | Partly |
| EW Signal Intelligence demo | Yes | CSV pulse data | Demo labels / tracks | Educational only |
| RadSeg | Yes | Signal sequence / segmentation data | Sample-wise annotations | No, related task |
| Real measured radar deinterleaving dataset | Rarely available | Real PDW / IQ | Usually unavailable | Very limited public access |

---

## 6. Recommended Experimental Setup

For researchers who want to study **unsupervised radar pulse deinterleaving**, the following setup is recommended.

### 6.1 Primary benchmark

Use the **Turing Synthetic Radar Dataset** as the main benchmark because it has:

- clear task definition;
- open data;
- ground-truth emitter labels;
- standard clustering metrics;
- baseline code;
- support for unknown emitter numbers.

### 6.2 Baseline methods

Implement and compare the following baselines:

| Category | Methods |
|---|---|
| Raw-feature clustering | K-means, GMM, DBSCAN, HDBSCAN, hierarchical clustering |
| PRI-based methods | PRI histogram, CDIF, SDIF, PRI transform |
| Hybrid methods | RF/PW/DOA coarse clustering + PRI refinement |
| Representation learning | Autoencoder + HDBSCAN, contrastive encoder + clustering |
| Supervised upper bound | Transformer metric learning, TCN, GCN, segmentation-based models |

### 6.3 Suggested pipeline

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

### 6.4 Important considerations

- Normalize PDW features carefully because TOA, RF, PW, and amplitude may have very different scales.
- Do not assume the number of emitters is known unless the method explicitly requires it.
- Evaluate both homogeneity and completeness.
- Report performance under different emitter densities and scan modes.
- Separate unsupervised methods from supervised or semi-supervised methods in comparisons.
- Synthetic data is currently the most practical option because real radar deinterleaving data with reliable ground truth is rarely public.

---

## 7. Which Methods Match the Strict Definition?

The table below summarizes whether each method family matches the strict definition:

> **Strict definition:** Given interleaved pulses, cluster them into groups where each group corresponds to one emitter, without using emitter labels during training.

| Method Family | Matches the task? | Strictly unsupervised? | Comment |
|---|---|---|---|
| K-means / GMM | Yes | Yes | Requires K or model selection |
| DBSCAN / HDBSCAN | Yes | Yes | Strong baseline for unknown K |
| Hierarchical clustering | Yes | Yes | Cut criterion matters |
| PRI histogram / CDIF / SDIF | Yes | Mostly | Classical deinterleaving, not general clustering |
| Autoencoder + clustering | Yes | Usually | Depends on training objective |
| Contrastive learning + clustering | Yes | Depends | Can be self-supervised or supervised |
| Transformer + triplet loss + clustering | Yes | No | Supervised metric learning during training |
| TCN / RNN sequence labeling | Yes | No | Usually supervised |
| GCN node/edge classification | Yes | No / semi-supervised | Depends on formulation |
| U-Net / semantic segmentation | Related | No | Requires dense labels |
| Mask R-CNN / SOLOv2 | Related | No | Instance segmentation formulation |
| RadSeg-style activity segmentation | Related | No | Detects radar activity, not emitter clustering |

---

## 8. Recommended Reading and Starting Points

### Best starting point for standard deinterleaving

- [Turing Deinterleaving Challenge](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
- [Turing Synthetic Radar Dataset](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)

### Best starting point for simple unsupervised clustering

- [radar_data_Kmeans](https://github.com/zda2019/radar_data_Kmeans)
- HDBSCAN baseline in the Turing Deinterleaving Challenge

### Best starting point for representation learning

- [Radar Pulse Sorting Based on Stream-ConAEnet](https://github.com/thebestdreamer/Radar-pulse-sorting-bases-on-Stream-ConAEnet)

### Best starting point for PRI-based temporal modeling

- [HMC-RFN](https://github.com/xm980426/HMC-RFN)

### Related but not strict deinterleaving

- [2nd-EBDSC](https://github.com/framist/2nd-EBDSC)
- [RadSeg](https://github.com/abcxyzi/radseg)

---

## 9. Notes

- Open-source radar deinterleaving resources are still limited compared with general RF modulation recognition or RF fingerprinting.
- Most public datasets are synthetic because real radar PDW or IQ data is difficult to release and difficult to label.
- When reporting results, it is important to clearly state whether the method is unsupervised, supervised, semi-supervised, or supervised representation learning followed by clustering.
- For a strict unsupervised deinterleaving study, Turing TSRD + HDBSCAN / DBSCAN / K-means / hierarchical clustering is the most transparent starting point.

---

## 10. Citation and Contribution

If you use this list, please cite the original papers and repositories whenever possible.

Contributions are welcome. Suggested contribution types include:

- adding newly released deinterleaving datasets;
- adding open-source implementations of PRI-based, clustering-based, or deep learning-based methods;
- adding reproducible benchmark results;
- correcting dataset availability information;
- adding notes about whether a method is strictly unsupervised.

