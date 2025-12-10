# High Scale Recommendation Modeling

This project implements a **Deep Learning Recommendation Model (DLRM)** using **TorchRec** and **PyTorch** to predict user movie preferences on the **MovieLens 1M dataset**.

The system utilizes state-of-the-art recommendation techniques, specifically designed to handle the combination of categorical (sparse) features and numerical (dense) features efficiently at scale.

## Conceptual Architecture

The core of this project is the DLRM architecture, which addresses the recommendation problem by hybridizing two types of data representations:

### 1. Sparse Features & Embedding Bags
High-cardinality categorical data (IDs and lists) are treated as **Sparse Features**. These are transformed into dense vectors via **Embedding Bags**.
* **User ID & Movie ID:** Mapped to learned embedding vectors.
* **Genres:** A movie can have multiple genres (e.g., `[Action, Adventure]`). This variable-length list is handled via `EmbeddingBagCollection`, which performs a reduction (sum or mean) on the embeddings of the genres to produce a single vector representation for the genre list.

### 2. Dense Features
Continuous or low-cardinality binary data are treated as **Dense Features**.
* **Gender:** Processed as a direct numerical input (Float) fed into the bottom MLP (Multi-Layer Perceptron) of the DLRM.

### 3. Feature Interaction
The DLRM model explicitly models the interaction between the embedding vectors (from sparse features) and the processed dense features. This is typically achieved through dot products, capturing how specific users interact with specific movie attributes.

---

## Data Representation Strategy

A significant challenge in recommendation systems is handling "jagged" data—features that do not have a fixed length across samples (e.g., one movie has 1 genre, another has 5).

### KeyedJaggedTensor
Instead of padding genre lists with zeros (which wastes memory and compute), this project utilizes **TorchRec's `KeyedJaggedTensor`** (KJT).

* **Efficient Storage:** KJT packs all values into a single 1D tensor and maintains a separate "offsets" or "lengths" tensor to reconstruct the structure.
* **Batching:** The custom `collate_fn` dynamically constructs this efficient structure for every batch, allowing the model to ingest variable-length genre sequences natively.

---

## Pipeline Overview

The project follows a modular Machine Learning pipeline:

### 1. Data Ingestion & Engineering
* **Source:** MovieLens 1M (Users, Movies, Ratings).
* **Normalization:** Ratings are normalized for regression tasks (though this specific implementation focuses on binary classification).
* **Encoding:**
    * `LabelEncoder` is used to map string-based Genres to integer IDs.
    * Ratings are binarized (`rating > 3` $\rightarrow$ `1`, else `0`) to frame the problem as Click-Through Rate (CTR) prediction.

### 2. The Dataset Class (`MovieLensDataset`)
A custom PyTorch `Dataset` wrapper that prepares the raw tensors. It creates a dictionary output containing:
* `userid`, `movieid` (Long Tensors)
* `genres` (Variable length Long Tensor)
* `gender` (Float Tensor)
* `label` (Target binary)

### 3. Model Training (`DLRMRecommender`)
The training loop utilizes the **DLRM** model from the `torchrec` library.
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) is used, as the model outputs raw logits for the probability of a positive rating.
* **Optimizer:** Adam optimizer adapts learning rates for the embedding parameters.

### 4. Evaluation
The model is evaluated on a held-out validation set using industry-standard metrics:
* **ROC-AUC (Area Under the Curve):** Measures the model's ability to distinguish between high and low ratings.
* **Accuracy:** Measures the raw percentage of correct classifications at a 0.5 threshold.

---

## Tech Stack

* **TorchRec:** For efficient handling of sparse features, jagged tensors, and DLRM architecture.
* **PyTorch:** Core deep learning framework.
* **Pandas & PyArrow:** High-performance data manipulation and Parquet file handling.
* **Scikit-Learn:** Metrics calculation (AUC, Accuracy) and preprocessing (Label Encoding).
