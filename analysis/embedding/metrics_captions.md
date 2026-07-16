# Embedding metric captions (Fig 2a)

Reference descriptions for metrics plotted in `Fig2_a_v8.ipynb`. All scores are computed on **10 stratified bootstrap subsamples** (10,000 cells each) unless noted otherwise. Neighbor graphs use **k = 15** and **cosine** distance unless stated otherwise. Cell-type labels use the `label` column; batch labels use the `batch` column.

---

## Biological conservation metrics

### Cluster–Label NMI (`NMI_cluster/label`)

**Short:** Normalized mutual information between Leiden clusters and annotated cell-type labels; higher values indicate stronger agreement.

**Long:** After building a k-nearest-neighbor graph on the embedding, Leiden clustering is run over a resolution grid and the resolution maximizing NMI with ground-truth labels (`label`) is selected. The reported score is the NMI between those optimal clusters and cell-type labels (range 0–1). It measures how well unsupervised structure in the embedding recovers known biology.

**Direction:** Higher is better.

---

### Cluster–Label ARI (`ARI_cluster/label`)

**Short:** Adjusted Rand index between Leiden clusters and cell-type labels; higher values indicate better cluster–label agreement beyond chance.

**Long:** Using the Leiden clustering at the NMI-optimized resolution, the adjusted Rand index (ARI) is computed between cluster assignments and annotated cell-type labels. Unlike NMI, ARI is adjusted for chance agreement and penalizes both over- and under-clustering. Higher scores indicate that embedding-derived clusters align more faithfully with ground-truth cell types.

**Direction:** Higher is better.

---

### Cell-Type Separation (`ASW_label`)

**Short:** Silhouette score measuring how cohesively cells of the same type group together in embedding space.

**Long:** The scIB label silhouette width (ASW) is computed on the embedding using Euclidean distance, measuring for each cell how much closer it is to same-type neighbors than to other types. The score is scaled to 0–1, where 1 indicates well-separated, compact cell-type groups and 0 indicates poor separation.

**Direction:** Higher is better.

---

### Graph Connectivity (`graph_conn`)

**Short:** Fraction of each cell type that remains connected in the kNN graph; higher values indicate preserved within-type structure.

**Long:** For each cell-type label, cells are subset and the k-nearest-neighbor connectivity graph is examined for connected components. The score for each type is the fraction of cells in the largest component, and the final metric is the mean across types (range 0–1). High connectivity means cells of the same type form contiguous neighborhoods rather than fragmenting into disconnected islands.

**Direction:** Higher is better.

---

### Overall Biology Score (`avg_bio`)

**Short:** Mean of Cluster–Label NMI, Cluster–Label ARI, and Cell-Type Separation; a composite biological conservation score.

**Long:** A simple aggregate of three core scIB biological conservation metrics: NMI at the optimized Leiden resolution, ARI at that same clustering, and label silhouette width (`ASW_label`). It summarizes how well the embedding jointly supports cluster–label agreement and cell-type separability, with equal weight given to each component.

**Direction:** Higher is better.

---

### Neighbor Label Purity (`knn_label_purity`)

**Short:** Mean fraction of each cell's k nearest neighbors sharing the same cell-type label.

**Long:** Using the precomputed kNN graph (k = 15), for every cell the fraction of neighbors with the same annotated label is computed, and these fractions are averaged across all cells (range 0–1). Higher values indicate that local neighborhoods are dominated by a single cell type, reflecting strong biological structure in the embedding.

**Direction:** Higher is better.

---

### kNN Label Accuracy (`kNN_label_acc`)

**Short:** Accuracy of a kNN classifier predicting cell-type labels from the embedding on a held-out test set.

**Long:** A distance-weighted k-nearest-neighbor classifier (k = 15) is trained on 80% of cells in embedding space and evaluated on a stratified 20% holdout set to predict cell-type labels. Accuracy is the fraction of correctly classified test cells (range 0–1). It provides a direct supervised readout of how informative the embedding is for cell-type identity.

**Direction:** Higher is better.

---

### kNN Label F1 (`kNN_label_f1_macro`)

**Short:** Macro-averaged F1 for kNN cell-type prediction; higher values indicate better label recovery across all types.

**Long:** Using the same kNN setup as kNN Label Accuracy (k = 15, distance-weighted, 80/20 stratified split), macro-averaged F1 is computed across cell types, giving equal weight to rare and abundant types. This is more informative than accuracy when labels are imbalanced. Higher scores indicate that the embedding preserves cell-type identity for all types, not just the majority.

**Direction:** Higher is better.

---

### Cell-Type Purity (`cLISI`)

**Short:** Scaled local cell-type homogeneity in kNN neighborhoods; higher values indicate cleaner label structure.

**Long:** Cell-type Local Inverse Simpson's Index (cLISI) is computed from the kNN graph using `scib-metrics`, measuring how homogeneous cell-type labels are in each cell's local neighborhood. With scaling enabled, the score is normalized to 0–1, where 1 indicates neighborhoods dominated by a single cell type (strong biological conservation) and lower values indicate more label mixing at local scales.

**Direction:** Higher is better.

---

## Batch mixing / integration metrics

### Batch Mixing ASW (`ASW_batch`)

**Short:** Global silhouette score with respect to batch labels; lower values indicate better batch mixing (weaker batch separation).

**Long:** The standard silhouette width is computed on the embedding using batch labels (`batch`) rather than cell types. Unlike `batch_ASW`, this is a single global score (not computed per cell type). Higher values mean batches are more separated in embedding space (stronger batch effect); lower values mean batches overlap more.

**Direction:** Lower is better.

---

### Biology–Batch ASW Ratio (`ASW_label/batch`)

**Short:** Identical to Batch Mixing Score in this pipeline; measures per–cell-type batch overlap in embedding space.

**Long:** Despite the name, this column stores the same value as `batch_ASW` — the scIB `silhouette_batch` score. For each cell type, the absolute silhouette width with respect to batch is computed and averaged across types, then scaled so that 1 indicates optimal batch overlap and 0 indicates poor mixing. Higher values indicate better technical batch integration while preserving cell-type structure.

**Direction:** Higher is better.

---

### Batch Mixing Score (`batch_ASW`)

**Short:** scIB batch ASW measuring per–cell-type batch overlap; higher values indicate better batch mixing.

**Long:** Computed via `scib.metrics.silhouette_batch`: for each cell type, the mean absolute silhouette width of cells with respect to batch labels is calculated, then averaged across types and scaled to 0–1 (1 = optimal batch overlap within each cell type). This is a standard scIB batch-integration metric that rewards embeddings where technical batches are intermingled within biological groups.

**Direction:** Higher is better.

---

### Batch Effect PCR (`PCR_batch`)

**Short:** Variance-weighted fraction of embedding PC variance explainable by batch; lower values indicate less batch effect (better correction).

**Long:** Principal Component Regression (PCR) fits a linear model predicting the top 50 embedding principal components from one-hot-encoded batch labels, then computes a variance-weighted sum of per-PC R² values (range 0–1). A high score means batch labels explain substantial variance in the embedding (strong residual batch effect); a low score means batch is poorly predictable from the representation (successful integration).

**Direction:** Lower is better.

---

### Batch Diversity (`iLISI`)

**Short:** Scaled local batch diversity in kNN neighborhoods; higher values indicate better batch mixing.

**Long:** Integration Local Inverse Simpson's Index (iLISI) measures how many distinct batch labels appear in each cell's k-nearest-neighbor neighborhood. Computed via `scib-metrics` with scaling to 0–1, where higher scores indicate that local neighborhoods contain more batch diversity — i.e., technical batches are well intermixed at the neighborhood level.

**Direction:** Higher is better.

---

### Batch Mixing (`kBET`)

**Short:** Fraction of cells passing a per–cell-type batch-equivalence test; higher values indicate better local batch mixing.

**Long:** k-nearest-neighbor Batch Effect Test (kBET) evaluates, for each cell within its cell type, whether the local batch composition in its neighborhood is statistically consistent with the global batch distribution (χ² test, α = 0.05). The score is the fraction of cells that pass this test (range 0–1). Higher acceptance rates indicate that batch proportions in local neighborhoods match the dataset-wide batch composition.

**Direction:** Higher is better.

---

### Neighbor Batch Purity (`knn_batch_purity`)

**Short:** Mean fraction of each cell's k nearest neighbors from the same batch; lower values indicate better batch mixing.

**Long:** Analogous to Neighbor Label Purity but computed on batch labels: for each cell, the fraction of k = 15 nearest neighbors sharing the same batch ID is calculated and averaged across cells (range 0–1). High batch purity means neighborhoods are batch-homogeneous (poor mixing); low purity means batches are interleaved locally.

**Direction:** Lower is better.

---

### Batch Predictability (`batch_pred_acc`)

**Short:** Accuracy of a kNN classifier predicting batch from the embedding; lower values indicate batch is harder to detect (better mixing).

**Long:** A distance-weighted kNN classifier (k = 15) is trained on 80% of cells and evaluated on a stratified 20% holdout to predict batch labels from the embedding. High accuracy means batch identity is easily recoverable from the representation (strong residual batch effect); low accuracy means the embedding obscures batch differences.

**Direction:** Lower is better.

---

## Quick reference: preferred direction

| Display name | Column | Higher better? |
|---|---|---|
| Cluster–Label NMI | `NMI_cluster/label` | Yes |
| Cluster–Label ARI | `ARI_cluster/label` | Yes |
| Cell-Type Separation | `ASW_label` | Yes |
| Graph Connectivity | `graph_conn` | Yes |
| Overall Biology Score | `avg_bio` | Yes |
| Neighbor Label Purity | `knn_label_purity` | Yes |
| kNN Label Accuracy | `kNN_label_acc` | Yes |
| kNN Label F1 | `kNN_label_f1_macro` | Yes |
| Cell-Type Purity | `cLISI` | Yes |
| Batch Mixing ASW | `ASW_batch` | **No** |
| Biology–Batch ASW Ratio | `ASW_label/batch` | Yes |
| Batch Mixing Score | `batch_ASW` | Yes |
| Batch Effect PCR | `PCR_batch` | **No** |
| Batch Diversity | `iLISI` | Yes |
| Batch Mixing | `kBET` | Yes |
| Neighbor Batch Purity | `knn_batch_purity` | **No** |
| Batch Predictability | `batch_pred_acc` | **No** |

**Note:** Fig 2a plots sort all models by descending median score within each group, regardless of metric direction. For metrics where lower is better (`ASW_batch`, `PCR_batch`, `knn_batch_purity`, `batch_pred_acc`), a model appearing further left does not necessarily indicate better performance.
