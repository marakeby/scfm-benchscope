# Classification metric captions (Fig 4)

Reference descriptions for metrics plotted in the classification analysis notebooks (e.g. `Fig4_a_v8.ipynb`, `Fig4_C_aggregation_strategy_v8.ipynb`, `Fig4_B_finetune_vs_frozen_v8.ipynb`). Metrics are computed via `eval_classifier` in `scfm_cancer_eval/evaluation/eval.py` on **patient-level predictions** from **5-fold cross-validation** splits (predefined per task, patient-held-out folds). The default classifier head is a **Random Forest** (`n_estimators = 100`, `max_depth = 5`, `random_state = 42`). Results are aggregated in `scFM_eval/results/classification.metrics.csv`.

**Seven shared classification tasks:** Treatment Naive vs Anti PD1; Treatment Naive vs Neoadjuvant Chemo; Treatment Naive vs TKI treated; T-cell exhaustion; ER+ vs TNBC; IO Response; MMRd vs MMRp.

---

## Patient-level aggregation strategies

Before metrics are computed, cell-level embeddings are aggregated to patient/sample labels using one of three strategies (`ClassifierPipeline` in `classify.py`). Each strategy produces one patient-level prediction and score per held-out fold.

### Vote (`vote`)

**Short:** Cell-level Random Forest predictions aggregated to patients by majority vote (label) and mean probability (score).

**Long:** A Random Forest is trained on **cell-level** embeddings in the training fold. Each cell receives a predicted label and class probability. For each patient in the test fold, the patient label is the **majority vote** across cells and the patient score is the **mean** of cell-level positive-class probabilities. Metrics are then computed on these patient-level predictions. This strategy uses all cells but treats them as independent training examples.

**Direction:** N/A (aggregation method).

---

### Avg (`avg`)

**Short:** Mean cell embedding per patient, then Random Forest trained and evaluated at patient level.

**Long:** For each patient, cell embeddings in the fold are averaged to a single **mean embedding vector**. A Random Forest is fit on these patient-level vectors in the training fold and evaluated on mean embeddings of held-out patients. This is a direct patient-level classifier with a simple, interpretable aggregation.

**Direction:** N/A (aggregation method).

---

### MIL (`MIL`)

**Short:** Multi-instance learning with attention over each patient's cell bag; patient label predicted from weighted cell evidence.

**Long:** A dedicated **multi-instance learning (MIL)** model (`MILExperiment`) treats each patient as a bag of cell embeddings. An attention mechanism learns to weight informative cells when producing a patient-level prediction and score. Unlike Vote and Avg, MIL explicitly models which cells drive the patient label. This is the primary strategy used for cross-model comparisons in Fig 4 when finetuned runs are included (finetuned models export MIL metrics only).

**Direction:** N/A (aggregation method).

---

## Classification performance metrics

All metrics below are computed on **held-out test patients** within each CV fold. For **binary** tasks, scores use the positive-class probability (`pred_score`). For **multiclass** tasks (if present), macro-averaged one-vs-rest formulations are used. All metrics range from 0 to 1 unless noted. **Higher is better** for every metric listed.

---

### AUPRC (`AUPRC`)

**Short:** Area under the precision–recall curve; primary metric for imbalanced patient-level classification.

**Long:** Average precision (area under the precision–recall curve) summarises the trade-off between precision and recall across classification thresholds. For binary tasks it is computed from patient-level positive-class scores against true labels; for multiclass tasks a **macro-averaged** average precision is used. AUPRC is emphasised in this benchmark because it is more informative than AUROC when positive patients are rare. It is the default metric for heatmaps and model ordering in Fig 4.

**Direction:** Higher is better.

---

### AUC (`AUC`)

**Short:** Area under the ROC curve; measures rank discrimination between classes across thresholds.

**Long:** For binary tasks, AUROC is computed from the ROC curve of patient-level scores. For multiclass tasks, **macro-averaged one-vs-rest** AUROC is used. AUC measures how well the model ranks positive patients above negative ones, independent of a single classification threshold. It can be optimistic under class imbalance relative to AUPRC.

**Direction:** Higher is better.

---

### F1 (`F1`)

**Short:** Macro-averaged F1 score at the default classification threshold; balances precision and recall equally across classes.

**Long:** Predicted patient labels (argmax / thresholded scores) are compared to ground truth and F1 is computed with **macro averaging**, giving equal weight to each class. `zero_division = 0` is used so undefined F1 for absent classes does not break aggregation. Macro F1 is useful when all classes matter equally, including minority groups.

**Direction:** Higher is better.

---

### Precision (`Precision`)

**Short:** Macro-averaged precision of predicted patient labels; fraction of positive calls that are correct, averaged across classes.

**Long:** At the default predicted label threshold, precision is computed per class and then **macro-averaged** across classes. High precision means the model's positive predictions are reliable (few false positives), which matters when the cost of incorrect positive calls is high.

**Direction:** Higher is better.

---

### Recall (`Recall`)

**Short:** Macro-averaged recall of predicted patient labels; fraction of true positives recovered, averaged across classes.

**Long:** At the default predicted label threshold, recall (sensitivity) is computed per class and **macro-averaged**. High recall means the model captures most true positive patients, which matters when missing positives is costly (e.g. treatment response prediction).

**Direction:** Higher is better.

---

### Accuracy (`Accuracy`)

**Short:** Fraction of patients correctly classified at the default threshold.

**Long:** Standard classification accuracy: the proportion of held-out patients whose predicted label exactly matches the true label. Unlike macro F1 / precision / recall, accuracy weights classes by their frequency in the test fold. It is easy to interpret but can be misleading when classes are imbalanced.

**Direction:** Higher is better.

---

## Quick reference

| Display / column | Category | Higher better? | Evaluated on |
|---|---|---|---|
| AUPRC | Ranking / threshold-free | Yes | Patient-level test fold |
| AUC | Ranking / threshold-free | Yes | Patient-level test fold |
| F1 | Thresholded classification | Yes | Patient-level test fold |
| Precision | Thresholded classification | Yes | Patient-level test fold |
| Recall | Thresholded classification | Yes | Patient-level test fold |
| Accuracy | Thresholded classification | Yes | Patient-level test fold |

| Strategy | Column value | Aggregation |
|---|---|---|
| Vote | `vote` | Majority vote + mean cell probability |
| Avg | `avg` | Mean embedding per patient |
| MIL | `MIL` | Attention-based multi-instance learning |

**Note:** Fig 4a heatmaps report the **mean metric across the seven tasks** per model (and per strategy where applicable). Models are grouped as Baseline, Geneformer, Other, and scGPT, and ordered within group by descending mean performance. Individual fold values are stored in per-run `*_cv_metrics.csv` files under each experiment's `cv/` directory.
