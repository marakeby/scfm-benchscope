# -*- coding: utf-8 -*-
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 

import torch
import torch.nn as nn


from sklearn.metrics import roc_auc_score, average_precision_score




class AttentionMIL(nn.Module):
    """
     attention-based MIL with regularization
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.25,
        attention_dim: int = 64,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Deeper feature extractor with normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_dim, 1)
        )
        
        # Deeper classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor):
        """
        x: [N, D] - bag of N instances with D features
        Returns:
          out: [1] bag-level logit
          A:   [N, 1] attention weights (sum to 1)
        """
        # Extract features
        H = self.feature_extractor(x)  # [N, hidden_dim]
        
        # Compute attention with temperature scaling
        A_logits = self.attention(H)   # [N, 1]
        A = torch.softmax(A_logits / self.temperature, dim=0)
        
        # Weighted aggregation
        M = torch.sum(A * H, dim=0)    # [hidden_dim]
        
        # Classification
        out = self.classifier(M)        # [1]
        
        return out, A


#  training configuration recommendations
# RECOMMENDED_CONFIG = {
#     # "hidden_dim": 128,
#     # "attention_dim": 64,
#     "epochs": 500,  # Much longer training
#     "lr": 1e-4,     # Lower learning rate
#     "weight_decay": 1e-5,  # L2 regularization
#     "dropout": 0.3,
#     "patience": 50,  # More patience with longer training
#     "min_delta": 0.001,  # Require meaningful improvement
#     "temperature": 1.0,  # Can tune between 0.5-2.0
# }

class MILExperiment:
    def __init__(
        self,
        embedding_col: str,
        label_key: str = "outcome",
        patient_key: str = "patient_id",
        celltype_key: Optional[str] = "cell_type",
        # **RECOMMENDED_CONFIG

        hidden_dim: int = 128,
        attention_dim = 64,

        epochs: int = 500,
        
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.3,
        seed: int = 42,
        pos_weight: Optional[float] = None,  # handle class imbalance (>1 favors positives)

        # Early stopping knobs
        patience: int = 50,
        temperature =  1.0,
        min_delta: float = 0.001,
        monitor: str = "val_loss",           # "val_loss" or "train_loss"
        mode: str = "min",                   # "min" for loss, ("max" if you later monitor AUC)
        device: Optional[str] = None
    ):
        self.embedding_col = embedding_col
        self.label_key = label_key
        self.patient_key = patient_key
        self.celltype_key = celltype_key

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pos_weight = pos_weight

        # early stopping config
        self.patience = int(patience)
        self.temperature = temperature
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode.lower()
        assert self.mode in {"min", "max"}, "mode must be 'min' or 'max'"

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model: Optional[AttentionMIL] = None
        self.input_dim: Optional[int] = None

    # -----------------------
    # Data utilities
    # -----------------------
    def _to_numpy(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

    def prepare_bags(self, adata, include_meta: bool = True):
        """
        Returns:
          pids:   List[str]
          bags:   List[torch.FloatTensor [N_i, D]]
          labels: List[torch.FloatTensor [1]]
          metas:  List[np.ndarray of length N_i] or None
        """
        self.input_dim = adata.obsm[self.embedding_col].shape[1]
        bags, labels, metas, pids = [], [], [], []
        for pid in adata.obs[self.patient_key].unique():
            # adata_p = adata[adata.obs[self.patient_key] == pid]
            # NEW (robust to AnnData index dtype):
            mask = (adata.obs[self.patient_key].to_numpy() == pid)
            adata_p = adata[mask].copy()
            X = self._to_numpy(adata_p.obsm[self.embedding_col])  # [N_i, D]
            y = float(adata_p.obs[self.label_key].iloc[0])

            bags.append(torch.tensor(X, dtype=torch.float32))
            labels.append(torch.tensor([y], dtype=torch.float32))
            pids.append(pid)

            if include_meta and (self.celltype_key is not None) and (self.celltype_key in adata_p.obs.columns):
                metas.append(adata_p.obs[self.celltype_key].to_numpy())
            else:
                metas.append(None)

        return pids, bags, labels, metas

    # -----------------------
    # Loss & metrics
    # -----------------------
    def _make_criterion(self):
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()

    def _dataset_loss(self, adata):
        """Average BCE-with-logits bag loss over a dataset."""
        _, bags, labels, _ = self.prepare_bags(adata, include_meta=False)
        self.model.eval()
        criterion = self._make_criterion()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in zip(bags, labels):
                x = x.to(self.device); y = y.to(self.device)
                out, _ = self.model(x)
                # loss = criterion(out.squeeze(), y.squeeze())
                loss = criterion(out.view(1), y.view(1))
                total += loss.item(); n += 1
        return total / max(n, 1)

    # -----------------------
    # Training with Early Stopping
    # -----------------------
    def train(self, adata_train, adata_val=None):
        """
        Train bag-by-bag. If adata_val is provided and monitor='val_loss', early stopping
        will watch validation loss; otherwise it will monitor train loss.
        """
        _, bags_train, labels_train, _ = self.prepare_bags(adata_train, include_meta=False)
        #self.model = AttentionMIL(
        #    input_dim=self.input_dim,
        #   hidden_dim=self.hidden_dim,
        #    dropout=self.dropout
        #.to(self.device)

        self.model = AttentionMIL(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            attention_dim=self.attention_dim,
            temperature=1.0
        ).to(self.device)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = self._make_criterion()

        # Early stopping state
        best_score = None
        best_state = None
        epochs_no_improve = 0

        def is_better(curr, best):
            if best is None:
                return True
            if self.mode == "min":
                return (best - curr) > self.min_delta
            else:  # "max"
                return (curr - best) > self.min_delta

        for epoch in range(self.epochs):
            self.model.train()
            idx = np.random.permutation(len(bags_train))
            total_loss = 0.0

            for i in idx:
                x = bags_train[i].to(self.device)
                y = labels_train[i].to(self.device)

                optimizer.zero_grad()
                out, _ = self.model(x)
                # loss = criterion(out.squeeze(), y.squeeze())
                loss = criterion(out.view(1), y.view(1))
                loss.backward()
                # Optional stability:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(bags_train)

            # Monitored metric
            if self.monitor == "val_loss" and adata_val is not None:
                monitored = self._dataset_loss(adata_val)
                mon_name = "val_loss"
            else:
                monitored = train_loss
                mon_name = "train_loss"

            # Early stopping bookkeeping
            improved = is_better(monitored, best_score)
            if improved:
                best_score = monitored
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | {mon_name}={monitored:.4f} "
                  f"| no_improve={epochs_no_improve}/{self.patience}")

            if epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}. Best {mon_name}={best_score:.4f}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    # -----------------------
    # Evaluation
    # -----------------------
    def evaluate(self, adata_test, threshold: float = 0.5, return_attention: bool = False):
        """
        Returns:
          pids, y_true, preds, pred_scores
          (optionally) attn_per_bag: list of np.ndarray (N_i,) attention weights
        """
        assert self.model is not None, "Call train() before evaluate()."
        pids, bags_test, labels_test, _ = self.prepare_bags(adata_test, include_meta=False)

        self.model.eval()
        pred_scores, attn_out = [], []
        with torch.no_grad():
            for x in bags_test:
                x = x.to(self.device)
                out, A = self.model(x)
                pred_scores.append(torch.sigmoid(out.squeeze()).item())
                if return_attention:
                    attn_out.append(A.squeeze(1).detach().cpu().numpy())

        y_true = [float(l.squeeze().item()) for l in labels_test]
        pred_scores = np.array(pred_scores)
        preds = (pred_scores >= threshold).astype(int).tolist()

        # Optional metrics
        metrics = {}
        # if _HAVE_SK:
        if True:
            # Only compute if both classes present
            if len(set(y_true)) > 1:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, pred_scores))
                except Exception:
                    metrics["roc_auc"] = None
                try:
                    metrics["avg_precision"] = float(average_precision_score(y_true, pred_scores))
                except Exception:
                    metrics["avg_precision"] = None
            else:
                metrics["roc_auc"] = None
                metrics["avg_precision"] = None

        if return_attention:
            return pids, y_true, preds, pred_scores, metrics, attn_out
        return pids, y_true, preds, pred_scores, metrics

    # -----------------------
    # Attention visualization
    # -----------------------
    def visualize_attention(self, adata):
        """
        Aggregates attention by cell type for each bag, then returns overall mean.
        Requires `celltype_key` to exist in adata.obs.
        """
        assert self.model is not None, "Train the model before visualizing attention."
        pids, bags, _, metas = self.prepare_bags(adata, include_meta=True)

        rows = []
        self.model.eval()
        with torch.no_grad():
            for x, meta in zip(bags, metas):
                if meta is None:
                    continue
                x = x.to(self.device)
                _, A = self.model(x)               # [N, 1]
                A = A.squeeze(1).cpu().numpy()     # [N]
                rows.append(pd.DataFrame({"attention": A, "cell_type": meta}))

        if not rows:
            print("No cell_type metadata found; cannot visualize attention.")
            return None

        combined = pd.concat(rows, ignore_index=True)
        summary = combined.groupby("cell_type", as_index=True)["attention"].mean().sort_values(ascending=False)

        ax = summary.plot(kind="bar", title="Average Attention by Cell Type")
        ax.set_ylabel("Average Attention Weight")
        plt.tight_layout()
        plt.show()
        return summary
    import math

    def _softmax0(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,1] -> softmax over instances (dim=0)
        return torch.softmax(x, dim=0)

    @torch.no_grad()
    def collect_attention_frame(self, adata, top_k: int = 10):
        """
        Returns a long DataFrame with one row per cell/instance:
          columns: ['patient_id','outcome','cell_type','attn','score','contrib',
                    'rank_within_patient','bag_logit','bag_prob','bag_entropy','n_cells']
        where:
          - attn      = attention weight A_i (sum to 1 per patient)
          - score     = w_cls · H_i  (cell's latent score before attention)
          - contrib   = attn * score (cell's contribution to bag logit, up to +bias)
          - bag_logit = sum_i contrib + bias
          - bag_prob  = sigmoid(bag_logit)
          - bag_entropy = -sum_i A_i log(A_i) / log(N)  (normalized: 0..1; 1 = very diffuse)
        """
        assert self.model is not None, "Train model first."
        self.model.eval()

        pids, bags, labels, metas = self.prepare_bags(adata, include_meta=True)
        rows = []

        W = self.model.classifier.weight.squeeze(0)      # [H]
        b = float(self.model.classifier.bias.squeeze())  # scalar

        for pid, x, y, meta in zip(pids, bags, labels, metas):
            x = x.to(self.device)                        # [N,D]
            # reuse model parts for efficiency
            H = self.model.feature_extractor(x)          # [N,H]
            A_logits = self.model.attention(H)           # [N,1]
            A = self._softmax0(A_logits).squeeze(1)      # [N]
            score = (H * W).sum(dim=1)                   # [N]  = H @ W
            contrib = A * score                          # [N]

            bag_logit = contrib.sum().item() + b
            bag_prob  = torch.sigmoid(torch.tensor(bag_logit)).item()
            n = A.shape[0]
            # normalized entropy: 0 (peaky) ... 1 (uniform)
            entropy = -(A * (A.clamp_min(1e-12)).log()).sum().item()
            norm_entropy = entropy / (math.log(n) if n > 1 else 1.0)

            # rank cells by attention (high->low)
            order = torch.argsort(A, descending=True).cpu().numpy()
            rank = np.empty_like(order)
            rank[order] = np.arange(1, n + 1)

            # prepare columns
            cell_type = meta if meta is not None else np.array(["NA"] * n)

            df = pd.DataFrame({
                "patient_id": pid,
                "outcome": float(y.item()),
                "cell_type": cell_type,
                "attn": A.detach().cpu().numpy(),
                "score": score.detach().cpu().numpy(),
                "contrib": contrib.detach().cpu().numpy(),
                "rank_within_patient": rank
            })
            df["bag_logit"] = bag_logit
            df["bag_prob"] = bag_prob
            df["bag_entropy"] = norm_entropy
            df["n_cells"] = n
            rows.append(df)

        full = pd.concat(rows, ignore_index=True)
        # helper flag for quick inspection of the top-k attention cells
        full["is_topk"] = full["rank_within_patient"] <= top_k
        return full


    def plot_attention_by_celltype(self, attn_df: pd.DataFrame):
        """
        Two plots:
          (1) Mean attention by cell_type (positives vs negatives).
          (2) Mean *contribution* (attn * score) by cell_type (positives vs negatives).
        """
        # aggregate
        g_attn = (attn_df
                  .groupby(["cell_type", "outcome"], as_index=False)["attn"]
                  .mean()
                  .sort_values(["outcome", "attn"], ascending=[True, False]))
        g_contrib = (attn_df
                     .groupby(["cell_type", "outcome"], as_index=False)["contrib"]
                     .mean()
                     .sort_values(["outcome", "contrib"], ascending=[True, False]))

        # pivot for side-by-side bars
        piv_a = g_attn.pivot(index="cell_type", columns="outcome", values="attn").fillna(0.0)
        piv_c = g_contrib.pivot(index="cell_type", columns="outcome", values="contrib").fillna(0.0)

        # Plot 1: attention
        plt.figure(figsize=(10, 4))
        for j, col in enumerate(sorted(piv_a.columns)):
            plt.bar(np.arange(len(piv_a)) + j*0.4, piv_a[col].values, width=0.4, label=f"outcome={int(col)}")
        plt.xticks(np.arange(len(piv_a)) + 0.2, piv_a.index, rotation=45, ha="right")
        plt.ylabel("Mean attention")
        plt.title("Mean Attention by Cell Type (split by outcome)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: contribution
        plt.figure(figsize=(10, 4))
        for j, col in enumerate(sorted(piv_c.columns)):
            plt.bar(np.arange(len(piv_c)) + j*0.4, piv_c[col].values, width=0.4, label=f"outcome={int(col)}")
        plt.xticks(np.arange(len(piv_c)) + 0.2, piv_c.index, rotation=45, ha="right")
        plt.ylabel("Mean contribution (attn * score)")
        plt.title("Mean Contribution by Cell Type (split by outcome)")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_entropy_vs_prob(self, attn_df: pd.DataFrame):
        """
        Patient-level scatter: attention entropy (diffuseness) vs bag probability.
        Often, underperformance comes from very *diffuse* attention (high entropy).
        """
        per_patient = (attn_df
                       .groupby(["patient_id", "outcome"], as_index=False)
                       .agg({
                           "bag_prob": "first",
                           "bag_entropy": "first",
                           "n_cells": "first"
                       }))

        # scatter
        plt.figure(figsize=(6, 5))
        for outc in sorted(per_patient["outcome"].unique()):
            sub = per_patient[per_patient["outcome"] == outc]
            plt.scatter(sub["bag_entropy"], sub["bag_prob"], s=16, alpha=0.8, label=f"outcome={int(outc)}")
        plt.xlabel("Attention entropy (0=peaky, 1=diffuse)")
        plt.ylabel("Bag probability (sigmoid)")
        plt.title("Entropy vs Bag Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return per_patient


    def topk_table(self, attn_df: pd.DataFrame, k: int = 5):
        """
        Returns a compact table of top-k attention cell types per patient with their mean attention and contribution.
        Useful to see if the model is attending to plausible biology or to noise.
        """
        topk = attn_df[attn_df["is_topk"]].copy()
        # summarize within patient x cell_type
        agg = (topk
               .groupby(["patient_id", "outcome", "cell_type"], as_index=False)
               .agg(n_topk=("attn", "size"),
                    mean_attn=("attn", "mean"),
                    mean_contrib=("contrib", "mean")))
        # pick best cell_type per patient
        best = agg.sort_values(["patient_id", "mean_contrib"], ascending=[True, False]) \
                  .groupby("patient_id", as_index=False).head(1)
        return best.sort_values(["outcome", "mean_contrib"], ascending=[True, False])

