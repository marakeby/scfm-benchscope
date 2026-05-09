# -*- coding: utf-8 -*-
import os
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
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model: Optional[AttentionMIL] = None
        self.input_dim: Optional[int] = None
        # Filled during ``train()``: list of dicts with epoch, train_loss, val_loss (None if no val set)
        self.loss_history_: List[dict] = []

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
            temperature=self.temperature,
        ).to(self.device)

        self.loss_history_ = []

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

            val_loss_epoch: Optional[float] = None
            if adata_val is not None:
                val_loss_epoch = self._dataset_loss(adata_val)

            # Monitored metric (for early stopping)
            if self.monitor == "val_loss" and adata_val is not None:
                monitored = val_loss_epoch if val_loss_epoch is not None else train_loss
                mon_name = "val_loss"
            else:
                monitored = train_loss
                mon_name = "train_loss"

            self.loss_history_.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss_epoch) if val_loss_epoch is not None else None,
            })

            # Early stopping bookkeeping
            improved = is_better(monitored, best_score)
            if improved:
                best_score = monitored
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            val_disp = f"{val_loss_epoch:.4f}" if val_loss_epoch is not None else "n/a"
            print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_disp} | "
                  f"{mon_name}={monitored:.4f} | no_improve={epochs_no_improve}/{self.patience}")

            if epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}. Best {mon_name}={best_score:.4f}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def training_loss_history_dataframe(self) -> pd.DataFrame:
        """Per-epoch bag-averaged BCE-with-logits from the last ``train()`` run."""
        return pd.DataFrame(self.loss_history_)

    def plot_training_loss_curves(self, save_path: str) -> None:
        """
        Plot ``train_loss`` and ``val_loss`` (when present) vs epoch from ``loss_history_``.
        Also writes ``<save_path_without_ext>_history.csv`` next to the figure if ``save_path`` ends in ``.png``.
        """
        if not self.loss_history_:
            return
        df = self.training_loss_history_dataframe()
        if save_path.lower().endswith(".png"):
            csv_path = save_path[:-4] + "_history.csv"
        else:
            csv_path = save_path + "_history.csv"
        df.to_csv(csv_path, index=False)

        epochs = df["epoch"].to_numpy()
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, df["train_loss"], label="train_loss (BCE)", marker="o", markersize=2, linewidth=1.5)
        if df["val_loss"].notna().any():
            plt.plot(epochs, df["val_loss"], label="val_loss (BCE)", marker="o", markersize=2, linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MIL training / validation loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

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
    def visualize_attention(self, adata, save_path: Optional[str] = None):
        """
        Aggregates attention by cell type for each bag, then returns overall mean.
        Requires `celltype_key` to exist in adata.obs.
        If ``save_path`` is set, writes a PNG instead of calling ``plt.show()``.
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
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        return summary

    def collect_attention_frame(self, adata, top_k: int = 10):
        """
        Returns a long DataFrame with one row per cell/instance:
          columns: ['patient_id','outcome','cell_type','attn','score','contrib',
                    'rank_within_patient','bag_logit','bag_prob','bag_entropy','n_cells']
        where:
          - attn       = attention A_i from the same forward as ``evaluate()`` (temperature-scaled softmax).
          - bag_logit  = scalar logit from the full ``AttentionMIL`` forward (MLP head on pooled M).
          - bag_prob   = sigmoid(bag_logit).
          - score      = (d bag_logit / d x_i) · x_i  (input–gradient dot product per cell; local sensitivity).
          - contrib    = attn * score  (heuristic; NOT a linear decomposition of the MLP logit).
          - bag_entropy = -sum_i A_i log(A_i) / log(N)  (normalized: 0..1; 1 = very diffuse).
        """
        assert self.model is not None, "Train model first."
        self.model.eval()

        pids, bags, labels, metas = self.prepare_bags(adata, include_meta=True)
        rows = []

        for pid, x, y, meta in zip(pids, bags, labels, metas):
            x = x.to(self.device)
            xg = x.detach().clone().requires_grad_(True)
            out, A = self.model(xg)
            logit = out.squeeze()
            grad_x = torch.autograd.grad(
                logit, xg, retain_graph=False, create_graph=False
            )[0]

            A_1d = A.squeeze(1).detach()
            score = (grad_x * xg).sum(dim=1).detach()
            contrib = A_1d * score

            bag_logit = float(logit.detach())
            bag_prob = float(torch.sigmoid(logit.detach()))
            n = int(A_1d.shape[0])
            entropy = -(A_1d * (A_1d.clamp_min(1e-12)).log()).sum().item()
            norm_entropy = entropy / (math.log(n) if n > 1 else 1.0)

            order = torch.argsort(A_1d, descending=True).cpu().numpy()
            rank = np.empty_like(order)
            rank[order] = np.arange(1, n + 1)

            cell_type = meta if meta is not None else np.array(["NA"] * n)

            df = pd.DataFrame({
                "patient_id": pid,
                "outcome": float(y.item()),
                "cell_type": cell_type,
                "attn": A_1d.cpu().numpy(),
                "score": score.cpu().numpy(),
                "contrib": contrib.cpu().numpy(),
                "rank_within_patient": rank,
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

    def plot_attention_by_celltype(
        self, attn_df: pd.DataFrame, save_path_prefix: Optional[str] = None
    ):
        """
        Two plots:
          (1) Mean attention by cell_type (positives vs negatives).
          (2) Mean ``contrib`` (attn × input-gradient saliency) by cell_type — heuristic from
              ``collect_attention_frame``, not an exact logit decomposition.
        If ``save_path_prefix`` is set, writes ``{prefix}_attention.png`` and ``{prefix}_contrib.png``.
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
        if save_path_prefix:
            d = os.path.dirname(save_path_prefix) or "."
            os.makedirs(d, exist_ok=True)
            plt.savefig(f"{save_path_prefix}_attention.png", dpi=150, bbox_inches="tight")
            plt.close()
        else:
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
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_contrib.png", dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_entropy_vs_prob(
        self, attn_df: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Patient-level scatter: attention entropy (diffuseness) vs bag probability.
        Often, underperformance comes from very *diffuse* attention (high entropy).
        If ``save_path`` is set, writes a PNG instead of ``plt.show()``.
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
        if save_path:
            d = os.path.dirname(save_path) or "."
            os.makedirs(d, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return per_patient

    def export_attention_debug(self, adata, out_dir: str, top_k: int = 20) -> Optional[pd.DataFrame]:
        """
        Save attention diagnostics under ``out_dir`` (non-interactive / batch runs):
          - ``mil_attn_mean_by_celltype.png`` — from ``visualize_attention``
          - ``mil_attention_per_cell.csv`` — from ``collect_attention_frame``
          - ``mil_celltype_by_outcome_{attention,contrib}.png`` — from ``plot_attention_by_celltype``
          - ``mil_entropy_vs_prob.png`` — from ``plot_entropy_vs_prob``
          - ``mil_top_celltype_per_patient.csv`` — from ``topk_table``
        Returns the long per-cell attention frame, or None if cell types are missing for the mean plot only
        (CSV and entropy plot are still written when possible).
        """
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, "mil_celltype_by_outcome")
        self.visualize_attention(adata, save_path=os.path.join(out_dir, "mil_attn_mean_by_celltype.png"))

        attn_df = self.collect_attention_frame(adata, top_k=top_k)
        attn_df.to_csv(os.path.join(out_dir, "mil_attention_per_cell.csv"), index=False)
        self.plot_attention_by_celltype(attn_df, save_path_prefix=prefix)
        self.plot_entropy_vs_prob(attn_df, save_path=os.path.join(out_dir, "mil_entropy_vs_prob.png"))

        top_tbl = self.topk_table(attn_df, k=top_k)
        top_tbl.to_csv(os.path.join(out_dir, "mil_top_celltype_per_patient.csv"), index=False)
        return attn_df


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

