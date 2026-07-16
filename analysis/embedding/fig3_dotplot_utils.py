"""Publication-quality dot plot helpers for Fig 3c TME marker figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

PUBLICATION_DPI = 450

# Sized for multi-panel figures: labels stay legible after ~50% downscaling.
PANEL_FONT_SIZES = {
    "tick": 14,
    "axis_label": 15,
    "gene_group": 14,
    "legend_title": 13,
    "legend_tick": 12,
}

PUBLICATION_RC = {
    "figure.dpi": 100,
    "savefig.dpi": PUBLICATION_DPI,
    "font.size": PANEL_FONT_SIZES["tick"],
    "axes.titlesize": PANEL_FONT_SIZES["axis_label"] + 1,
    "axes.labelsize": PANEL_FONT_SIZES["axis_label"],
    "xtick.labelsize": PANEL_FONT_SIZES["tick"],
    "ytick.labelsize": PANEL_FONT_SIZES["tick"],
    "legend.fontsize": PANEL_FONT_SIZES["legend_tick"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

TME_MARKER_DICT: dict[str, list[str]] = {
    "B cells": ["MS4A1", "CD19", "CD79A"],
    "Cancer cells": ["EPCAM", "KRT19", "KRT8", "KRT18"],
    "Endothelial cells": ["PECAM1", "VWF", "CD34"],
    "Fibroblasts": ["FAP", "PDPN", "ACTA2", "COL1A1"],
    "Mast cells": ["TPSAB1", "KIT", "CPA3"],
    "Myeloid cells": ["LYZ", "CD14", "CD68", "CD163", "FCGR3A"],
    "T cells": ["CD3D", "CD3E", "CD2"],
    "pDC": ["CLEC4C", "LILRA4", "GZMB"],
}

DEFAULT_DOTPLOT_KWARGS = {
    "standard_scale": "var",
    "dot_max": 0.7,
    "color_map": "Reds",
    "dendrogram": False,
    "largest_dot": 260.0,
}

# Keys handled by make_publication_dotplot itself — never pass to Scanpy/matplotlib.
_NON_SCANPY_KEYS = frozenset(
    {
        "model_label",
        "metrics_dir",
        "save_tables",
        "leiden_resolution",
        "marker_dict",
        "groupby",
        "figsize",
        "show",
        "output_stem",
    }
)


def apply_publication_style() -> None:
    plt.rcParams.update(PUBLICATION_RC)


def filter_markers(
    adata: ad.AnnData,
    marker_dict: Mapping[str, list[str]] = TME_MARKER_DICT,
) -> tuple[dict[str, list[str]], list[str]]:
    """Return marker groups present in ``adata`` and the list of missing genes."""
    all_markers = {gene for genes in marker_dict.values() for gene in genes}
    present_markers = {gene for gene in all_markers if gene in adata.var_names}
    missing_markers = sorted(all_markers - present_markers)

    filtered_marker_dict = {
        cell_type: [gene for gene in genes if gene in present_markers]
        for cell_type, genes in marker_dict.items()
        if any(gene in present_markers for gene in genes)
    }
    return filtered_marker_dict, missing_markers


def cluster_from_embedding(
    adata: ad.AnnData,
    embedding_key: str,
    *,
    cluster_key: str = "cluster",
    n_neighbors: int = 15,
    resolution: float = 0.3,
    random_state: int = 42,
) -> None:
    """Build a neighbors graph and Leiden clusters from an embedding matrix."""
    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=0,
    )
    sc.tl.leiden(
        adata,
        key_added=cluster_key,
        random_state=random_state,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )


def transfer_clusters(
    source: ad.AnnData,
    target: ad.AnnData,
    *,
    cluster_key: str = "cluster",
) -> None:
    """Copy clustering metadata from ``source`` onto ``target``."""
    target.uns = source.uns
    target.obs[cluster_key] = source.obs[cluster_key]


def _dotplot_figsize(marker_dict: Mapping[str, list[str]], n_clusters: int) -> tuple[float, float]:
    n_genes = sum(len(genes) for genes in marker_dict.values())
    width = max(10.0, n_genes * 0.45 + 3.0)
    height = max(6.5, n_clusters * 0.50 + 2.5)
    return (width, height)


def _style_dotplot_for_panel(dotplot) -> None:
    """Override Scanpy's relative font sizes with explicit panel-ready values."""
    axes = dotplot.get_axes()

    main_ax = axes.get("mainplot_ax")
    if main_ax is not None:
        main_ax.set_xlabel(main_ax.get_xlabel(), fontsize=PANEL_FONT_SIZES["axis_label"])
        main_ax.set_ylabel(main_ax.get_ylabel(), fontsize=PANEL_FONT_SIZES["axis_label"])
        main_ax.tick_params(axis="both", labelsize=PANEL_FONT_SIZES["tick"])
        for label in main_ax.get_xticklabels() + main_ax.get_yticklabels():
            label.set_fontsize(PANEL_FONT_SIZES["tick"])

    gene_group_ax = axes.get("gene_group_ax")
    if gene_group_ax is not None:
        for text in gene_group_ax.texts:
            text.set_fontsize(PANEL_FONT_SIZES["gene_group"])
        gene_group_ax.tick_params(labelsize=PANEL_FONT_SIZES["gene_group"])

    size_legend_ax = axes.get("size_legend_ax")
    if size_legend_ax is not None:
        size_legend_ax.set_title(
            size_legend_ax.get_title(),
            fontsize=PANEL_FONT_SIZES["legend_title"],
        )
        size_legend_ax.tick_params(labelsize=PANEL_FONT_SIZES["legend_tick"])
        for label in size_legend_ax.get_xticklabels():
            label.set_fontsize(PANEL_FONT_SIZES["legend_tick"])

    color_legend_ax = axes.get("color_legend_ax")
    if color_legend_ax is not None:
        color_legend_ax.set_title(
            color_legend_ax.get_title(),
            fontsize=PANEL_FONT_SIZES["legend_title"],
        )
        color_legend_ax.tick_params(labelsize=PANEL_FONT_SIZES["legend_tick"])
        for label in color_legend_ax.get_xticklabels():
            label.set_fontsize(PANEL_FONT_SIZES["legend_tick"])


def prepare_expression(adata: ad.AnnData, *, target_sum: float = 1e4) -> None:
    """Normalize counts and log-transform in place."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def _gene_to_cell_type(marker_dict: Mapping[str, list[str]]) -> dict[str, str]:
    return {
        gene: cell_type
        for cell_type, genes in marker_dict.items()
        for gene in genes
    }


def compute_dotplot_tables(
    adata: ad.AnnData,
    marker_dict: Mapping[str, list[str]] = TME_MARKER_DICT,
    *,
    groupby: str = "cluster",
    expression_cutoff: float = 0.0,
    standard_scale: str = "var",
) -> dict[str, pd.DataFrame]:
    """Compute the expression statistics underlying the dot plot."""
    filtered_marker_dict, _ = filter_markers(adata, marker_dict)
    genes = [gene for genes in filtered_marker_dict.values() for gene in genes]
    gene_to_cell_type = _gene_to_cell_type(filtered_marker_dict)

    plot_adata = adata.copy()
    prepare_expression(plot_adata)

    cluster_labels = plot_adata.obs[groupby].astype(str).values
    expr = plot_adata[:, genes].to_df()
    expr[groupby] = cluster_labels

    expressed = expr[genes].gt(expression_cutoff)
    expressed[groupby] = cluster_labels

    mean_expression = expr.groupby(groupby, observed=True)[genes].mean()
    fraction_expressing = expressed.groupby(groupby, observed=True)[genes].mean()
    mean_expression.index.name = groupby
    fraction_expressing.index.name = groupby

    if standard_scale == "var":
        scaled_mean = mean_expression.sub(mean_expression.min(axis=0), axis=1).div(
            mean_expression.max(axis=0), axis=1
        ).fillna(0.0)
    elif standard_scale == "group":
        scaled_mean = mean_expression.sub(mean_expression.min(axis=1), axis=0).div(
            mean_expression.max(axis=1), axis=0
        ).fillna(0.0)
    else:
        scaled_mean = mean_expression.copy()

    long_df = mean_expression.stack(future_stack=True).rename("mean_expression").reset_index()
    if "level_1" in long_df.columns:
        long_df = long_df.rename(columns={"level_1": "gene"})
    long_df["fraction_expressing"] = fraction_expressing.stack().values
    long_df["scaled_mean_expression"] = scaled_mean.stack().values
    long_df["cell_type"] = long_df["gene"].map(gene_to_cell_type)
    long_df = long_df.sort_values([groupby, "cell_type", "gene"]).reset_index(drop=True)

    cluster_sizes = (
        plot_adata.obs[groupby]
        .astype(str)
        .value_counts()
        .rename_axis(groupby)
        .reset_index(name="n_cells")
        .sort_values(groupby)
    )

    enrichment = (
        long_df.groupby([groupby, "cell_type"], observed=True)
        .agg(
            mean_expression=("mean_expression", "mean"),
            mean_fraction_expressing=("fraction_expressing", "mean"),
            n_markers=("gene", "nunique"),
        )
        .reset_index()
    )
    max_fraction = enrichment["mean_fraction_expressing"].max() or 1.0
    max_expression = enrichment["mean_expression"].max() or 1.0
    enrichment["enrichment_score"] = (
        0.65 * (enrichment["mean_fraction_expressing"] / max_fraction)
        + 0.35 * (enrichment["mean_expression"] / max_expression)
    )
    best_idx = enrichment.groupby("cell_type", observed=True)["enrichment_score"].idxmax()
    cell_type_best = enrichment.loc[best_idx].sort_values("cell_type").reset_index(drop=True)

    return {
        "values": long_df,
        "cluster_sizes": cluster_sizes,
        "enrichment": enrichment,
        "cell_type_best_clusters": cell_type_best,
    }


def build_figure_caption(
    model_label: str,
    tables: Mapping[str, pd.DataFrame],
    *,
    groupby: str = "cluster",
    n_cells: int | None = None,
    resolution: float = 0.3,
    dataset_label: str = "pre-treatment BRCA tumor cells",
) -> str:
    """Draft a publication-ready figure caption from dot-plot statistics."""
    cluster_sizes = tables["cluster_sizes"]
    best = tables["cell_type_best_clusters"]

    n_clusters = len(cluster_sizes)
    if n_cells is None:
        n_cells = int(cluster_sizes["n_cells"].sum())

    top_examples = best.nlargest(min(3, len(best)), "enrichment_score")
    examples = [
        f"{row['cell_type']} markers in cluster {row[groupby]} "
        f"({100 * row['mean_fraction_expressing']:.0f}% expressing cells)"
        for _, row in top_examples.iterrows()
    ]
    enrichment_text = ", ".join(examples)

    return (
        f"Dot plot of canonical tumor microenvironment (TME) marker genes across "
        f"Leiden clusters (resolution {resolution:g}) derived from {model_label} "
        f"embeddings in {dataset_label} (n={n_cells:,} cells, {n_clusters} clusters). "
        f"Dot size indicates the fraction of cells with detectable expression; "
        f"color indicates mean log-normalized expression (scaled per gene). "
        f"Embedding-derived clusters recovered expected marker structure, with the "
        f"strongest enrichments including {enrichment_text}. "
        f"Per-cluster marker statistics are provided in the accompanying tables."
    )


def save_dotplot_tables(
    tables: Mapping[str, pd.DataFrame],
    output_stem: str | Path,
) -> dict[str, Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "values": output_stem.with_name(f"{output_stem.name}_values.csv"),
        "cluster_sizes": output_stem.with_name(f"{output_stem.name}_cluster_sizes.csv"),
        "enrichment": output_stem.with_name(f"{output_stem.name}_enrichment.csv"),
        "cell_type_best_clusters": output_stem.with_name(
            f"{output_stem.name}_cell_type_best_clusters.csv"
        ),
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    return paths


def save_publication_figure(fig, path: str | Path, **kwargs) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(
        path,
        dpi=PUBLICATION_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        **kwargs,
    )


def make_publication_dotplot(
    adata: ad.AnnData,
    output_stem: str | Path,
    *,
    marker_dict: Mapping[str, list[str]] = TME_MARKER_DICT,
    groupby: str = "cluster",
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    model_label: str | None = None,
    metrics_dir: str | Path | None = None,
    save_tables: bool = True,
    leiden_resolution: float = 0.3,
    **dotplot_kwargs,
) -> dict:
    """Create and save publication-resolution TME marker dot plots and tables."""
    apply_publication_style()

    filtered_marker_dict, missing_markers = filter_markers(adata, marker_dict)
    if missing_markers:
        print("Missing marker genes (not in data):", missing_markers)
    if not filtered_marker_dict:
        raise ValueError("No marker genes from the dictionary are present in the AnnData object.")

    safe_dotplot_kwargs = {
        key: value
        for key, value in dotplot_kwargs.items()
        if key not in _NON_SCANPY_KEYS
    }
    kwargs = {**DEFAULT_DOTPLOT_KWARGS, **safe_dotplot_kwargs}
    tables = compute_dotplot_tables(
        adata,
        filtered_marker_dict,
        groupby=groupby,
        standard_scale=kwargs.get("standard_scale", "var"),
    )

    plot_adata = adata.copy()
    prepare_expression(plot_adata)

    if figsize is None:
        n_clusters = plot_adata.obs[groupby].nunique()
        figsize = _dotplot_figsize(filtered_marker_dict, n_clusters)

    dotplot = sc.pl.DotPlot(
        plot_adata,
        filtered_marker_dict,
        groupby,
        figsize=figsize,
        standard_scale=kwargs.get("standard_scale"),
    )
    if kwargs.get("dendrogram"):
        dotplot.add_dendrogram()
    dotplot = dotplot.style(
        cmap=kwargs.get("color_map", "Reds"),
        dot_max=kwargs.get("dot_max"),
        largest_dot=kwargs.get("largest_dot", DEFAULT_DOTPLOT_KWARGS["largest_dot"]),
    ).legend(
        colorbar_title=kwargs.get("colorbar_title"),
        size_title=kwargs.get("size_title"),
    )
    dotplot.make_figure()
    _style_dotplot_for_panel(dotplot)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "png": output_stem.with_suffix(".png"),
        "pdf": output_stem.with_suffix(".pdf"),
        "svg": output_stem.with_suffix(".svg"),
    }
    save_publication_figure(dotplot, figure_paths["png"])
    dotplot.savefig(figure_paths["pdf"], bbox_inches="tight", facecolor="white", edgecolor="none")
    dotplot.savefig(figure_paths["svg"], bbox_inches="tight", facecolor="white", edgecolor="none")

    if show:
        plt.show()
    else:
        plt.close(dotplot.fig)

    metrics_dir = Path(metrics_dir or output_stem.parent.parent / "fig3_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    table_paths = {}
    if save_tables:
        table_paths = save_dotplot_tables(tables, metrics_dir / output_stem.name)

    caption = build_figure_caption(
        model_label or output_stem.name.replace("dotplot_", "").replace("_", " "),
        tables,
        groupby=groupby,
        n_cells=adata.n_obs,
        resolution=leiden_resolution,
    )
    caption_path = metrics_dir / f"{output_stem.name}_caption.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n")

    return {
        "figures": figure_paths,
        "tables": tables,
        "table_paths": table_paths,
        "caption": caption,
        "caption_path": caption_path,
    }
