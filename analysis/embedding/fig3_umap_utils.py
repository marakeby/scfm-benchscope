"""Publication-quality UMAP panel helpers for Fig 3a embedding figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D

PUBLICATION_DPI = 450

PANEL_FONT_SIZES = {
    "tick": 14,
    "axis_label": 15,
    "title": 16,
    "legend": 12,
}

PUBLICATION_RC = {
    "figure.dpi": 100,
    "savefig.dpi": PUBLICATION_DPI,
    "font.size": PANEL_FONT_SIZES["tick"],
    "axes.titlesize": PANEL_FONT_SIZES["title"],
    "axes.labelsize": PANEL_FONT_SIZES["axis_label"],
    "xtick.labelsize": PANEL_FONT_SIZES["tick"],
    "ytick.labelsize": PANEL_FONT_SIZES["tick"],
    "legend.fontsize": PANEL_FONT_SIZES["legend"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_publication_style() -> None:
    plt.rcParams.update(PUBLICATION_RC)


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


def pretty_category(name: str) -> str:
    """Turn stored labels like ``Cancer_cell`` into legend text."""
    return str(name).replace("_", " ")


def load_embedding_adata(
    path: str | Path,
    embedding_key: str,
    *,
    obs_columns: Sequence[str] = ("label",),
) -> ad.AnnData:
    """Load only the embedding matrix and columns needed for UMAP plots."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    src = ad.read_h5ad(path, backed="r")
    try:
        missing_obs = [col for col in obs_columns if col not in src.obs.columns]
        if missing_obs:
            raise KeyError(f"{path} is missing obs columns: {missing_obs}")
        if embedding_key not in src.obsm:
            raise KeyError(
                f"{path} is missing obsm key {embedding_key!r}; "
                f"available: {list(src.obsm.keys())}"
            )

        adata = ad.AnnData(obs=src.obs.loc[:, list(obs_columns)].copy())
        adata.obsm[embedding_key] = np.asarray(src.obsm[embedding_key])
    finally:
        src.file.close()

    return adata


def compute_umap(
    adata: ad.AnnData,
    embedding_key: str,
    *,
    n_neighbors: int = 25,
    random_state: int = 0,
) -> None:
    """Build a neighbor graph in embedding space and compute UMAP in place."""
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Missing obsm key {embedding_key!r}; available: {list(adata.obsm.keys())}"
        )
    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=random_state,
    )
    sc.tl.umap(adata, random_state=random_state)


def _category_order(adatas: Sequence[ad.AnnData], color: str) -> list[str]:
    first = adatas[0].obs[color]
    if hasattr(first, "cat"):
        ordered = [str(cat) for cat in first.astype("category").cat.categories]
    else:
        ordered = [str(cat) for cat in first.astype(str).unique()]

    extras: list[str] = []
    seen = set(ordered)
    for adata in adatas[1:]:
        for cat in adata.obs[color].astype(str).unique():
            if cat not in seen:
                extras.append(cat)
                seen.add(cat)
    return ordered + extras


def _palette_for_categories(categories: Sequence[str]) -> dict[str, str]:
    colors = list(sc.pl.palettes.default_20)
    if len(categories) > len(colors):
        colors.extend(sc.pl.palettes.default_102)
    return dict(zip(categories, colors[: len(categories)]))


def _strip_axes(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure_caption(
    panels: Sequence[Mapping],
    *,
    color: str = "label",
    color_label: str | None = None,
    n_neighbors: int = 25,
    dataset_label: str = "pre-treatment BRCA tumor cells",
) -> str:
    n_cells = panels[0]["adata"].n_obs
    n_labels = panels[0]["adata"].obs[color].nunique()
    titles = ", ".join(panel["title"] for panel in panels)
    if color_label is None:
        color_label = "cell-type label" if color == "label" else pretty_category(color)
    return (
        f"UMAP of {dataset_label} (n={n_cells:,} cells) colored by "
        f"{color_label} ({n_labels} categories). "
        f"Panels compare {titles}. Neighbor graphs were computed in embedding "
        f"space (k={n_neighbors})."
    )


def plot_umap_panels(
    panels: Sequence[Mapping],
    output_stem: str | Path,
    *,
    color: str = "label",
    n_neighbors: int = 25,
    figsize: tuple[float, float] | None = None,
    point_size: float | None = None,
    n_legend_cols: int | None = None,
    show: bool = False,
    metrics_dir: str | Path | None = None,
) -> dict:
    """Plot side-by-side UMAPs with a shared legend and save PNG/PDF/SVG."""
    if not panels:
        raise ValueError("panels must contain at least one dataset.")

    apply_publication_style()

    adatas = [panel["adata"] for panel in panels]
    for panel in panels:
        if "X_umap" not in panel["adata"].obsm:
            raise KeyError(
                f"Panel {panel['title']!r} has no X_umap; call compute_umap first."
            )

    categories = _category_order(adatas, color)
    palette = _palette_for_categories(categories)
    n_panels = len(panels)
    figsize = figsize or (4.2 * n_panels, 4.4)
    n_legend_cols = n_legend_cols or min(len(categories), 8)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, panel in zip(axes, panels):
        adata = panel["adata"]
        adata.obs[color] = adata.obs[color].astype("category")
        adata.uns[f"{color}_colors"] = [palette[str(cat)] for cat in adata.obs[color].cat.categories]
        sc.pl.umap(
            adata,
            color=color,
            ax=ax,
            show=False,
            title=panel["title"],
            frameon=False,
            legend_loc=None,
            size=point_size,
            palette=palette,
        )
        ax.set_title(panel["title"], fontsize=PANEL_FONT_SIZES["title"], pad=8)
        _strip_axes(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[cat],
            markeredgecolor="none",
            markersize=8,
            label=pretty_category(cat),
        )
        for cat in categories
    ]
    fig.legend(
        handles,
        [pretty_category(cat) for cat in categories],
        loc="lower center",
        ncol=n_legend_cols,
        frameon=False,
        fontsize=PANEL_FONT_SIZES["legend"],
        bbox_to_anchor=(0.5, 0.0),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "png": output_stem.with_suffix(".png"),
        "pdf": output_stem.with_suffix(".pdf"),
        "svg": output_stem.with_suffix(".svg"),
    }
    for path in figure_paths.values():
        save_publication_figure(fig, path)

    caption = build_figure_caption(panels, color=color, n_neighbors=n_neighbors)
    metrics_dir = Path(metrics_dir or output_stem.parent.parent / "fig3_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    caption_path = metrics_dir / f"{output_stem.name}_caption.txt"
    caption_path.write_text(caption + "\n")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "figures": figure_paths,
        "caption": caption,
        "caption_path": caption_path,
        "categories": categories,
    }
