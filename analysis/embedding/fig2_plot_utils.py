"""Plot helpers for Fig2 embedding metric figures."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex

PUBLICATION_RC = {
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
}


def apply_publication_style():
    sns.set_theme(style="white", context="paper", rc=PUBLICATION_RC)
    plt.rcParams.update(PUBLICATION_RC)


def save_publication_figure(fig, path, **kwargs):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none", **kwargs)


def _lighten_color(color, amount=0.45):
    c = np.array(to_rgb(color))
    white = np.array([1, 1, 1])
    return to_hex(c * (1 - amount) + white * amount)


def plot_metric_grouped_single_axis(
    df,
    metric_col,
    group_col="group",
    model_col="model",
    title=None,
    ylabel=None,
    ylim=None,
    group_order=None,
    sort_models_within_group=True,
    figsize=None,
    point_size=2.8,
    point_alpha=0.55,
    box_alpha=0.45,
    show_points=True,
    show_boxes=True,
    separator=True,
    save_path=None,
):
    apply_publication_style()

    plot_df = df[[group_col, model_col, metric_col]].dropna().copy()
    if plot_df.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    if group_order is None:
        group_order = list(plot_df[group_col].drop_duplicates())
    else:
        group_order = [g for g in group_order if g in set(plot_df[group_col])]

    group_colors = dict(zip(group_order, sns.color_palette("husl", len(group_order))))

    ordered_models = []
    model_to_group = {}
    group_centers = {}
    group_boundaries = []
    start = 0

    for group in group_order:
        sub = plot_df[plot_df[group_col] == group]
        if sort_models_within_group:
            models = (
                sub.groupby(model_col, observed=True)[metric_col]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            models = list(sub[model_col].drop_duplicates())

        ordered_models.extend(models)
        for model in models:
            model_to_group[model] = group

        end = start + len(models) - 1
        group_centers[group] = (start + end) / 2
        group_boundaries.append(end + 0.5)
        start = end + 1

    group_boundaries = group_boundaries[:-1]
    plot_df[model_col] = pd.Categorical(plot_df[model_col], categories=ordered_models, ordered=True)

    model_palette = {
        model: _lighten_color(group_colors[model_to_group[model]], amount=box_alpha)
        for model in ordered_models
    }

    if figsize is None:
        figsize = (max(6, len(ordered_models) * 0.5 + 1.5), max(5.5, 6))

    fig, ax = plt.subplots(figsize=figsize)

    if show_boxes:
        sns.boxplot(
            data=plot_df,
            x=model_col,
            y=metric_col,
            order=ordered_models,
            hue=model_col,
            palette=model_palette,
            dodge=False,
            width=0.35,
            linewidth=0.8,
            showfliers=False,
            showcaps=False,
            legend=False,
            ax=ax,
            boxprops={"edgecolor": "none", "alpha": 0.35},
            medianprops={"color": "#1F2937", "linewidth": 1.6},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"linewidth": 0, "alpha": 0},
            zorder=2,
        )

    if show_points:
        for group in group_order:
            sub = plot_df[plot_df[group_col] == group]
            sns.stripplot(
                data=sub,
                x=model_col,
                y=metric_col,
                order=ordered_models,
                color=group_colors[group],
                size=point_size,
                alpha=point_alpha,
                jitter=0.16,
                linewidth=0,
                ax=ax,
                zorder=3,
            )

    if separator:
        for boundary in group_boundaries:
            ax.axvline(boundary, color="#CCCCCC", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        y_min = plot_df[metric_col].min()
        y_max = plot_df[metric_col].max()
        y_pad = 0.08 * (y_max - y_min if y_max > y_min else 1)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad * 1.6)

    y0, y1 = ax.get_ylim()
    group_label_y = y1 + 0.04 * (y1 - y0)
    for group, center in group_centers.items():
        ax.text(center, group_label_y, str(group), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel or metric_col, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=18)

    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=3, width=0.8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="top", rotation_mode="anchor", fontsize=10)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.86, bottom=0.30, left=0.10, right=0.99)

    if save_path:
        save_publication_figure(fig, save_path)

    return fig, ax
