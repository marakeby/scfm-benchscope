"""
Boxplot visualization generator for results summaries.

Creates boxplot visualizations at different summary levels:
- Method level: Metric distributions across datasets for one method
- Task level: Metric distributions across methods for one task
- Global level: Overall metric distributions across methods/tasks/datasets
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logs_ import get_logger
from utils.metric_definitions import (
    METHOD_COLORS,
    TASK_COLORS,
    is_lower_better,
)

# Suppress seaborn's internal deprecation warning about 'vert' parameter
# This warning comes from seaborn's internal code, not ours - we're already using
# the modern 'orientation' parameter correctly
warnings.filterwarnings(
    'ignore',
    category=PendingDeprecationWarning,
    message='vert: bool will be deprecated',
    module='seaborn.categorical'
)

logger = get_logger()


class BoxplotGenerator:
    """
    Generate boxplot visualizations for results summaries.
    
    Creates boxplots at different summary levels:
    - Method level: Metric distributions across datasets for one method
    - Task level: Metric distributions across methods for one task
    - Global level: Overall metric distributions across methods/tasks/datasets
    
    Attributes:
        save_dir: Directory to save generated plots
        figsize: Default figure size (width, height) in inches
        dpi: Resolution for saved figures
        style: Seaborn style for plots
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 150,
        style: str = 'whitegrid'
    ):
        """
        Initialize boxplot generator.
        
        Args:
            save_dir: Directory to save plots
            figsize: Default figure size (width, height)
            dpi: Resolution for saved figures
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set style
        sns.set_style(style)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def _get_color_palette(
        self,
        n_colors: int,
        palette_type: str = 'method'
    ) -> List[str]:
        """
        Get a color palette for visualization.
        
        Args:
            n_colors: Number of colors needed
            palette_type: Type of palette ('method', 'task', 'dataset')
        
        Returns:
            List of hex color codes
        """
        if palette_type == 'method':
            base_palette = METHOD_COLORS
        elif palette_type == 'task':
            base_palette = TASK_COLORS
        else:
            base_palette = sns.color_palette('husl', n_colors).as_hex()
            return base_palette
        
        if n_colors <= len(base_palette):
            return base_palette[:n_colors]
        else:
            # Extend palette if needed
            return sns.color_palette('husl', n_colors).as_hex()
    
    def _clean_metric_name(self, metric_name: str) -> str:
        """
        Clean metric name for display (remove prefixes like 'batch_effects_').
        
        Args:
            metric_name: Raw metric name
        
        Returns:
            Cleaned metric name for display
        """
        # Remove common prefixes
        prefixes = ['batch_effects_', 'biological_signal_', 'classification_', 'annotation_', 'drug_response_']
        for prefix in prefixes:
            if metric_name.startswith(prefix):
                return metric_name[len(prefix):]
        return metric_name

    def _metric_aggregation_group(self, metric_name: str) -> str:
        """
        Return aggregation group for grouping task boxplots by prediction type (avg, mil, vote, cell).
        Used for filenames like boxplot_task_<task>_mil.png instead of part1, part2.
        
        Args:
            metric_name: e.g. 'global_score', 'classification_mil_AUC', 'classification_cell_AUC'
        
        Returns:
            'overview' for global_score, 'avg'/'mil'/'vote'/'cell' for classification_<x>_*, else 'other'
        """
        if metric_name == 'global_score':
            return 'overview'
        if metric_name.startswith('classification_'):
            parts = metric_name.split('_')
            if len(parts) >= 2:
                return parts[1]
        return 'other'
    
    def plot_method_summary_boxplots(
        self,
        results_df: pd.DataFrame,
        method_name: str,
        task_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        max_metrics_per_plot: int = 8
    ) -> List[Path]:
        """
        Create boxplots showing metric distributions across datasets for a method.
        
        Purpose: "How consistent is this method across different datasets?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            method_name: Method to visualize
            task_name: Optional task filter
            metrics: List of metrics to plot (if None, uses all available)
            max_metrics_per_plot: Maximum metrics per figure (creates multiple plots if needed)
        
        Returns:
            List of paths to saved plot files
        """
        # Filter data
        mask = results_df['method'] == method_name
        if task_name is not None:
            mask &= results_df['task'] == task_name
        
        method_df = results_df[mask].copy()
        
        if method_df.empty:
            logger.warning(f"No data for method={method_name}, task={task_name}")
            return []
        
        # Get metric columns (only numeric; exclude identifiers and subgroups)
        if metrics is None:
            exclude_cols = {'dataset', 'task', 'method', 'n_metrics', 'dataset_subgroup'}
            candidates = [
                c for c in method_df.columns
                if c not in exclude_cols and not c.endswith('_std') and method_df[c].notna().any()
            ]
            # Ensure we only plot numeric columns (avoid "Unable to parse string" in sns.boxplot)
            metrics = [
                c for c in candidates
                if pd.api.types.is_numeric_dtype(method_df[c])
            ]
            if not metrics and candidates:
                logger.warning(
                    f"Method summary boxplot: no numeric metric columns (excluded non-numeric: {candidates})"
                )
        
        # Filter to metrics that have variance (boxplots need distribution)
        metrics = [m for m in metrics if method_df[m].dropna().nunique() > 1 or len(method_df) > 1]
        
        if not metrics:
            logger.warning(f"No plottable metrics for method={method_name}")
            return []
        
        saved_paths = []
        
        # Create plots in batches
        for batch_idx in range(0, len(metrics), max_metrics_per_plot):
            batch_metrics = metrics[batch_idx:batch_idx + max_metrics_per_plot]
            
            # Melt data for seaborn
            plot_df = method_df[['dataset'] + batch_metrics].melt(
                id_vars=['dataset'],
                var_name='metric',
                value_name='value'
            )
            
            # Clean metric names for display
            plot_df['metric_display'] = plot_df['metric'].apply(self._clean_metric_name)
            
            # Create figure
            n_metrics = len(batch_metrics)
            fig_width = max(10, n_metrics * 1.5)
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            
            # Get unique datasets and create color palette
            datasets = sorted(method_df['dataset'].unique())
            dataset_palette = self._get_color_palette(len(datasets), 'dataset')
            
            # Create boxplot
            sns.boxplot(
                data=plot_df,
                x='metric_display',
                y='value',
                ax=ax,
                color='#1f77b4',
                width=0.6,
                showfliers=True,
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                orient='v',
            )
            
            # Overlay individual points colored by dataset
            sns.stripplot(
                data=plot_df,
                x='metric_display',
                y='value',
                hue='dataset',
                ax=ax,
                palette=dataset_palette,
                size=5,
                alpha=0.7,
                jitter=True,
                dodge=False,
                legend=True
            )
            
            # Add legend with dataset names
            ax.legend(
                title='Dataset',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=9,
                title_fontsize=10
            )
            
            # Formatting
            task_str = f" ({task_name})" if task_name else ""
            ax.set_title(f'Method: {method_name}{task_str}\nMetric Distributions Across Datasets', fontsize=12)
            ax.set_xlabel('Metric', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Annotations
            n_datasets = len(method_df['dataset'].unique())
            ax.text(
                0.02, 0.98, f'n_datasets = {n_datasets}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            ax.text(
                0.5, 0.02, 'Values in original scale; direction per metric in metric_reference.csv',
                transform=ax.transAxes, fontsize=8, ha='center', color='gray', style='italic'
            )
            plt.tight_layout()
            
            # Save
            suffix = f'_part{batch_idx // max_metrics_per_plot + 1}' if len(metrics) > max_metrics_per_plot else ''
            task_suffix = f'_{task_name}' if task_name else ''
            filename = f'boxplot_method_{method_name}{task_suffix}{suffix}.png'
            save_path = self.save_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            saved_paths.append(save_path)
            logger.info(f"Saved method boxplot to {save_path}")
        
        return saved_paths
    
    def plot_task_summary_boxplots(
        self,
        results_df: pd.DataFrame,
        task_name: str,
        metrics: Optional[List[str]] = None,
        show_individual_datasets: bool = True,
        max_metrics_per_plot: int = 6
    ) -> List[Path]:
        """
        Create boxplots comparing methods for a specific task.
        
        Purpose: "How do different methods compare for this task?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            task_name: Task to visualize
            metrics: List of metrics to plot (if None, uses global_score + key metrics)
            show_individual_datasets: If True, shows individual dataset points
            max_metrics_per_plot: Maximum metrics per figure
        
        Returns:
            List of paths to saved plot files
        """
        task_df = results_df[results_df['task'] == task_name].copy()
        
        if task_df.empty:
            logger.warning(f"No data for task={task_name}")
            return []
        
        # Get metric columns (numeric only; exclude identifiers, subgroups, and _std)
        if metrics is None:
            exclude_cols = {'dataset', 'task', 'method', 'n_metrics', 'global_score_std', 'dataset_subgroup'}
            extra = [
                c for c in task_df.columns
                if c not in exclude_cols and c != 'global_score' and not c.endswith('_std')
                and task_df[c].notna().any() and pd.api.types.is_numeric_dtype(task_df[c])
            ]
            metrics = ['global_score'] + extra
        
        # Filter to metrics with data; exclude _std columns (used only for error bars)
        metrics = [
            m for m in metrics
            if m in task_df.columns and not m.endswith('_std') and task_df[m].notna().any()
        ]
        if not metrics:
            logger.warning(f"No plottable metrics for task={task_name}")
            return []
        # Group metrics by aggregation type (avg, mil, vote, cell, overview, other) for separate files
        aggregation_order = ['overview', 'avg', 'mil', 'vote', 'cell', 'other']
        groups: Dict[str, List[str]] = {}
        for m in metrics:
            grp = self._metric_aggregation_group(m)
            groups.setdefault(grp, []).append(m)
        methods = sorted(task_df['method'].unique())
        palette = self._get_color_palette(len(methods), 'method')
        # Build CV support / label-splits line per dataset for figure title
        support_parts: List[str] = []
        if 'cv_label_splits' in task_df.columns:
            for ds in sorted(task_df['dataset'].unique()):
                row = task_df[task_df['dataset'] == ds].iloc[0]
                s = row.get('cv_label_splits')
                if pd.notna(s) and str(s).strip():
                    support_parts.append(f"{ds}: {s}")
        title_support = ("CV support: " + "; ".join(support_parts)) if support_parts else None
        # Remove old part1/part2/... files from previous naming scheme (now we use _avg, _mil, etc.)
        for old_path in self.save_dir.glob(f'boxplot_task_{task_name}_part*.png'):
            try:
                old_path.unlink()
                logger.debug(f"Removed old task boxplot {old_path.name}")
            except OSError:
                pass
        saved_paths = []
        for grp in aggregation_order:
            if grp not in groups or not groups[grp]:
                continue
            group_metrics = groups[grp]
            # Within group, create batches if needed
            for batch_idx in range(0, len(group_metrics), max_metrics_per_plot):
                batch_metrics = group_metrics[batch_idx:batch_idx + max_metrics_per_plot]
                n_metrics = len(batch_metrics)
                n_cols = min(3, n_metrics)
                n_rows = (n_metrics + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
                axes = axes.flatten()
                for idx, metric in enumerate(batch_metrics):
                    ax = axes[idx]
                    std_col = f'{metric}_std'
                    has_cv_std = std_col in task_df.columns and task_df[std_col].notna().any()
                    if has_cv_std:
                        # Cross-validated results: plot mean ± std (from CV) per method
                        def method_std(g: pd.DataFrame, **kwargs) -> float:
                            if len(g) == 1 and pd.notna(g[std_col].iloc[0]):
                                return float(g[std_col].iloc[0])
                            if len(g) > 1:
                                return float(g[metric].std()) if g[metric].notna().any() else np.nan
                            return np.nan
                        means = task_df.groupby('method')[metric].mean()
                        stds = task_df.groupby('method', sort=True).apply(method_std)
                        means = means.reindex(methods)
                        stds = stds.reindex(methods)
                        std_vals = np.where(np.isnan(stds.values), 0.0, stds.values)
                        x_pos = np.arange(len(methods))
                        ax.bar(
                            x_pos,
                            means.values,
                            yerr=std_vals,
                            color=[palette[i] for i in range(len(methods))],
                            capsize=4,
                            error_kw={'elinewidth': 1.5},
                        )
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(methods, rotation=45, ha='right')
                        ax.text(
                            0.02, 0.98, 'Mean ± SD (CV)',
                            transform=ax.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        )
                    else:
                        sns.boxplot(
                            data=task_df,
                            x='method',
                            y=metric,
                            hue='method',
                            ax=ax,
                            palette=palette,
                            width=0.6,
                            showfliers=False,
                            legend=False,
                        )
                        if show_individual_datasets:
                            datasets = sorted(task_df['dataset'].unique())
                            dataset_palette = self._get_color_palette(len(datasets), 'dataset')
                            sns.stripplot(
                                data=task_df,
                                x='method',
                                y=metric,
                                hue='dataset',
                                ax=ax,
                                palette=dataset_palette,
                                size=5,
                                alpha=0.7,
                                jitter=True,
                                dodge=False,
                                legend=idx == 0
                            )
                            if idx == 0:
                                ax.legend(
                                    title='Dataset',
                                    bbox_to_anchor=(1.02, 1),
                                    loc='upper left',
                                    fontsize=8,
                                    title_fontsize=9
                                )
                    metric_display = self._clean_metric_name(metric)
                    ax.set_title(metric_display, fontsize=11)
                    ax.set_xlabel('')
                    ax.set_ylabel('Value', fontsize=9)
                    ax.tick_params(axis='x', rotation=45)
                    if is_lower_better(metric):
                        ax.text(
                            0.98, 0.02, '↓ lower is better',
                            transform=ax.transAxes, fontsize=8, ha='right',
                            color='gray', style='italic'
                        )
                    else:
                        ax.text(
                            0.98, 0.02, '↑ higher is better',
                            transform=ax.transAxes, fontsize=8, ha='right',
                            color='gray', style='italic'
                        )
                # Hide unused subplots
                for j in range(len(batch_metrics), len(axes)):
                    axes[j].set_visible(False)
                title_lines = [f'Task: {task_name}']
                if title_support:
                    title_lines.append(title_support)
                title_lines.append('Method Comparison Across Datasets')
                fig.suptitle('\n'.join(title_lines), fontsize=13, y=1.02)
                plt.tight_layout()
                n_batches_in_group = (len(group_metrics) + max_metrics_per_plot - 1) // max_metrics_per_plot
                suffix = f'_part{batch_idx // max_metrics_per_plot + 1}' if n_batches_in_group > 1 else ''
                # Primary plot: "other" (all methods, all task metrics) gets the main name; "overview" gets _overview
                if grp == 'other':
                    filename = f'boxplot_task_{task_name}{suffix}.png'
                else:
                    filename = f'boxplot_task_{task_name}_{grp}{suffix}.png'
                save_path = self.save_dir / filename
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_paths.append(save_path)
                logger.info(f"Saved task boxplot to {save_path}")
        return saved_paths
    
    def plot_global_score_comparison(
        self,
        results_df: pd.DataFrame,
        group_by: str = 'method',
        facet_by: Optional[str] = 'task'
    ) -> Optional[Path]:
        """
        Create boxplot comparing global scores across methods/tasks.
        
        Purpose: "Overall performance comparison across all experiments"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            group_by: Primary grouping ('method' or 'task')
            facet_by: Secondary faceting ('task', 'method', or None)
        
        Returns:
            Path to saved plot file, or None if failed
        """
        if results_df.empty:
            logger.warning("No data for global score comparison")
            return None
        
        if 'global_score' not in results_df.columns:
            logger.warning("global_score column not found")
            return None
        
        # Determine plot structure
        if facet_by and facet_by in results_df.columns:
            facets = sorted(results_df[facet_by].unique())
            n_facets = len(facets)
            
            n_cols = min(3, n_facets)
            n_rows = (n_facets + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
            axes = axes.flatten()
            
            groups = sorted(results_df[group_by].unique())
            palette = self._get_color_palette(len(groups), group_by)
            
            for idx, facet_val in enumerate(facets):
                ax = axes[idx]
                facet_df = results_df[results_df[facet_by] == facet_val]
                
                sns.boxplot(
                    data=facet_df,
                    x=group_by,
                    y='global_score',
                    hue=group_by,
                    ax=ax,
                    palette=palette,
                    width=0.6,
                    showfliers=False,
                    legend=False,
                )
                
                # Get unique datasets and create color palette
                datasets = sorted(facet_df['dataset'].unique())
                dataset_palette = self._get_color_palette(len(datasets), 'dataset')
                
                sns.stripplot(
                    data=facet_df,
                    x=group_by,
                    y='global_score',
                    hue='dataset',
                    ax=ax,
                    palette=dataset_palette,
                    size=5,
                    alpha=0.7,
                    jitter=True,
                    dodge=False,
                    legend=idx == 0  # Show legend only on first subplot
                )
                
                # Add legend with dataset names if this is the first subplot
                if idx == 0:
                    ax.legend(
                        title='Dataset',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        fontsize=8,
                        title_fontsize=9
                    )
                
                ax.set_title(facet_val, fontsize=11)
                ax.set_xlabel('')
                ax.set_ylabel('Global Score', fontsize=9)
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for idx in range(n_facets, len(axes)):
                axes[idx].set_visible(False)
            
            fig.suptitle(f'Global Score Comparison by {group_by.title()}\n(faceted by {facet_by})', fontsize=13, y=1.02)
            filename = f'boxplot_global_score_by_{group_by}_facet_{facet_by}.png'
        
        else:
            # Single plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            groups = sorted(results_df[group_by].unique())
            palette = self._get_color_palette(len(groups), group_by)
            
            sns.boxplot(
                data=results_df,
                x=group_by,
                y='global_score',
                hue=group_by,
                ax=ax,
                palette=palette,
                width=0.6,
                showfliers=False,
                legend=False,
            )
            
            # Get unique datasets and create color palette
            datasets = sorted(results_df['dataset'].unique())
            dataset_palette = self._get_color_palette(len(datasets), 'dataset')
            
            sns.stripplot(
                data=results_df,
                x=group_by,
                y='global_score',
                hue='dataset',
                ax=ax,
                palette=dataset_palette,
                size=5,
                alpha=0.7,
                jitter=True,
                dodge=False,
                legend=True
            )
            
            # Add legend with dataset names
            ax.legend(
                title='Dataset',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=9,
                title_fontsize=10
            )
            
            ax.set_title(f'Global Score Comparison by {group_by.title()}', fontsize=12)
            ax.set_xlabel(group_by.title(), fontsize=10)
            ax.set_ylabel('Global Score', fontsize=10)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            filename = f'boxplot_global_score_by_{group_by}.png'
        
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved global score boxplot to {save_path}")
        return save_path
    
    def plot_metric_heatmap_with_boxplots(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        aggregate_by: str = 'method'
    ) -> Optional[Path]:
        """
        Create a combined heatmap with marginal boxplots.
        
        Shows mean metric values as heatmap with boxplot distributions on margins.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metrics: List of metrics to include
            aggregate_by: Aggregation axis ('method' or 'task')
        
        Returns:
            Path to saved plot file, or None if failed
        """
        if results_df.empty:
            logger.warning("No data for heatmap")
            return None
        
        # Get metric columns (numeric only; exclude identifiers and subgroups)
        if metrics is None:
            exclude_cols = {'dataset', 'task', 'method', 'n_metrics', 'global_score', 'global_score_std', 'dataset_subgroup'}
            candidates = [
                c for c in results_df.columns
                if c not in exclude_cols and not c.endswith('_std') and results_df[c].notna().any()
            ]
            metrics = [c for c in candidates if pd.api.types.is_numeric_dtype(results_df[c])]
        
        if not metrics:
            logger.warning("No metrics for heatmap")
            return None
        
        # Limit to top metrics by variance
        if len(metrics) > 15:
            variances = results_df[metrics].var().sort_values(ascending=False)
            metrics = variances.head(15).index.tolist()
        
        # Aggregate by method/task
        if aggregate_by == 'method':
            pivot_df = results_df.groupby('method')[metrics].mean()
            ylabel = 'Method'
        else:
            pivot_df = results_df.groupby('task')[metrics].mean()
            ylabel = 'Task'
        
        # Clean metric names for display
        pivot_df.columns = [self._clean_metric_name(m) for m in pivot_df.columns]
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(
            2, 2, 
            width_ratios=[4, 1], 
            height_ratios=[1, 4],
            wspace=0.05, 
            hspace=0.05
        )
        
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        
        # Heatmap
        sns.heatmap(
            pivot_df,
            ax=ax_heatmap,
            cmap='RdYlGn',
            annot=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.5, 'label': 'Mean Value'}
        )
        ax_heatmap.set_xlabel('Metric', fontsize=10)
        ax_heatmap.set_ylabel(ylabel, fontsize=10)
        ax_heatmap.tick_params(axis='x', rotation=45)
        
        # Top boxplot (metric distributions)
        metric_means = pivot_df.mean(axis=0)
        metric_stds = pivot_df.std(axis=0)
        ax_top.bar(range(len(metric_means)), metric_means, yerr=metric_stds, color='#1f77b4', alpha=0.7, capsize=3)
        ax_top.set_xlim(-0.5, len(metrics) - 0.5)
        ax_top.set_ylabel('Mean ± Std', fontsize=9)
        ax_top.tick_params(labelbottom=False)
        
        # Right boxplot (method/task distributions)
        row_means = pivot_df.mean(axis=1)
        row_stds = pivot_df.std(axis=1)
        ax_right.barh(range(len(row_means)), row_means, xerr=row_stds, color='#ff7f0e', alpha=0.7, capsize=3)
        ax_right.set_ylim(-0.5, len(row_means) - 0.5)
        ax_right.set_xlabel('Mean ± Std', fontsize=9)
        ax_right.tick_params(labelleft=False)
        ax_right.invert_yaxis()
        
        fig.suptitle(f'Metric Overview by {ylabel}\n(Mean values with marginal distributions)', fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        filename = f'heatmap_with_boxplots_by_{aggregate_by}.png'
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved heatmap with boxplots to {save_path}")
        return save_path
    
    def plot_dataset_comparison_boxplots(
        self,
        results_df: pd.DataFrame,
        metric: str = 'global_score'
    ) -> Optional[Path]:
        """
        Create boxplots comparing datasets, grouped by task.
        
        Purpose: "How do different datasets compare in difficulty/performance?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric: Metric to compare
        
        Returns:
            Path to saved plot file, or None if failed
        """
        if results_df.empty or metric not in results_df.columns:
            logger.warning(f"No data for dataset comparison with metric={metric}")
            return None
        
        tasks = sorted(results_df['task'].unique())
        n_tasks = len(tasks)
        
        n_cols = min(2, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, task in enumerate(tasks):
            ax = axes[idx]
            task_df = results_df[results_df['task'] == task]
            
            datasets = sorted(task_df['dataset'].unique())
            palette = self._get_color_palette(len(datasets), 'dataset')
            
            sns.boxplot(
                data=task_df,
                x='dataset',
                y=metric,
                hue='dataset',
                ax=ax,
                palette=palette,
                width=0.6,
                showfliers=False,
                legend=False,
            )
            
            sns.stripplot(
                data=task_df,
                x='dataset',
                y=metric,
                hue='method',
                ax=ax,
                size=6,
                alpha=0.7,
                dodge=True,
                legend=True if idx == 0 else False
            )
            
            ax.set_title(f'{task}', fontsize=11)
            ax.set_xlabel('Dataset', fontsize=10)
            ax.set_ylabel(self._clean_metric_name(metric), fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            if idx == 0:
                ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_tasks, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Dataset Comparison: {self._clean_metric_name(metric)}\n(colored by method)', fontsize=13, y=1.02)
        
        plt.tight_layout()
        
        filename = f'boxplot_dataset_comparison_{metric}.png'
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved dataset comparison boxplot to {save_path}")
        return save_path
    
    def plot_task_summary_by_subgroup(
        self,
        results_df: pd.DataFrame,
        task_name: str,
        metrics: Optional[List[str]] = None,
        max_metrics_per_plot: int = 6
    ) -> List[Path]:
        """
        Create boxplots comparing methods for a task, faceted by dataset_subgroup.
        
        Purpose: "How do methods compare within each dataset subgroup?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            task_name: Task to visualize
            metrics: List of metrics to plot (if None, uses global_score + key metrics)
            max_metrics_per_plot: Maximum metrics per figure
        
        Returns:
            List of paths to saved plot files
        """
        if 'dataset_subgroup' not in results_df.columns:
            return []
        
        task_df = results_df[(results_df['task'] == task_name) & results_df['dataset_subgroup'].notna()].copy()
        
        if task_df.empty:
            logger.warning(f"No subgroup data for task={task_name}")
            return []
        
        subgroups = sorted(task_df['dataset_subgroup'].unique())
        if len(subgroups) < 2:
            logger.info(f"Only one subgroup for task={task_name}, skipping faceted plot")
            return []
        
        # Get metric columns (numeric only)
        if metrics is None:
            exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'n_metrics', 'global_score_std'}
            extra = [
                c for c in task_df.columns
                if c not in exclude_cols and c != 'global_score' and not c.endswith('_std')
                and task_df[c].notna().any() and pd.api.types.is_numeric_dtype(task_df[c])
            ]
            metrics = ['global_score'] + extra
        metrics = [m for m in metrics if m in task_df.columns and task_df[m].notna().any()]
        if not metrics:
            return []
        methods = sorted(task_df['method'].unique())
        method_palette = self._get_color_palette(len(methods), 'method')
        saved_paths = []
        # Create one figure per metric, faceted by subgroup
        for metric in metrics[:max_metrics_per_plot]:
            n_subgroups = len(subgroups)
            n_cols = min(3, n_subgroups)
            n_rows = (n_subgroups + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
            axes = axes.flatten()
            
            for idx, subgroup in enumerate(subgroups):
                ax = axes[idx]
                subgroup_data = task_df[task_df['dataset_subgroup'] == subgroup]

                # Drop rows where this metric is NaN so seaborn's boxplot loop runs (avoids
                # UnboundLocalError in seaborn when iter_data yields zero groups)
                plot_data = subgroup_data.dropna(subset=[metric])
                if plot_data.empty or not plot_data[metric].notna().any():
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{subgroup}\n(n=0)', fontsize=10)
                    continue

                sns.boxplot(
                    data=plot_data,
                    x='method',
                    y=metric,
                    hue='method',
                    ax=ax,
                    palette=method_palette,
                    width=0.6,
                    showfliers=False,
                    legend=False,
                    orient='v',
                )

                # Overlay individual dataset points
                datasets = sorted(plot_data['dataset'].unique())
                dataset_palette = self._get_color_palette(len(datasets), 'dataset')
                
                sns.stripplot(
                    data=plot_data,
                    x='method',
                    y=metric,
                    hue='dataset',
                    ax=ax,
                    palette=dataset_palette,
                    size=5,
                    alpha=0.7,
                    jitter=True,
                    dodge=False,
                    legend=idx == 0
                )

                if idx == 0 and len(datasets) <= 10:
                    ax.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
                elif idx == 0:
                    ax.get_legend().remove()

                ax.set_title(f'{subgroup}\n(n={len(plot_data)} datasets)', fontsize=10)
                ax.set_xlabel('')
                ax.set_ylabel('Value' if idx % n_cols == 0 else '', fontsize=9)
                ax.tick_params(axis='x', rotation=45)
                
                # Direction indicator
                if is_lower_better(metric):
                    ax.text(0.98, 0.02, '↓', transform=ax.transAxes, fontsize=8, ha='right', color='gray')
                else:
                    ax.text(0.98, 0.02, '↑', transform=ax.transAxes, fontsize=8, ha='right', color='gray')
            
            # Hide unused subplots
            for idx in range(n_subgroups, len(axes)):
                axes[idx].set_visible(False)
            
            metric_display = self._clean_metric_name(metric)
            fig.suptitle(f'Task: {task_name} | Metric: {metric_display}\nMethod Comparison by Dataset Subgroup', fontsize=12, y=1.02)
            
            plt.tight_layout()
            
            filename = f'boxplot_task_{task_name}_by_subgroup_{metric}.png'
            save_path = self.save_dir / 'by_subgroup' / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            saved_paths.append(save_path)
            logger.info(f"Saved subgroup-faceted boxplot to {save_path}")
        
        return saved_paths
    
    def plot_method_subgroup_comparison(
        self,
        results_df: pd.DataFrame,
        method_name: str,
        task_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        max_metrics_per_plot: int = 6
    ) -> List[Path]:
        """
        Create boxplots comparing subgroups for a specific method.
        
        Purpose: "How does this method perform across different dataset subgroups?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            method_name: Method to visualize
            task_name: Optional task filter
            metrics: List of metrics to plot
            max_metrics_per_plot: Maximum metrics per figure
        
        Returns:
            List of paths to saved plot files
        """
        if 'dataset_subgroup' not in results_df.columns:
            return []
        
        mask = (results_df['method'] == method_name) & results_df['dataset_subgroup'].notna()
        if task_name:
            mask &= results_df['task'] == task_name
        
        method_df = results_df[mask].copy()
        
        if method_df.empty:
            return []
        
        subgroups = sorted(method_df['dataset_subgroup'].unique())
        if len(subgroups) < 2:
            return []
        
        # Get metric columns (numeric only)
        if metrics is None:
            exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'n_metrics', 'global_score_std'}
            extra = [
                c for c in method_df.columns
                if c not in exclude_cols and c != 'global_score' and not c.endswith('_std')
                and method_df[c].notna().any() and pd.api.types.is_numeric_dtype(method_df[c])
            ]
            metrics = ['global_score'] + extra
        metrics = [m for m in metrics if m in method_df.columns and method_df[m].notna().any()]
        if not metrics:
            return []
        subgroup_palette = self._get_color_palette(len(subgroups), 'task')  # Use task colors for variety
        
        saved_paths = []
        
        # Create subplots for metrics
        n_metrics = min(len(metrics), max_metrics_per_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics[:max_metrics_per_plot]):
            ax = axes[idx]
            
            sns.boxplot(
                data=method_df,
                x='dataset_subgroup',
                y=metric,
                hue='dataset_subgroup',
                ax=ax,
                palette=subgroup_palette,
                width=0.6,
                showfliers=False,
                legend=False,
                orient='v',
            )
            
            # Overlay individual dataset points
            sns.stripplot(
                data=method_df,
                x='dataset_subgroup',
                y=metric,
                hue='dataset',
                ax=ax,
                size=5,
                alpha=0.7,
                jitter=True,
                dodge=False,
                legend=idx == 0
            )
            
            if idx == 0:
                datasets = method_df['dataset'].unique()
                if len(datasets) <= 10:
                    ax.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
                else:
                    ax.get_legend().remove()
            
            metric_display = self._clean_metric_name(metric)
            ax.set_title(metric_display, fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            
            if is_lower_better(metric):
                ax.text(0.98, 0.02, '↓ lower is better', transform=ax.transAxes, fontsize=8, ha='right', color='gray', style='italic')
            else:
                ax.text(0.98, 0.02, '↑ higher is better', transform=ax.transAxes, fontsize=8, ha='right', color='gray', style='italic')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        task_str = f" ({task_name})" if task_name else ""
        fig.suptitle(f'Method: {method_name}{task_str}\nSubgroup Comparison', fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        task_suffix = f'_{task_name}' if task_name else ''
        filename = f'boxplot_method_{method_name}{task_suffix}_subgroup_comparison.png'
        save_path = self.save_dir / 'by_subgroup' / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append(save_path)
        logger.info(f"Saved method subgroup comparison to {save_path}")
        
        return saved_paths
    
    def plot_global_score_by_subgroup(
        self,
        results_df: pd.DataFrame
    ) -> Optional[Path]:
        """
        Create boxplot comparing global scores across methods, faceted by subgroup.
        
        Purpose: "Overall comparison: which method wins in each subgroup?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
        
        Returns:
            Path to saved plot file, or None if failed
        """
        if 'dataset_subgroup' not in results_df.columns or 'global_score' not in results_df.columns:
            return None
        
        subgroup_df = results_df[results_df['dataset_subgroup'].notna()].copy()
        if subgroup_df.empty:
            return None
        
        subgroups = sorted(subgroup_df['dataset_subgroup'].unique())
        if len(subgroups) < 2:
            return None
        
        methods = sorted(subgroup_df['method'].unique())
        method_palette = self._get_color_palette(len(methods), 'method')
        
        n_subgroups = len(subgroups)
        n_cols = min(4, n_subgroups)
        n_rows = (n_subgroups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, subgroup in enumerate(subgroups):
            ax = axes[idx]
            subgroup_data = subgroup_df[subgroup_df['dataset_subgroup'] == subgroup]
            
            sns.boxplot(
                data=subgroup_data,
                x='method',
                y='global_score',
                hue='method',
                ax=ax,
                palette=method_palette,
                width=0.6,
                showfliers=False,
                legend=False,
                orient='v',
            )
            
            sns.stripplot(
                data=subgroup_data,
                x='method',
                y='global_score',
                hue='dataset',
                ax=ax,
                size=5,
                alpha=0.7,
                jitter=True,
                dodge=False,
                legend=False
            )
            
            n_datasets = len(subgroup_data['dataset'].unique())
            ax.set_title(f'{subgroup}\n(n={n_datasets} datasets)', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Global Score' if idx % n_cols == 0 else '', fontsize=9)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_subgroups, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Global Score Comparison by Dataset Subgroup\n(higher is better)', fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        filename = 'boxplot_global_score_by_subgroup.png'
        save_path = self.save_dir / 'by_subgroup' / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved global score by subgroup boxplot to {save_path}")
        return save_path

    def plot_batch_bio_integration_summary(
        self,
        results_df: pd.DataFrame,
        task_name: str = 'batch_bio_integration',
    ) -> List[Path]:
        """
        Create a combined batch + bio + label-transfer comparison plot per subgroup.

        For each dataset subgroup, produces a 1x5 figure with metric panels:
          1. batch_effects_bras (BRAS)
          2. batch_effects_CiLISI_batch (CiLISI)
          3. biological_signal_nmi (NMI)
          4. biological_signal_ari (ARI)
          5. annotation_round_robin_F1 (F1 Label Transfer)

        Within each panel: methods on x-axis, boxplot + stripplot (datasets as
        coloured dots sharing a single legend across all panels).

        Also generates one "all_subgroups" overview plot when multiple subgroups
        exist.

        Args:
            results_df: DataFrame from scan_all_experiments()
            task_name: Task to filter on (default ``'batch_bio_integration'``).

        Returns:
            List of paths to saved plot files.
        """
        task_df = results_df[results_df['task'] == task_name].copy()
        if task_df.empty:
            logger.info(f"No data for task={task_name}, skipping integration summary")
            return []

        # Ordered metric panels ---------------------------------------------------
        METRIC_PANELS: List[Tuple[str, str]] = [
            ('batch_effects_bras', 'BRAS\n(Batch)'),
            ('batch_effects_CiLISI_batch', 'CiLISI\n(Batch)'),
            ('biological_signal_nmi', 'NMI\n(Bio)'),
            ('biological_signal_ari', 'ARI\n(Bio)'),
            ('annotation_round_robin_F1', 'F1\n(Label Transfer)'),
        ]

        # Keep only panels present in the data
        available_panels = [
            (col, label) for col, label in METRIC_PANELS
            if col in task_df.columns and task_df[col].notna().any()
        ]
        if not available_panels:
            logger.warning(f"No integration metrics found for task={task_name}")
            return []

        all_methods = sorted(task_df['method'].unique())
        method_to_color = dict(
            zip(all_methods, self._get_color_palette(len(all_methods), 'method'))
        )

        # Consistent dataset palette across all panels / subgroups
        all_datasets = sorted(task_df['dataset'].unique())
        dataset_palette = self._get_color_palette(len(all_datasets), 'dataset')
        dataset_color_map = dict(zip(all_datasets, dataset_palette))

        saved_paths: List[Path] = []

        # Determine subgroups (or treat entire df as one group)
        has_subgroups = (
            'dataset_subgroup' in task_df.columns
            and task_df['dataset_subgroup'].notna().any()
        )
        if has_subgroups:
            subgroups: List[Optional[str]] = sorted(
                task_df['dataset_subgroup'].dropna().unique()
            )
        else:
            subgroups = [None]

        # Optionally prepend an "all" view when there are multiple subgroups
        if has_subgroups and len(subgroups) > 1:
            subgroups = [None] + subgroups  # None = all data

        for subgroup in subgroups:
            if subgroup is None:
                plot_df = task_df
                subgroup_label = 'all_subgroups'
            else:
                plot_df = task_df[task_df['dataset_subgroup'] == subgroup]
                subgroup_label = str(subgroup)

            if plot_df.empty:
                continue

            # Filter panels that have data in this subgroup slice
            panels = [
                (col, label) for col, label in available_panels
                if col in plot_df.columns and plot_df[col].notna().any()
            ]
            if not panels:
                continue

            # Order methods worst (left) to best (right) by mean global_score on this subset
            if (
                'global_score' in plot_df.columns
                and plot_df['global_score'].notna().any()
            ):
                method_means = plot_df.groupby('method')['global_score'].mean()
                methods_ordered = method_means.sort_values(ascending=True).index.tolist()
            else:
                methods_ordered = sorted(plot_df['method'].unique())
            sub_palette = [method_to_color[m] for m in methods_ordered]

            n_panels = len(panels)
            fig_width = max(12, 4.0 * n_panels)
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(fig_width, 5),
                squeeze=False,
            )
            axes = axes.flatten()

            datasets_in_subgroup = sorted(plot_df['dataset'].unique())
            sub_dataset_palette = [dataset_color_map[d] for d in datasets_in_subgroup]

            for idx, (metric_col, metric_label) in enumerate(panels):
                ax = axes[idx]
                panel_df = plot_df.dropna(subset=[metric_col])

                if panel_df.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10, color='gray')
                    ax.set_title(metric_label, fontsize=11)
                    continue

                # Boxplot by method (order: worst to best by global_score on this subset)
                sns.boxplot(
                    data=panel_df,
                    x='method',
                    y=metric_col,
                    hue='method',
                    ax=ax,
                    palette=sub_palette,
                    width=0.6,
                    showfliers=False,
                    legend=False,
                    order=methods_ordered,
                )

                # Overlay individual dataset points
                sns.stripplot(
                    data=panel_df,
                    x='method',
                    y=metric_col,
                    hue='dataset',
                    ax=ax,
                    palette=dataset_color_map,
                    size=5,
                    alpha=0.7,
                    jitter=True,
                    dodge=False,
                    order=methods_ordered,
                    legend=(idx == n_panels - 1),  # legend only on last panel
                )

                ax.set_title(metric_label, fontsize=11)
                ax.set_xlabel('')
                ax.set_ylabel('Value' if idx == 0 else '', fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)

                # Direction arrow
                if is_lower_better(metric_col):
                    ax.text(0.98, 0.02, '↓', transform=ax.transAxes,
                            fontsize=9, ha='right', color='gray')
                else:
                    ax.text(0.98, 0.02, '↑', transform=ax.transAxes,
                            fontsize=9, ha='right', color='gray')

            # Move the legend from last panel to figure right margin
            last_ax = axes[n_panels - 1]
            legend = last_ax.get_legend()
            if legend is not None:
                legend.set_title('Dataset')
                legend.set_bbox_to_anchor((1.02, 1))
                legend._loc = 2  # upper left

            # Hide unused axes (shouldn't happen but safety)
            for j in range(n_panels, len(axes)):
                axes[j].set_visible(False)

            n_datasets = len(datasets_in_subgroup)
            fig.suptitle(
                f'Batch-Bio Integration: {subgroup_label}\n'
                f'({n_datasets} datasets, {len(methods_ordered)} methods, ordered worst→best by global score)',
                fontsize=13, y=1.04,
            )
            plt.tight_layout()

            filename = f'boxplot_integration_{subgroup_label}.png'
            save_path = self.save_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(save_path)
            logger.info(f"Saved integration summary boxplot to {save_path}")

        return saved_paths

    def _safe_subgroup_dir_name(self, subgroup: str) -> str:
        """Return a filesystem-safe directory name for a subgroup."""
        return str(subgroup).replace('/', '_').replace('\\', '_').replace(' ', '_')

    def generate_per_subgroup_boxplots(
        self,
        results_df: pd.DataFrame,
        save_dir_by_subgroup: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """
        Generate a full set of boxplots for each subgroup (same types as generate_all_boxplots),
        scoped to datasets in that subgroup. Each subgroup gets its own subfolder with:
        global score comparisons, task summary, method summaries, dataset comparison, heatmaps.
        Works with 1 or more subgroups.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            save_dir_by_subgroup: Base directory for subgroup outputs (default: self.save_dir / 'by_subgroup')
        
        Returns:
            List of paths to all saved plot files (across all subgroups)
        """
        if 'dataset_subgroup' not in results_df.columns or results_df['dataset_subgroup'].notna().sum() == 0:
            return []
        base_dir = Path(save_dir_by_subgroup) if save_dir_by_subgroup is not None else self.save_dir / 'by_subgroup'
        base_dir.mkdir(parents=True, exist_ok=True)
        subgroups = sorted(results_df['dataset_subgroup'].dropna().unique())
        all_paths: List[Path] = []
        for subgroup in subgroups:
            sub_df = results_df[results_df['dataset_subgroup'] == subgroup].copy()
            if sub_df.empty:
                continue
            safe_name = self._safe_subgroup_dir_name(subgroup)
            subgroup_plots_dir = base_dir / safe_name / 'plots'
            subgroup_plots_dir.mkdir(parents=True, exist_ok=True)
            subgroup_gen = BoxplotGenerator(
                save_dir=subgroup_plots_dir,
                figsize=self.figsize,
                dpi=self.dpi,
                style=self.style,
            )
            subgroup_plot_dict = subgroup_gen.generate_all_boxplots(sub_df)
            for paths in subgroup_plot_dict.values():
                all_paths.extend(paths)
            logger.info(f"Subgroup '{subgroup}': saved {sum(len(p) for p in subgroup_plot_dict.values())} boxplots to {subgroup_plots_dir}")
        return all_paths

    # =====================================================================
    # Cancer task heatmap (MIL AUPRC across cancer classification tasks)
    # =====================================================================

    # Human-readable task labels for the heatmap y-axis
    CANCER_TASK_LABELS: Dict[str, str] = {
        'TKI_identification': 'Treatment Naive vs TKI Treated',
        'treatment_response': 'T-cell exhaustion',
        'cancer_subtype_classification': 'ER+ vs TNBC',
        'cancer_stage_classification': 'Early vs Late Stage',
        'chemo_identification': 'Treatment Naive vs Neoadjuvant Chemo',
        'pre_post': 'Pre vs Post Treatment',
    }
    # Methods to show in the cancer heatmap (excludes scgpt, stack_small_data, scimilarity)
    CANCER_HEATMAP_METHODS: List[str] = ['scconcept', 'stack', 'state']
    # Methods excluded from all batch_bio_integration plots
    BATCH_BIO_INTEGRATION_EXCLUDED_METHODS: List[str] = ['stack_small_data']

    def plot_cancer_tasks_heatmap(
        self,
        eval_results_dir: Optional[Path] = None,
        metric: str = 'AUPRC',
        train_func: str = 'mil',
    ) -> Optional[Path]:
        """
        Generate a heatmap of classification metric across cancer tasks and methods.

        Scans the eval_results directory for cancer classification tasks,
        reads MIL CV aggregated metrics, and produces a heatmap with:
        - Y-axis: cancer tasks (human-readable labels)
        - X-axis: methods (grouped)
        - Cell values: metric mean (annotated)
        - Top marginal bar: per-method average across tasks
        - Colorbar: viridis (0.5–1.0 range)

        Args:
            eval_results_dir: Path to eval_results root. If None, uses setup_path.OUTPUT_PATH.
            metric: Metric name to extract from mil_cv_metrics_aggregated.csv (default 'AUPRC').
            train_func: Training function to read (default 'mil').

        Returns:
            Path to saved plot, or None if no data found.
        """
        if eval_results_dir is None:
            try:
                from setup_path import OUTPUT_PATH
                eval_results_dir = Path(OUTPUT_PATH)
            except ImportError:
                logger.warning("Cannot import OUTPUT_PATH; provide eval_results_dir explicitly.")
                return None

        eval_results_dir = Path(eval_results_dir)

        # Discover cancer task directories
        cancer_tasks = [
            d.name for d in eval_results_dir.iterdir()
            if d.is_dir() and d.name in self.CANCER_TASK_LABELS
        ]
        if not cancer_tasks:
            logger.info("No cancer task directories found; skipping cancer heatmap.")
            return None

        # Collect data: {(task, method): mean_value}
        records: List[Dict[str, object]] = []
        for task_name in cancer_tasks:
            task_dir = eval_results_dir / task_name
            for method_dir in task_dir.iterdir():
                if not method_dir.is_dir() or method_dir.name.startswith('.') or method_dir.name == 'summaries':
                    continue
                method_name = method_dir.name
                # Find the aggregated CSV (one dataset per task)
                csv_pattern = f"{train_func}_cv_metrics_aggregated.csv"
                csv_files = list(method_dir.rglob(f"**/cv/{csv_pattern}"))
                if not csv_files:
                    continue
                csv_path = csv_files[0]
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    if metric in df.index and 'mean' in df.columns:
                        val = df.loc[metric, 'mean']
                        std = df.loc[metric, 'std'] if 'std' in df.columns else np.nan
                        if pd.notna(val) and str(val).strip() != '':
                            records.append({
                                'task': task_name,
                                'method': method_name,
                                'value': float(val),
                                'std': float(std) if pd.notna(std) and str(std).strip() != '' else np.nan,
                            })
                except Exception as e:
                    logger.warning(f"Error reading {csv_path}: {e}")

        if not records:
            logger.info("No cancer MIL metrics found; skipping cancer heatmap.")
            return None

        data_df = pd.DataFrame(records)

        # Build pivot table (tasks x methods)
        pivot = data_df.pivot_table(index='task', columns='method', values='value', aggfunc='first')

        # Order tasks by the canonical list (top = easiest)
        task_order = [t for t in self.CANCER_TASK_LABELS if t in pivot.index]
        pivot = pivot.reindex(task_order)

        # Restrict to selected methods (scconcept, scimilarity, stack, state; exclude scgpt, stack_small_data)
        # Match case-insensitively so that e.g. scConcept (dir name) matches 'scconcept'
        method_order = []
        for desired in self.CANCER_HEATMAP_METHODS:
            for col in pivot.columns:
                if col.lower() == desired.lower():
                    method_order.append(col)
                    break
        if not method_order:
            logger.warning("No cancer heatmap methods found in data; skipping cancer heatmap.")
            return None
        pivot = pivot[method_order]

        # Readable labels
        task_labels = [self.CANCER_TASK_LABELS.get(t, t) for t in task_order]

        # Per-method average (for top marginal bar) and per-task average (for right marginal bar)
        method_avg = pivot.mean(axis=0, skipna=True)
        task_avg = pivot.mean(axis=1, skipna=True)

        # ---- Figure layout: top bar + heatmap, right task-avg bar ----
        n_tasks = len(task_order)
        n_methods = len(method_order)

        fig_width = max(8, 1.2 * n_methods + 3.5)
        fig_height = max(4, 0.7 * n_tasks + 3)

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1, max(3, n_tasks)],
            width_ratios=[max(6, n_methods), 1.0],
            hspace=0.08, wspace=0.12,
        )

        ax_bar = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_task_bar = fig.add_subplot(gs[1, 1])  # Right: avg AUPRC per task

        # Colormap and range
        vmin, vmax = 0.5, 1.0
        cmap = plt.cm.viridis

        # --- Top marginal bar (Avg metric per method) ---
        bar_colors = [cmap((v - vmin) / (vmax - vmin)) if pd.notna(v) else '#cccccc' for v in method_avg.values]
        bars = ax_bar.bar(range(n_methods), method_avg.values, color=bar_colors, width=0.7, edgecolor='white', linewidth=0.5)
        for i, (bar, val) in enumerate(zip(bars, method_avg.values)):
            if pd.notna(val):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontdict={'size': 18, 'weight': 'normal'})
        ax_bar.set_xlim(-0.5, n_methods - 0.5)
        ax_bar.set_ylim(0, min(1.15, method_avg.max() + 0.08) if method_avg.notna().any() else 1.15)
        ax_bar.set_xticks([])
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['bottom'].set_visible(False)
        ax_bar.set_ylabel(f'Avg {metric}', fontsize=9)
        ax_bar.tick_params(axis='y', labelsize=8)

        # --- Heatmap ---
        heatmap_data = pivot.values.astype(float)
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_methods):
                val = heatmap_data[i, j]
                if np.isnan(val):
                    ax_heat.text(j, i, '—', ha='center', va='center', fontdict={'size': 18, 'weight': 'normal'}, color='gray')
                else:
                    text_color = 'white' if val < 0.7 else 'black'
                    ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                                 fontdict={'size': 18, 'weight': 'normal'}, color=text_color)

        ax_heat.set_xticks(range(n_methods))
        ax_heat.set_xticklabels(method_order, rotation=90, ha='center', va='top', fontsize=14)
        ax_heat.set_yticks(range(n_tasks))
        ax_heat.set_yticklabels(task_labels, fontsize=9)
        ax_heat.grid(False)

        # --- Right marginal bar (Avg metric per task) ---
        task_bar_colors = [cmap((v - vmin) / (vmax - vmin)) if pd.notna(v) else '#cccccc' for v in task_avg.values]
        bars_rt = ax_task_bar.barh(range(n_tasks), task_avg.values, color=task_bar_colors, height=0.7, edgecolor='white', linewidth=0.5)
        for i, (bar, val) in enumerate(zip(bars_rt, task_avg.values)):
            if pd.notna(val):
                ax_task_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                                 f'{val:.2f}', ha='left', va='center', fontdict={'size': 18, 'weight': 'normal'})
        ax_task_bar.set_ylim(n_tasks - 0.5, -0.5)  # Match heatmap y order
        ax_task_bar.set_xlim(0, min(1.15, task_avg.max() + 0.1) if task_avg.notna().any() else 1.15)
        ax_task_bar.set_yticks([])
        ax_task_bar.set_xlabel(f'Avg {metric}', fontsize=9)
        ax_task_bar.spines['top'].set_visible(False)
        ax_task_bar.spines['right'].set_visible(False)
        ax_task_bar.spines['left'].set_visible(False)
        ax_task_bar.tick_params(axis='x', labelsize=8)

        fig.suptitle(f'Cancer Classification Tasks — {train_func.upper()} {metric}',
                     fontsize=13, fontweight='bold', y=0.98)

        # Save
        plots_dir = self.save_dir / 'cancer'
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f'heatmap_cancer_tasks_{train_func}_{metric}.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved cancer tasks heatmap: {save_path}")
        return save_path

    def generate_all_boxplots(self, results_df: pd.DataFrame) -> Dict[str, List[Path]]:
        """
        Generate all boxplot visualizations from results.
        
        Creates a comprehensive set of boxplots:
        1. Global score comparisons (by method, by task)
        2. Per-task method comparisons
        3. Per-method dataset comparisons  
        4. Dataset comparison plots
        5. Heatmap with marginal boxplots
        
        Args:
            results_df: DataFrame from scan_all_experiments()
        
        Returns:
            Dictionary mapping plot type to list of saved paths
        """
        if results_df.empty:
            logger.warning("No results data for boxplot generation")
            return {}
        
        all_plots: Dict[str, List[Path]] = {
            'global_score': [],
            'task_summaries': [],
            'method_summaries': [],
            'dataset_comparisons': [],
            'heatmaps': [],
            'subgroup_comparisons': [],
            'integration_summaries': [],
        }

        # Exclude selected methods from batch_bio_integration for all plots
        if 'batch_bio_integration' in results_df['task'].values:
            mask = ~(
                (results_df['task'] == 'batch_bio_integration')
                & (results_df['method'].isin(self.BATCH_BIO_INTEGRATION_EXCLUDED_METHODS))
            )
            results_df = results_df.loc[mask].copy()

        # 1. Global score comparisons
        logger.info("Generating global score comparison boxplots...")
        
        # By method, faceted by task
        path = self.plot_global_score_comparison(results_df, group_by='method', facet_by='task')
        if path:
            all_plots['global_score'].append(path)
        
        # By method, overall
        path = self.plot_global_score_comparison(results_df, group_by='method', facet_by=None)
        if path:
            all_plots['global_score'].append(path)
        
        # By task, overall
        path = self.plot_global_score_comparison(results_df, group_by='task', facet_by=None)
        if path:
            all_plots['global_score'].append(path)
        
        # 2. Task-level summaries
        logger.info("Generating task summary boxplots...")
        for task in results_df['task'].unique():
            paths = self.plot_task_summary_boxplots(results_df, task)
            all_plots['task_summaries'].extend(paths)
        
        # 3. Method-level summaries
        logger.info("Generating method summary boxplots...")
        for method in results_df['method'].unique():
            # Overall (across all tasks)
            paths = self.plot_method_summary_boxplots(results_df, method)
            all_plots['method_summaries'].extend(paths)
            
            # Per-task
            for task in results_df['task'].unique():
                task_method_df = results_df[(results_df['method'] == method) & (results_df['task'] == task)]
                if not task_method_df.empty and len(task_method_df) > 1:
                    paths = self.plot_method_summary_boxplots(results_df, method, task_name=task)
                    all_plots['method_summaries'].extend(paths)
        
        # 4. Dataset comparisons
        logger.info("Generating dataset comparison boxplots...")
        path = self.plot_dataset_comparison_boxplots(results_df, metric='global_score')
        if path:
            all_plots['dataset_comparisons'].append(path)
        
        # 5. Heatmaps
        logger.info("Generating heatmap with boxplots...")
        path = self.plot_metric_heatmap_with_boxplots(results_df, aggregate_by='method')
        if path:
            all_plots['heatmaps'].append(path)
        
        path = self.plot_metric_heatmap_with_boxplots(results_df, aggregate_by='task')
        if path:
            all_plots['heatmaps'].append(path)
        
        # 6. Subgroup comparisons (if dataset_subgroup exists)
        has_subgroups = 'dataset_subgroup' in results_df.columns and results_df['dataset_subgroup'].notna().any()
        if has_subgroups:
            logger.info("Generating subgroup comparison boxplots...")
            
            # Global score by subgroup
            path = self.plot_global_score_by_subgroup(results_df)
            if path:
                all_plots['subgroup_comparisons'].append(path)
            
            # Task summaries faceted by subgroup
            for task in results_df['task'].unique():
                paths = self.plot_task_summary_by_subgroup(results_df, task)
                all_plots['subgroup_comparisons'].extend(paths)
            
            # Method subgroup comparisons
            for method in results_df['method'].unique():
                paths = self.plot_method_subgroup_comparison(results_df, method)
                all_plots['subgroup_comparisons'].extend(paths)
                
                # Also per-task
                for task in results_df['task'].unique():
                    task_method_df = results_df[
                        (results_df['method'] == method) & 
                        (results_df['task'] == task) & 
                        results_df['dataset_subgroup'].notna()
                    ]
                    if not task_method_df.empty and len(task_method_df['dataset_subgroup'].unique()) > 1:
                        paths = self.plot_method_subgroup_comparison(results_df, method, task_name=task)
                        all_plots['subgroup_comparisons'].extend(paths)
        
        # 7. Batch-bio integration combined summary (if applicable)
        if 'batch_bio_integration' in results_df['task'].unique():
            logger.info("Generating batch-bio integration summary boxplots...")
            paths = self.plot_batch_bio_integration_summary(results_df)
            all_plots['integration_summaries'].extend(paths)

        # 8. Cancer tasks heatmap (MIL AUPRC across cancer classification tasks)
        logger.info("Generating cancer tasks heatmap...")
        cancer_path = self.plot_cancer_tasks_heatmap()
        if cancer_path:
            all_plots.setdefault('cancer_heatmaps', []).append(cancer_path)

        # Summary
        total_plots = sum(len(v) for v in all_plots.values())
        logger.info(f"Generated {total_plots} boxplot visualizations")
        for plot_type, paths in all_plots.items():
            if paths:
                logger.info(f"  {plot_type}: {len(paths)} plots")
        
        return all_plots
