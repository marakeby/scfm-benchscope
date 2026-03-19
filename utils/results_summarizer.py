"""
Results summarizer module.

Generates comprehensive comparison tables across methods, tasks, datasets, and metrics.

**Metric values in tables and plots:** All metric columns (e.g. ASW_batch, iLISI, cLISI)
are kept in their **original range and direction** as stored by the evaluators.
Direction (lower vs higher is better) is defined in utils/metric_definitions.py and
used for global_score normalization and for plot labels only; raw values are never
transformed for display.

Summarizer Structure:
--------------------
The summarizer operates at three different scoping levels:

1. **Method-level** (`scope='method'`):
   - Scans: `task/method/dataset/`
   - Purpose: Compare one method across multiple datasets for a specific task
   - Example: How does scimilarity perform on dkd, gtex_v9, tabula_sapiens for biological_signal_preservation?
   - Output: `task/method/summaries/`

2. **Task-level** (`scope='task'`):
   - Scans: `task/method/dataset/`
   - Purpose: Compare multiple methods across datasets for one task
   - Example: How do scimilarity and scConcept compare for biological_signal_preservation across all datasets?
   - Output: `task/summaries/`

3. **Root-level** (`scope='root'`):
   - Scans: `task/method/dataset/`
   - Purpose: Compare all methods, tasks, and datasets
   - Example: Overall comparison across all experiments
   - Output: `eval_results/summaries/`

Global Score Computation:
------------------------
Only the composite global_score is normalized (for ranking). Each metric is normalized
to a 0-1 scale where higher is always better, using utils/metric_definitions (direction
and known range); then global_score = mean(normalized metrics). Table and boxplot
values remain in original range and direction.
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from utils.logs_ import get_logger
from utils.metric_definitions import is_lower_better, get_metric_range
from utils.metric_collector import MetricCollector
from setup_path import OUTPUT_PATH
from utils.boxplot_generator import BoxplotGenerator
from utils.confusion_matrix_generator import ConfusionMatrixGenerator

logger = get_logger()


class ResultsSummarizer:
    """
    Generates comparison tables and summaries across methods, tasks, datasets, and metrics.
    
    Supports different scoping levels:
    - 'root': Scans OUTPUT_PATH / task / method / dataset (full hierarchy)
    - 'task': Scans task_dir / method / dataset (single task)
    - 'method': Scans method_dir / dataset (single method within a task)
    
    Attributes:
        output_base_dir: Base directory for scanning and saving results
        scope: Scanning scope ('root', 'task', 'method', or 'auto')
        metric_collector: MetricCollector instance for gathering metrics
    """
    
    def __init__(self, output_base_dir: str, scope: str = 'auto'):
        """
        Initialize results summarizer.
        
        Args:
            output_base_dir: Base output directory to scan from.
            scope: Scanning scope - 'root', 'task', 'method', 'method_subgroup', or 'auto'.
                   - 'root': Expects task/method/dataset (or task/method/subgroup/dataset)
                   - 'task': Expects method/dataset (or method/subgroup/dataset)
                   - 'method': Expects dataset structure (task/method inferred from path)
                   - 'method_subgroup': Expects dataset structure under task/method/subgroup
                   - 'auto': Automatically detect based on directory structure
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine scope
        if scope == 'auto':
            self.scope = self._detect_scope()
        else:
            self.scope = scope
        
        # Initialize metric collector
        self.metric_collector = MetricCollector()
    
    def _detect_scope(self) -> str:
        """
        Auto-detect the scanning scope based on directory structure.
        
        Looks for 'metrics' directories at different depths to determine the scope:
        - If metrics found at depth 3 (task/method/dataset/metrics): 'root' scope
        - If metrics found at depth 2 (method/dataset/metrics): 'task' scope  
        - If metrics found at depth 1 (dataset/metrics): 'method' scope
        
        Returns:
            Detected scope: 'root', 'task', or 'method'
        """
        excluded_dirs = {'summaries', 'embeddings', 'config', 'logs', 'plots', 'metrics'}

        # Check for metrics at different depths
        # Depth 1: method scope (dataset/metrics) or method_subgroup (task/method/subgroup/dataset)
        try:
            is_under_task = (self.output_base_dir.resolve().parent.parent.parent == Path(OUTPUT_PATH).resolve())
        except (ValueError, IndexError):
            is_under_task = False
        for child in self.output_base_dir.iterdir():
            if child.is_dir() and not child.name.startswith('.') and child.name not in excluded_dirs:
                metrics_dir = child / 'metrics'
                if metrics_dir.exists():
                    if is_under_task:
                        logger.info(f"Detected 'method_subgroup' scope (task/method/subgroup/dataset/metrics found)")
                        return 'method_subgroup'
                    logger.info(f"Detected 'method' scope (dataset/metrics found)")
                    return 'method'
        
        # Depth 2: task scope (method/dataset/metrics)
        for child in self.output_base_dir.iterdir():
            if child.is_dir() and not child.name.startswith('.') and child.name not in excluded_dirs:
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and not grandchild.name.startswith('.') and grandchild.name not in excluded_dirs:
                        metrics_dir = grandchild / 'metrics'
                        if metrics_dir.exists():
                            logger.info(f"Detected 'task' scope (method/dataset/metrics found)")
                            return 'task'
        
        # Depth 3: task scope with hierarchy (method/subgroup/dataset/metrics) when base is a task dir
        if self.output_base_dir.resolve().parent == Path(OUTPUT_PATH).resolve():
            for child in self.output_base_dir.iterdir():
                if child.is_dir() and not child.name.startswith('.') and child.name not in excluded_dirs:
                    for grandchild in child.iterdir():
                        if grandchild.is_dir() and not grandchild.name.startswith('.') and grandchild.name not in excluded_dirs:
                            for great in grandchild.iterdir():
                                if great.is_dir() and not great.name.startswith('.') and great.name not in excluded_dirs:
                                    if (great / 'metrics').exists():
                                        logger.info(f"Detected 'task' scope (method/subgroup/dataset/metrics found)")
                                        return 'task'
        
        # Default to root scope (handles both task/method/dataset and task/method/subgroup/dataset)
        logger.info(f"Detected 'root' scope (default)")
        return 'root'
    
    def collect_metrics_from_experiment(
        self, 
        experiment_dir: Path,
        dataset_name: str,
        task_name: str,
        method_name: str
    ) -> Dict[str, float]:
        """
        Collect all metrics from a single experiment directory.
        
        Delegates to MetricCollector.
        
        Args:
            experiment_dir: Path to experiment directory containing metrics/
            dataset_name: Name of the dataset
            task_name: Name of the task
            method_name: Name of the method
            
        Returns:
            Dictionary of metric_name: value (flattened from all metric files)
        """
        return self.metric_collector.collect_metrics_from_experiment(
            experiment_dir, dataset_name, task_name, method_name
        )
    
    def normalize_metric(
        self,
        value: float,
        metric_name: str,
        all_values: Optional[np.ndarray] = None
    ) -> float:
        """
        Normalize a metric value to 0-1 scale where higher is always better.
        
        Args:
            value: The metric value to normalize
            metric_name: Name of the metric (for determining directionality and range)
            all_values: Optional array of all values for this metric (for min-max normalization)
        
        Returns:
            Normalized value in [0, 1] where higher is better
        """
        if pd.isna(value):
            return np.nan
        
        # Get known range or compute from data
        known_range = get_metric_range(metric_name)
        
        if known_range is not None:
            min_val, max_val = known_range
        elif all_values is not None and len(all_values) > 0:
            min_val = np.nanmin(all_values)
            max_val = np.nanmax(all_values)
        else:
            # Fallback: assume reasonable range based on value
            if value < 0:
                min_val, max_val = -1.0, 1.0
            elif value > 1:
                min_val, max_val = 0.0, max(10.0, value * 1.5)
            else:
                min_val, max_val = 0.0, 1.0
        
        # Normalize to 0-1
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5  # All values are the same
        
        # Invert if lower is better
        if is_lower_better(metric_name):
            normalized = 1.0 - normalized
        
        # Clamp to [0, 1]
        return float(np.clip(normalized, 0.0, 1.0))
    
    def compute_global_metric(
        self,
        metrics_dict: Dict[str, float],
        all_metrics_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute a global normalized metric across all metrics for a single experiment.
        
        This function:
        1. Normalizes all metrics to 0-1 scale (handling directionality)
        2. Computes mean and std across all normalized metrics
        3. Returns a global score that can be compared across tasks/methods/datasets
        
        Args:
            metrics_dict: Dictionary of metric_name: value for one experiment
            all_metrics_df: Optional DataFrame with all experiments (for min-max normalization)
        
        Returns:
            Dictionary with:
            - 'global_score': Mean of normalized metrics (0-1, higher is better)
            - 'global_score_std': Std of normalized metrics (lower = more consistent)
            - 'n_metrics': Number of metrics used
            - 'normalized_metrics': Dict of individual normalized metric values
        """
        if not metrics_dict:
            return {
                'global_score': np.nan, 
                'global_score_std': np.nan, 
                'n_metrics': 0,
                'normalized_metrics': {}
            }
        
        normalized_metrics: Dict[str, float] = {}
        
        for metric_name, value in metrics_dict.items():
            if pd.isna(value):
                continue
            
            # Get all values for this metric for min-max normalization
            all_values = None
            if all_metrics_df is not None and metric_name in all_metrics_df.columns:
                all_values = all_metrics_df[metric_name].dropna().values
            
            normalized = self.normalize_metric(value, metric_name, all_values)
            
            if not np.isnan(normalized):
                normalized_metrics[metric_name] = normalized
        
        if len(normalized_metrics) == 0:
            return {
                'global_score': np.nan, 
                'global_score_std': np.nan, 
                'n_metrics': 0,
                'normalized_metrics': {}
            }
        
        normalized_array = np.array(list(normalized_metrics.values()))
        
        return {
            'global_score': float(np.mean(normalized_array)),
            'global_score_std': float(np.std(normalized_array)),
            'n_metrics': len(normalized_metrics),
            'normalized_metrics': normalized_metrics
        }
    
    def _iter_method_datasets(
        self,
        method_dir: Path,
        task_name: str,
        method_name: str,
        excluded_dirs: set,
    ):
        """
        Yield (experiment_dir, dataset_name, dataset_subgroup) for all datasets under a method dir.
        Supports both flat (method/dataset/) and hierarchical (method/subgroup/dataset/) layout.
        """
        for child in method_dir.iterdir():
            if not child.is_dir() or child.name.startswith('.') or child.name in excluded_dirs:
                continue
            if (child / 'metrics').exists():
                yield child, child.name, None
            else:
                for grandchild in child.iterdir():
                    if not grandchild.is_dir() or grandchild.name.startswith('.') or grandchild.name in excluded_dirs:
                        continue
                    if (grandchild / 'metrics').exists():
                        yield grandchild, grandchild.name, child.name

    def scan_all_experiments(self) -> pd.DataFrame:
        """
        Scan experiment directories and collect metrics based on the detected scope.
        
        Scope determines the directory structure:
        - 'root': OUTPUT_PATH / task / method / dataset / or task / method / subgroup / dataset /
        - 'task': task_dir / method / dataset / (or method / subgroup / dataset /)
        - 'method': method_dir / dataset / (or subgroup / dataset /)
        
        Returns:
            DataFrame with columns: dataset, task, method, dataset_subgroup (if any), and all metric columns
        """
        all_results: List[Dict] = []
        excluded_dirs = {'summaries', 'embeddings', 'config', 'logs', 'plots', 'metrics'}

        def append_result(dataset_name: str, task_name: str, method_name: str, experiment_dir: Path, dataset_subgroup: Optional[str] = None) -> None:
            metrics = self.collect_metrics_from_experiment(
                experiment_dir, dataset_name, task_name, method_name
            )
            if metrics:
                row: Dict = {
                    'dataset': dataset_name,
                    'task': task_name,
                    'method': method_name,
                    **metrics
                }
                if dataset_subgroup is not None:
                    row['dataset_subgroup'] = dataset_subgroup
                all_results.append(row)
        
        if self.scope == 'method':
            method_name = self.output_base_dir.name
            task_name = self.output_base_dir.parent.name
            for experiment_dir, dataset_name, dataset_subgroup in self._iter_method_datasets(
                self.output_base_dir, task_name, method_name, excluded_dirs
            ):
                append_result(dataset_name, task_name, method_name, experiment_dir, dataset_subgroup)

        elif self.scope == 'method_subgroup':
            # output_base_dir is task/method/subgroup; children are dataset dirs
            task_name = self.output_base_dir.parent.parent.name
            method_name = self.output_base_dir.parent.name
            subgroup_name = self.output_base_dir.name
            for dataset_dir in self.output_base_dir.iterdir():
                if not dataset_dir.is_dir() or dataset_dir.name.startswith('.') or dataset_dir.name in excluded_dirs:
                    continue
                if (dataset_dir / 'metrics').exists():
                    append_result(
                        dataset_dir.name, task_name, method_name, dataset_dir, dataset_subgroup=subgroup_name
                    )

        elif self.scope == 'task':
            task_name = self.output_base_dir.name
            for method_dir in self.output_base_dir.iterdir():
                if not method_dir.is_dir() or method_dir.name.startswith('.') or method_dir.name in excluded_dirs:
                    continue
                method_name = method_dir.name
                for experiment_dir, dataset_name, dataset_subgroup in self._iter_method_datasets(
                    method_dir, task_name, method_name, excluded_dirs
                ):
                    append_result(dataset_name, task_name, method_name, experiment_dir, dataset_subgroup)

        else:
            for task_dir in self.output_base_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith('.') or task_dir.name in excluded_dirs:
                    continue
                task_name = task_dir.name
                for method_dir in task_dir.iterdir():
                    if not method_dir.is_dir() or method_dir.name.startswith('.') or method_dir.name in excluded_dirs:
                        continue
                    method_name = method_dir.name
                    for experiment_dir, dataset_name, dataset_subgroup in self._iter_method_datasets(
                        method_dir, task_name, method_name, excluded_dirs
                    ):
                        append_result(dataset_name, task_name, method_name, experiment_dir, dataset_subgroup)
        
        if not all_results:
            logger.warning("No experiment results found!")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_results)
        
        # Compute global metrics for each row (exclude _std and cv_label_splits - for plotting only)
        metric_columns = [
            col for col in results_df.columns
            if col not in ['dataset', 'task', 'method', 'dataset_subgroup']
            and not col.endswith('_std') and col != 'cv_label_splits'
        ]
        global_scores = []
        for idx, row in results_df.iterrows():
            metrics_dict = {col: row[col] for col in metric_columns if col in row and pd.notna(row[col])}
            global_metric = self.compute_global_metric(metrics_dict, results_df[metric_columns])
            global_scores.append(global_metric)
        
        # Add global metrics to dataframe
        results_df['global_score'] = [g['global_score'] for g in global_scores]
        results_df['global_score_std'] = [g['global_score_std'] for g in global_scores]
        results_df['n_metrics'] = [g['n_metrics'] for g in global_scores]
        
        return results_df
    
    def create_method_summary(
        self,
        results_df: pd.DataFrame,
        method_name: str,
        task_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create summary for one method across multiple datasets.
        
        Purpose: "How does this method perform across different datasets?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            method_name: Method to summarize
            task_name: Optional task filter
        
        Returns:
            DataFrame with datasets as rows, metrics as columns
        """
        # Filter by method and optionally task
        mask = results_df['method'] == method_name
        if task_name is not None:
            mask &= results_df['task'] == task_name
        
        method_df = results_df[mask].copy()
        
        if method_df.empty:
            logger.warning(f"No results found for method={method_name}, task={task_name}")
            return pd.DataFrame()
        
        # Set dataset as index
        method_df = method_df.set_index('dataset')
        
        # Drop non-metric columns (cv_label_splits is string, for boxplot header only)
        drop_cols = ['task', 'method', 'dataset_subgroup', 'cv_label_splits']
        method_df = method_df.drop(columns=[c for c in drop_cols if c in method_df.columns])
        # Add summary statistics row (only numeric columns)
        numeric_cols = [c for c in method_df.columns if pd.api.types.is_numeric_dtype(method_df[c])]
        summary_row = pd.DataFrame({
            col: [method_df[col].mean(), method_df[col].std()]
            for col in numeric_cols
        }, index=['MEAN', 'STD'])
        
        method_df = pd.concat([method_df, summary_row])
        
        return method_df
    
    def create_task_summary(
        self,
        results_df: pd.DataFrame,
        task_name: str,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Create summary for one task across multiple methods and datasets.
        
        Purpose: "How do different methods compare for this task?"
        Aggregates across datasets to show mean performance per method.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            task_name: Task to summarize
            aggregation: How to aggregate across datasets ('mean', 'median', 'std')
        
        Returns:
            DataFrame with methods as rows, metrics (aggregated across datasets) as columns
        """
        task_df = results_df[results_df['task'] == task_name].copy()
        
        if task_df.empty:
            logger.warning(f"No results found for task={task_name}")
            return pd.DataFrame()
        
        # Get metric columns
        exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
        metric_cols = [c for c in task_df.columns if c not in exclude_cols and not c.endswith('_std') and c != 'cv_label_splits']
        # Group by method and aggregate across datasets
        agg_func = aggregation if aggregation in ['mean', 'median', 'std', 'min', 'max'] else 'mean'
        
        summary_rows = []
        for method in task_df['method'].unique():
            method_data = task_df[task_df['method'] == method]
            row = {'method': method, 'n_datasets': len(method_data)}
            
            # Global score aggregation
            global_vals = method_data['global_score'].dropna()
            if len(global_vals) > 0:
                row['global_score_mean'] = global_vals.mean()
                row['global_score_std'] = global_vals.std()
                row['global_score_min'] = global_vals.min()
                row['global_score_max'] = global_vals.max()
            
            # Individual metric aggregation
            for metric in metric_cols:
                vals = method_data[metric].dropna()
                if len(vals) > 0:
                    if agg_func == 'mean':
                        row[f'{metric}'] = vals.mean()
                    elif agg_func == 'median':
                        row[f'{metric}'] = vals.median()
                    elif agg_func == 'std':
                        row[f'{metric}'] = vals.std()
                    elif agg_func == 'min':
                        row[f'{metric}'] = vals.min()
                    elif agg_func == 'max':
                        row[f'{metric}'] = vals.max()
                    
                    # Also store std for reference
                    row[f'{metric}_std'] = vals.std()
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Sort by global_score_mean descending (best first)
        if 'global_score_mean' in summary_df.columns:
            summary_df = summary_df.sort_values('global_score_mean', ascending=False)
        
        return summary_df
    
    def create_comprehensive_comparison_table(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive comparison table: Methods × (Tasks × Datasets × Metrics).
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include (if None, includes all)
            
        Returns:
            DataFrame with methods as rows, task×dataset×metric as columns
        """
        if results_df.empty:
            return pd.DataFrame()
        
        # Get all metric columns (exclude dataset, task, method, and global metrics)
        if metric_columns is None:
            exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
            metric_columns = [col for col in results_df.columns if col not in exclude_cols and not col.endswith('_std') and col != 'cv_label_splits']
        # Create wide format: method × (task_dataset_metric)
        comparison_rows = []
        
        for method in sorted(results_df['method'].unique()):
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            # Create columns: task_dataset_metric
            for task in sorted(results_df['task'].unique()):
                task_df = method_df[method_df['task'] == task]
                for dataset in sorted(results_df['dataset'].unique()):
                    dataset_df = task_df[task_df['dataset'] == dataset]
                    
                    prefix = f'{task}_{dataset}'
                    
                    if not dataset_df.empty:
                        # Add global_score
                        row[f'{prefix}_global_score'] = dataset_df['global_score'].iloc[0]
                        
                        # Add individual metrics
                        for metric in metric_columns:
                            if metric in dataset_df.columns:
                                row[f'{prefix}_{metric}'] = dataset_df[metric].iloc[0]
                            else:
                                row[f'{prefix}_{metric}'] = np.nan
                    else:
                        row[f'{prefix}_global_score'] = np.nan
                        for metric in metric_columns:
                            row[f'{prefix}_{metric}'] = np.nan
            
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def create_distribution_summary(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create distribution summary: Methods × Tasks with metric distributions across datasets.
        
        Shows: mean ± std (min, max) [n_datasets] for each method-task combination.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include
            
        Returns:
            DataFrame with methods as rows, task_metric as columns showing distributions
        """
        if results_df.empty:
            return pd.DataFrame()
        
        if metric_columns is None:
            exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
            metric_columns = [col for col in results_df.columns if col not in exclude_cols and not col.endswith('_std') and col != 'cv_label_splits']
        distribution_rows = []
        for method in sorted(results_df['method'].unique()):
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            for task in sorted(results_df['task'].unique()):
                task_df = method_df[method_df['task'] == task]
                
                # Global score distribution
                global_values = task_df['global_score'].dropna()
                if len(global_values) > 0:
                    mean_val = global_values.mean()
                    std_val = global_values.std() if len(global_values) > 1 else 0.0
                    min_val = global_values.min()
                    max_val = global_values.max()
                    n_datasets = len(global_values)
                    row[f'{task}_global_score'] = f'{mean_val:.4f}±{std_val:.4f} ({min_val:.4f}, {max_val:.4f}) [n={n_datasets}]'
                else:
                    row[f'{task}_global_score'] = 'N/A'
                
                # Individual metrics
                for metric in metric_columns:
                    if metric in task_df.columns:
                        values = task_df[metric].dropna()
                        if len(values) > 0:
                            mean_val = values.mean()
                            std_val = values.std() if len(values) > 1 else 0.0
                            min_val = values.min()
                            max_val = values.max()
                            n_datasets = len(values)
                            row[f'{task}_{metric}'] = f'{mean_val:.4f}±{std_val:.4f} ({min_val:.4f}, {max_val:.4f}) [n={n_datasets}]'
                        else:
                            row[f'{task}_{metric}'] = 'N/A'
                    else:
                        row[f'{task}_{metric}'] = 'N/A'
            
            distribution_rows.append(row)
        
        return pd.DataFrame(distribution_rows)
    
    def create_task_aggregated_table(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Create table with methods × tasks, aggregating across datasets.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include
            aggregation: Aggregation method ('mean', 'median', 'std', 'min', 'max')
            
        Returns:
            DataFrame with methods as rows, task_metric as columns
        """
        if results_df.empty:
            return pd.DataFrame()
        
        if metric_columns is None:
            exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
            metric_columns = [col for col in results_df.columns if col not in exclude_cols and not col.endswith('_std') and col != 'cv_label_splits']
        agg_funcs = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
        }
        agg_func = agg_funcs.get(aggregation, np.mean)
        
        aggregated_rows = []
        
        for method in sorted(results_df['method'].unique()):
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            for task in sorted(results_df['task'].unique()):
                task_df = method_df[method_df['task'] == task]
                
                # Global score aggregation
                global_values = task_df['global_score'].dropna()
                row[f'{task}_global_score'] = agg_func(global_values) if len(global_values) > 0 else np.nan
                row[f'{task}_n_datasets'] = len(global_values)
                
                # Individual metrics
                for metric in metric_columns:
                    if metric in task_df.columns:
                        values = task_df[metric].dropna()
                        row[f'{task}_{metric}'] = agg_func(values) if len(values) > 0 else np.nan
                    else:
                        row[f'{task}_{metric}'] = np.nan
            
            aggregated_rows.append(row)
        
        return pd.DataFrame(aggregated_rows)
    
    def create_subgroup_comparison_table(
        self,
        results_df: pd.DataFrame,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Create comparison table: Methods × Subgroups (aggregated across datasets within each subgroup).
        
        Purpose: "How do methods compare across dataset subgroups?"
        Only applicable when dataset_subgroup column exists.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            aggregation: How to aggregate across datasets ('mean', 'median', 'std')
        
        Returns:
            DataFrame with methods as rows, subgroup_metric as columns
        """
        if results_df.empty or 'dataset_subgroup' not in results_df.columns:
            return pd.DataFrame()
        
        # Filter to rows with subgroups
        subgroup_df = results_df[results_df['dataset_subgroup'].notna()].copy()
        if subgroup_df.empty:
            return pd.DataFrame()
        
        exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
        metric_cols = [c for c in subgroup_df.columns if c not in exclude_cols and not c.endswith('_std') and c != 'cv_label_splits']
        agg_funcs = {'mean': np.mean, 'median': np.median, 'std': np.std}
        agg_func = agg_funcs.get(aggregation, np.mean)
        comparison_rows = []
        for method in sorted(subgroup_df['method'].unique()):
            method_data = subgroup_df[subgroup_df['method'] == method]
            row = {'method': method}
            
            for subgroup in sorted(subgroup_df['dataset_subgroup'].unique()):
                subgroup_data = method_data[method_data['dataset_subgroup'] == subgroup]
                prefix = subgroup
                
                row[f'{prefix}_n_datasets'] = len(subgroup_data)
                
                # Global score
                global_vals = subgroup_data['global_score'].dropna()
                if len(global_vals) > 0:
                    row[f'{prefix}_global_score'] = agg_func(global_vals)
                    row[f'{prefix}_global_score_std'] = global_vals.std() if len(global_vals) > 1 else 0.0
                else:
                    row[f'{prefix}_global_score'] = np.nan
                    row[f'{prefix}_global_score_std'] = np.nan
                
                # Individual metrics
                for metric in metric_cols:
                    if metric in subgroup_data.columns:
                        vals = subgroup_data[metric].dropna()
                        row[f'{prefix}_{metric}'] = agg_func(vals) if len(vals) > 0 else np.nan
                    else:
                        row[f'{prefix}_{metric}'] = np.nan
            
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def create_per_subgroup_summary(
        self,
        results_df: pd.DataFrame,
        subgroup_name: str,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Create summary for a specific subgroup: Methods × Metrics (aggregated across datasets in this subgroup).
        
        Purpose: "How do methods compare within this specific subgroup (e.g., HBio_HTech)?"
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            subgroup_name: Name of the subgroup to summarize
            aggregation: How to aggregate across datasets
        
        Returns:
            DataFrame with methods as rows, metrics as columns
        """
        if results_df.empty or 'dataset_subgroup' not in results_df.columns:
            return pd.DataFrame()
        
        subgroup_df = results_df[results_df['dataset_subgroup'] == subgroup_name].copy()
        if subgroup_df.empty:
            return pd.DataFrame()
        
        exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score_std', 'n_metrics'}
        metric_cols = ['global_score'] + [
            c for c in subgroup_df.columns
            if c not in exclude_cols and c != 'global_score' and not c.endswith('_std') and c != 'cv_label_splits'
        ]
        agg_funcs = {'mean': np.mean, 'median': np.median, 'std': np.std}
        agg_func = agg_funcs.get(aggregation, np.mean)
        
        summary_rows = []
        for method in sorted(subgroup_df['method'].unique()):
            method_data = subgroup_df[subgroup_df['method'] == method]
            row = {'method': method, 'n_datasets': len(method_data)}
            
            for metric in metric_cols:
                if metric in method_data.columns:
                    vals = method_data[metric].dropna()
                    if len(vals) > 0:
                        row[metric] = agg_func(vals)
                        row[f'{metric}_std'] = vals.std() if len(vals) > 1 else 0.0
                    else:
                        row[metric] = np.nan
                        row[f'{metric}_std'] = np.nan
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        if 'global_score' in summary_df.columns:
            summary_df = summary_df.sort_values('global_score', ascending=False)
        
        return summary_df

    def create_ranking_table(
        self,
        results_df: pd.DataFrame,
        rank_by: str = 'global_score'
    ) -> pd.DataFrame:
        """
        Create ranking table showing method rankings per task.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            rank_by: Metric to rank by ('global_score' or specific metric name)
        
        Returns:
            DataFrame with tasks as columns, methods ranked within each task
        """
        if results_df.empty:
            return pd.DataFrame()
        
        ranking_data = {}
        
        for task in sorted(results_df['task'].unique()):
            task_df = results_df[results_df['task'] == task]
            
            # Aggregate by method (mean across datasets)
            method_scores = task_df.groupby('method')[rank_by].mean()
            
            # Rank (1 = best = highest score)
            rankings = method_scores.rank(ascending=False, method='min')
            
            # Create formatted ranking with score
            ranking_data[task] = {
                method: f"{int(rank)} ({score:.4f})"
                for method, (rank, score) in zip(rankings.index, zip(rankings.values, method_scores.values))
            }
        
        ranking_df = pd.DataFrame(ranking_data)
        return ranking_df
    
    def generate_all_summaries(
        self, 
        save_dir: Optional[Path] = None,
        generate_boxplots: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all summary tables and visualizations.
        
        Args:
            save_dir: Directory to save summaries (default: output_base_dir/summaries)
            generate_boxplots: If True, also generate boxplot visualizations
        
        Returns:
            Dictionary of summary name -> DataFrame
        """
        if save_dir is None:
            save_dir = self.output_base_dir / 'summaries'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Scanning all experiment results...")
        results_df = self.scan_all_experiments()
        
        if results_df.empty:
            logger.warning("No results found. Cannot generate summaries.")
            return {}
        
        logger.info(f"Found results for {len(results_df)} experiment configurations")
        logger.info(f"Datasets: {list(results_df['dataset'].unique())}")
        logger.info(f"Tasks: {list(results_df['task'].unique())}")
        logger.info(f"Methods: {list(results_df['method'].unique())}")
        
        summaries: Dict[str, pd.DataFrame] = {}
        
        # Get metric columns (exclude _std and cv_label_splits - for boxplot error bars / header only)
        exclude_cols = {'dataset', 'task', 'method', 'dataset_subgroup', 'global_score', 'global_score_std', 'n_metrics'}
        metric_columns = [
            col for col in results_df.columns
            if col not in exclude_cols and not col.endswith('_std') and col != 'cv_label_splits'
        ]
        # 1. Raw results
        logger.info("Saving raw results...")
        raw_path = save_dir / 'all_results_raw.csv'
        results_df.to_csv(raw_path, index=False)
        summaries['raw'] = results_df
        logger.info(f"Saved raw results to {raw_path}")
        
        # 2. Comprehensive comparison table
        logger.info("Creating comprehensive comparison table...")
        comprehensive_table = self.create_comprehensive_comparison_table(results_df, metric_columns)
        if not comprehensive_table.empty:
            comprehensive_path = save_dir / 'comprehensive_comparison.csv'
            comprehensive_table.to_csv(comprehensive_path, index=False)
            summaries['comprehensive'] = comprehensive_table
            logger.info(f"Saved comprehensive comparison to {comprehensive_path}")
        
        # 3. Distribution summary
        logger.info("Creating distribution summary...")
        distribution_summary = self.create_distribution_summary(results_df, metric_columns)
        if not distribution_summary.empty:
            distribution_path = save_dir / 'distribution_summary.csv'
            distribution_summary.to_csv(distribution_path, index=False)
            summaries['distribution'] = distribution_summary
            logger.info(f"Saved distribution summary to {distribution_path}")
        
        # 4. Task-aggregated tables
        for agg_func in ['mean', 'median', 'std']:
            logger.info(f"Creating task-aggregated table ({agg_func})...")
            aggregated_table = self.create_task_aggregated_table(results_df, metric_columns, aggregation=agg_func)
            if not aggregated_table.empty:
                agg_path = save_dir / f'task_aggregated_{agg_func}.csv'
                aggregated_table.to_csv(agg_path, index=False)
                summaries[f'task_aggregated_{agg_func}'] = aggregated_table
                logger.info(f"Saved {agg_func} aggregated table to {agg_path}")
        
        # 5. Method summaries (one per method)
        logger.info("Creating method summaries...")
        for method in results_df['method'].unique():
            method_summary = self.create_method_summary(results_df, method)
            if not method_summary.empty:
                method_path = save_dir / f'method_summary_{method}.csv'
                method_summary.to_csv(method_path)
                summaries[f'method_{method}'] = method_summary
                logger.info(f"Saved method summary for {method} to {method_path}")
        
        # 6. Task summaries (one per task)
        logger.info("Creating task summaries...")
        for task in results_df['task'].unique():
            task_summary = self.create_task_summary(results_df, task)
            if not task_summary.empty:
                task_path = save_dir / f'task_summary_{task}.csv'
                task_summary.to_csv(task_path, index=False)
                summaries[f'task_{task}'] = task_summary
                logger.info(f"Saved task summary for {task} to {task_path}")
        
        # 7. Subgroup summaries (if dataset_subgroup exists)
        has_subgroups = 'dataset_subgroup' in results_df.columns and results_df['dataset_subgroup'].notna().any()
        if has_subgroups:
            subgroups = sorted(results_df['dataset_subgroup'].dropna().unique())
            logger.info(f"Dataset subgroups found: {subgroups}")
            
            # 7a. Subgroup comparison table (methods × subgroups)
            logger.info("Creating subgroup comparison table...")
            subgroup_comparison = self.create_subgroup_comparison_table(results_df)
            if not subgroup_comparison.empty:
                subgroup_comp_path = save_dir / 'subgroup_comparison.csv'
                subgroup_comparison.to_csv(subgroup_comp_path, index=False)
                summaries['subgroup_comparison'] = subgroup_comparison
                logger.info(f"Saved subgroup comparison to {subgroup_comp_path}")
            
            # 7b. Per-subgroup summaries (method comparison within each subgroup)
            subgroup_summaries_dir = save_dir / 'by_subgroup'
            subgroup_summaries_dir.mkdir(parents=True, exist_ok=True)
            
            for subgroup in subgroups:
                logger.info(f"Creating summary for subgroup: {subgroup}")
                subgroup_summary = self.create_per_subgroup_summary(results_df, subgroup)
                if not subgroup_summary.empty:
                    subgroup_path = subgroup_summaries_dir / f'subgroup_{subgroup}_summary.csv'
                    subgroup_summary.to_csv(subgroup_path, index=False)
                    summaries[f'subgroup_{subgroup}'] = subgroup_summary
                    logger.info(f"Saved subgroup summary for {subgroup} to {subgroup_path}")
        
        # 8. Ranking table
        logger.info("Creating ranking table...")
        ranking_table = self.create_ranking_table(results_df)
        if not ranking_table.empty:
            ranking_path = save_dir / 'method_rankings.csv'
            ranking_table.to_csv(ranking_path)
            summaries['rankings'] = ranking_table
            logger.info(f"Saved ranking table to {ranking_path}")
        
        # 9. Metric directionality reference
        logger.info("Saving metric reference...")
        metric_ref = []
        for metric in metric_columns:
            metric_ref.append({
                'metric': metric,
                'direction': 'lower_is_better' if is_lower_better(metric) else 'higher_is_better',
                'known_range': str(get_metric_range(metric))
            })
        metric_ref_df = pd.DataFrame(metric_ref)
        metric_ref_path = save_dir / 'metric_reference.csv'
        metric_ref_df.to_csv(metric_ref_path, index=False)
        summaries['metric_reference'] = metric_ref_df
        logger.info(f"Saved metric reference to {metric_ref_path}")
        
        # 10. Generate boxplot visualizations
        if generate_boxplots:
            logger.info("=" * 60)
            logger.info("Generating boxplot visualizations...")
            plots_dir = save_dir / 'plots'
            boxplot_gen = BoxplotGenerator(save_dir=plots_dir)
            boxplot_paths = boxplot_gen.generate_all_boxplots(results_df)
            # Per-subgroup: full set of boxplots per subgroup (each in by_subgroup/{subgroup}/plots/)
            if has_subgroups:
                subgroup_base = save_dir / 'by_subgroup'
                per_subgroup_paths = boxplot_gen.generate_per_subgroup_boxplots(
                    results_df,
                    save_dir_by_subgroup=subgroup_base,
                )
                if per_subgroup_paths:
                    logger.info(f"Per-subgroup boxplot sets saved under {subgroup_base} ({len(per_subgroup_paths)} plots total)")
            summaries['_boxplot_paths'] = boxplot_paths
            logger.info(f"All boxplots saved to {plots_dir}")
        
        # 11. Generate confusion matrix heatmaps for label_transfer task
        # Check if we're in a label_transfer context
        # Note: Plots are saved to dataset-specific directories: task/method/dataset/plots/evaluations/
        is_label_transfer = (
            'label_transfer' in results_df['task'].unique() if not results_df.empty else False
        ) or 'label_transfer' in str(self.output_base_dir)
        
        if is_label_transfer:
            logger.info("=" * 60)
            logger.info("Generating confusion matrix heatmaps for label transfer...")
            logger.info("Plots will be saved to dataset-specific directories: task/method/dataset/plots/evaluations/")
            cm_generator = ConfusionMatrixGenerator(output_base_dir=self.output_base_dir)
            confusion_heatmap_paths = cm_generator.generate_heatmaps()
            if confusion_heatmap_paths:
                summaries['_confusion_heatmap_paths'] = confusion_heatmap_paths
                logger.info(f"Generated {len(confusion_heatmap_paths)} confusion matrix heatmaps")
        
        logger.info("=" * 60)
        logger.info(f"All summaries saved to {save_dir}")
        return summaries
    
    
    def generate_boxplots_only(
        self,
        save_dir: Optional[Path] = None,
        results_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[Path]]:
        """
        Generate only boxplot visualizations (no numeric summaries).
        
        Useful for regenerating plots with different settings or from cached results.
        
        Args:
            save_dir: Directory to save plots (default: output_base_dir/summaries/plots)
            results_df: Pre-computed results DataFrame (if None, scans experiments)
        
        Returns:
            Dictionary mapping plot type to list of saved paths
        """
        if save_dir is None:
            save_dir = self.output_base_dir / 'summaries' / 'plots'
        save_dir = Path(save_dir)
        
        if results_df is None:
            logger.info("Scanning all experiment results...")
            results_df = self.scan_all_experiments()
        
        if results_df.empty:
            logger.warning("No results found. Cannot generate boxplots.")
            return {}
        
        logger.info(f"Generating boxplots for {len(results_df)} experiments...")
        boxplot_gen = BoxplotGenerator(save_dir=save_dir)
        return boxplot_gen.generate_all_boxplots(results_df)


def main():
    """CLI entry point for generating summaries."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate result summaries and visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summaries with boxplots (default)
  python -m utils.results_summarizer /path/to/eval_results
  
  # Generate only numeric summaries (no plots)
  python -m utils.results_summarizer /path/to/eval_results --no-plots
  
  # Specify scope explicitly
  python -m utils.results_summarizer /path/to/task_dir --scope task
  
  # Custom save directory
  python -m utils.results_summarizer /path/to/eval_results --save-dir /custom/output
        """
    )
    parser.add_argument('output_dir', type=str, help='Base output directory to scan')
    parser.add_argument('--scope', type=str, default='auto', 
                       choices=['auto', 'root', 'task', 'method'],
                       help='Scanning scope (default: auto-detect)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save summaries (default: output_dir/summaries)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip boxplot generation (faster, numeric summaries only)')
    
    args = parser.parse_args()
    
    summarizer = ResultsSummarizer(args.output_dir, scope=args.scope)
    save_dir = Path(args.save_dir) if args.save_dir else None
    summarizer.generate_all_summaries(save_dir, generate_boxplots=not args.no_plots)


if __name__ == '__main__':
    main()
