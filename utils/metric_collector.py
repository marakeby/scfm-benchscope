"""
Metric collection utilities for gathering metrics from experiment directories.

Handles collection of metrics from different task types:
- Batch effects metrics
- Biological signal metrics
- Annotation (label transfer) metrics
- Classification metrics
- Drug response metrics
"""
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
from utils.logs_ import get_logger
from utils.metric_definitions import METRIC_CATEGORIES

logger = get_logger()


class MetricCollector:
    """
    Collect metrics from experiment directories.
    
    Handles different task types and metric file formats.
    """
    
    def collect_metrics_from_experiment(
        self, 
        experiment_dir: Path,
        dataset_name: str,
        task_name: str,
        method_name: str
    ) -> Dict[str, Union[float, str]]:
        """
        Collect all metrics from a single experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory containing metrics/
            dataset_name: Name of the dataset
            task_name: Name of the task
            method_name: Name of the method
            
        Returns:
            Dictionary of metric_name: value (flattened from all metric files)
        """
        metrics_dict: Dict[str, Union[float, str]] = {}
        metrics_dir = experiment_dir / 'metrics'
        
        if not metrics_dir.exists():
            logger.debug(f"No metrics directory found at {metrics_dir}")
            return metrics_dict
        
        # Collect metrics based on task type
        if task_name == 'batch_denoising':
            self._collect_batch_effects_metrics(metrics_dir, method_name, metrics_dict)
        
        elif task_name == 'biological_signal_preservation':
            self._collect_biological_signal_metrics(metrics_dir, method_name, metrics_dict)
        
        elif task_name == 'label_transfer':
            self._collect_annotation_metrics(metrics_dir, metrics_dict)
        
        elif task_name == 'batch_bio_integration':
            self._collect_batch_effects_metrics(metrics_dir, method_name, metrics_dict)
            self._collect_biological_signal_metrics(metrics_dir, method_name, metrics_dict)
            self._collect_annotation_metrics(metrics_dir, metrics_dict)
        
        elif any(x in task_name for x in ['classification', 'cell_type', 'cancer_stage', 
                                           'cancer_subtype', 'chemo_sensitivity', 'treatment_response']):
            self._collect_classification_metrics(metrics_dir, metrics_dict)
        
        elif task_name == 'drug_response':
            self._collect_drug_response_metrics(metrics_dir, metrics_dict)
        
        else:
            # Generic collection: try all known file patterns
            self._collect_batch_effects_metrics(metrics_dir, method_name, metrics_dict)
            self._collect_biological_signal_metrics(metrics_dir, method_name, metrics_dict)
            self._collect_annotation_metrics(metrics_dir, metrics_dict)
            self._collect_classification_metrics(metrics_dir, metrics_dict)
        
        return metrics_dict
    
    def _collect_batch_effects_metrics(
        self, 
        metrics_dir: Path, 
        method_name: str, 
        metrics_dict: Dict[str, Union[float, str]]
    ) -> None:
        """Collect batch effects metrics from batch_effects_metrics.csv."""
        batch_effects_file = metrics_dir / 'batch_effects' / 'batch_effects_metrics.csv'
        if not batch_effects_file.exists():
            batch_effects_file = metrics_dir / 'batch_effects_metrics.csv'
        
        if not batch_effects_file.exists():
            return
        
        try:
            df = pd.read_csv(batch_effects_file, index_col=0)
            # Find the main embedding row (not baseline)
            main_rows = [idx for idx in df.index if not str(idx).startswith('X_baseline_')]
            
            if main_rows:
                row_idx = main_rows[0]
            elif df.shape[0] > 0:
                row_idx = df.index[0]
            else:
                return
            
            allowed = set(METRIC_CATEGORIES.get('batch_effects', []))
            for col in df.columns:
                if col not in allowed:
                    continue
                value = df.loc[row_idx, col]
                if pd.notna(value):
                    metrics_dict[f'batch_effects_{col}'] = float(value)
                    
        except Exception as e:
            logger.warning(f"Error reading batch_effects metrics from {batch_effects_file}: {e}")
    
    def _collect_biological_signal_metrics(
        self, 
        metrics_dir: Path, 
        method_name: str, 
        metrics_dict: Dict[str, Union[float, str]]
    ) -> None:
        """Collect biological signal metrics from biological_signal_metrics.csv."""
        bio_signal_file = metrics_dir / 'biological_signal' / 'biological_signal_metrics.csv'
        if not bio_signal_file.exists():
            bio_signal_file = metrics_dir / 'biological_signal_metrics.csv'
        
        if not bio_signal_file.exists():
            return
        
        try:
            df = pd.read_csv(bio_signal_file, index_col=0)
            # Find the main embedding row (not baseline)
            main_rows = [idx for idx in df.index if not str(idx).startswith('X_baseline_')]
            
            if main_rows:
                row_idx = main_rows[0]
            elif df.shape[0] > 0:
                row_idx = df.index[0]
            else:
                return
            
            for col in df.columns:
                if col == 'avg_bio':
                    continue  # Exclude composite avg_bio from summaries/plots
                value = df.loc[row_idx, col]
                if pd.notna(value):
                    metrics_dict[f'biological_signal_{col}'] = float(value)
                    
        except Exception as e:
            logger.warning(f"Error reading biological_signal metrics from {bio_signal_file}: {e}")
    
    def _collect_annotation_metrics(
        self, 
        metrics_dir: Path, 
        metrics_dict: Dict[str, Union[float, str]]
    ) -> None:
        """Collect annotation (label transfer) metrics."""
        # Priority 1: Read summary file
        summary_file = metrics_dir / 'annotation_summary.csv'
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                for _, row in df.iterrows():
                    strategy = row.get('strategy', 'unknown')
                    for col in df.columns:
                        if col != 'strategy':
                            try:
                                value = float(row[col])
                                if pd.notna(value):
                                    metrics_dict[f'annotation_{strategy}_{col}'] = value
                            except (ValueError, TypeError):
                                pass
                return
            except Exception as e:
                logger.warning(f"Error reading annotation summary from {summary_file}: {e}")
        
        # Priority 2: Read individual strategy files
        for csv_file in metrics_dir.glob('annotation_*_metrics.csv'):
            if 'fold' not in csv_file.name and 'per_class' not in csv_file.name:
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                    strategy = csv_file.stem.replace('annotation_', '').replace('_metrics', '')
                    
                    for metric_name in df.index:
                        if len(df.columns) > 0:
                            value = df.iloc[df.index.get_loc(metric_name), 0]
                            # Handle formatted strings like "0.95 ± 0.02"
                            if isinstance(value, str) and '±' in value:
                                try:
                                    value = float(value.split('±')[0].strip())
                                except ValueError:
                                    continue
                            if pd.notna(value):
                                metrics_dict[f'annotation_{strategy}_{metric_name}'] = float(value)
                                
                except Exception as e:
                    logger.warning(f"Error reading annotation metrics from {csv_file}: {e}")
    
    def _collect_drug_response_metrics(
        self,
        metrics_dir: Path,
        metrics_dict: Dict[str, Union[float, str]],
    ) -> None:
        """
        Collect drug response metrics from drug_response_summary.csv.

        Summary CSV has a 3-row header (metric, model, split) then data rows
        with split name and values for (pearson_r, r2, rmse) x (mlp, rf).
        Flattens to keys: drug_response_{metric}_{model}_{split}.
        """
        summary_file = metrics_dir / 'drug_response' / 'drug_response_summary.csv'
        if not summary_file.exists():
            return
        try:
            # Header is first 3 rows; data starts row 3 (0-indexed)
            df = pd.read_csv(summary_file, header=None)
            if df.shape[0] < 4 or df.shape[1] < 7:
                return
            row0 = df.iloc[0]  # metric names: '', pearson_r, pearson_r, r2, r2, rmse, rmse
            row1 = df.iloc[1]  # model names: model, mlp, rf, mlp, rf, mlp, rf
            for data_idx in range(3, len(df)):
                row = df.iloc[data_idx]
                split_name = str(row.iloc[0]).strip()
                if not split_name or split_name == 'nan':
                    continue
                for col_idx in range(1, 7):
                    metric = str(row0.iloc[col_idx]).strip() if pd.notna(row0.iloc[col_idx]) else ''
                    model = str(row1.iloc[col_idx]).strip() if pd.notna(row1.iloc[col_idx]) else ''
                    if not metric or not model:
                        continue
                    try:
                        val = row.iloc[col_idx]
                        if pd.notna(val):
                            metrics_dict[f'drug_response_{metric}_{model}_{split_name}'] = float(val)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"Error reading drug_response metrics from {summary_file}: {e}")
    
    def _format_cv_label_splits_from_per_class(self, df: pd.DataFrame) -> str:
        """
        Build 'n=total (class1=s1, class2=s2)' from per-class aggregated CSV.
        Expects index rows like 'classname_support' with 'mean' column.
        """
        support_parts: List[str] = []
        total = 0.0
        for metric_name in df.index:
            if not str(metric_name).endswith('_support') or 'mean' not in df.columns:
                continue
            try:
                val = df.loc[metric_name, 'mean']
                if pd.isna(val):
                    continue
                total += float(val)
                class_name = str(metric_name)[:-len('_support')]
                support_parts.append(f"{class_name}={int(round(float(val)))}")
            except (ValueError, TypeError):
                continue
        if not support_parts:
            return ""
        total_int = int(round(total))
        return f"n={total_int} ({', '.join(support_parts)})"

    def _collect_classification_metrics(
        self, 
        metrics_dir: Path, 
        metrics_dict: Dict[str, Union[float, str]]
    ) -> None:
        """Collect classification metrics.

        Prefers CV aggregated metrics (mean ± std across folds) from metrics/cv/ when present,
        so summaries and boxplots can show cross-validated performance (important for tasks
        with a single dataset per method, e.g. cancer tasks). Also collects CV label/support
        from per-class aggregated CSVs for display on boxplots (e.g. n=10 (X=7, Y=3)).
        """
        # CV label/support from per-class aggregated (for boxplot header)
        per_class_candidates: List[Path] = []
        per_class_candidates.extend(metrics_dir.glob('*_cv_per_class_metrics_aggregated.csv'))
        cv_dir = metrics_dir / 'cv'
        if cv_dir.exists():
            per_class_candidates.extend(cv_dir.glob('*_cv_per_class_metrics_aggregated.csv'))
        for csv_file in per_class_candidates:
            if 'train' not in csv_file.name:
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                    label_splits = self._format_cv_label_splits_from_per_class(df)
                    if label_splits:
                        metrics_dict['cv_label_splits'] = label_splits
                        break
                except Exception as e:
                    logger.debug(f"Could not read CV per-class for label splits from {csv_file}: {e}")
        # Priority 1: CV aggregated metrics (mean ± std across folds) - patient-level
        # Look in metrics_dir and metrics_dir/cv/ (cancer tasks write CV results under cv/)
        cv_candidates: List[Path] = []
        cv_candidates.extend(metrics_dir.glob('*_cv_metrics_aggregated.csv'))
        if cv_dir.exists():
            cv_candidates.extend(cv_dir.glob('*_cv_metrics_aggregated.csv'))
        for csv_file in cv_candidates:
            if 'train' not in csv_file.name and 'per_class' not in csv_file.name:
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                    train_func = csv_file.stem.replace('_cv_metrics_aggregated', '')
                    has_std = 'std' in df.columns
                    for metric_name in df.index:
                        if 'mean' in df.columns:
                            value = df.loc[metric_name, 'mean']
                            if pd.notna(value):
                                key = f'classification_{train_func}_{metric_name}'
                                metrics_dict[key] = float(value)
                                if has_std:
                                    std_val = df.loc[metric_name, 'std']
                                    if pd.notna(std_val):
                                        metrics_dict[f'{key}_std'] = float(std_val)
                except Exception as e:
                    logger.warning(f"Error reading CV aggregated metrics from {csv_file}: {e}")
        
        # Priority 1b: Cell-level CV aggregated metrics (mean across folds)
        cell_cv_dir = metrics_dir / 'cell_level_pred' / 'cv'
        if cell_cv_dir.exists():
            for csv_file in cell_cv_dir.glob('cell_cv_metrics_aggregated.csv'):
                if 'train' not in csv_file.name and 'per_class' not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        
                        for metric_name in df.index:
                            if 'mean' in df.columns:
                                value = df.loc[metric_name, 'mean']
                                if pd.notna(value):
                                    # Use 'cell' as the train_func identifier for cell-level predictions
                                    metrics_dict[f'classification_cell_{metric_name}'] = float(value)
                                    
                    except Exception as e:
                        logger.warning(f"Error reading cell-level CV aggregated metrics from {csv_file}: {e}")
        
        # Priority 2: Non-CV metrics (cls_metrics_*.csv)
        for csv_file in metrics_dir.glob('cls_metrics_*.csv'):
            if 'train' not in csv_file.name:
                try:
                    df = pd.read_csv(csv_file, index_col=0)
                    postfix = csv_file.stem.replace('cls_metrics_', '')
                    
                    for metric_name in df.index:
                        if len(df.columns) > 0:
                            value = df.iloc[df.index.get_loc(metric_name), 0]
                            key = f'classification_{postfix}_{metric_name}'
                            if key not in metrics_dict and pd.notna(value):
                                metrics_dict[key] = float(value)
                                
                except Exception as e:
                    logger.warning(f"Error reading cls_metrics from {csv_file}: {e}")
        
        # Priority 2b: Cell-level non-CV metrics
        cell_level_dir = metrics_dir / 'cell_level_pred'
        if cell_level_dir.exists():
            for csv_file in cell_level_dir.glob('*_metrics_cell.csv'):
                if 'train' not in csv_file.name and 'per_class' not in csv_file.name and 'cv' not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        # Extract model name from filename (e.g., random_forest_per_class_metrics_cell.csv -> random_forest)
                        model_name = csv_file.stem.replace('_per_class_metrics_cell', '').replace('_metrics_cell', '')
                        
                        for metric_name in df.index:
                            if len(df.columns) > 0:
                                value = df.iloc[df.index.get_loc(metric_name), 0]
                                key = f'classification_cell_{model_name}_{metric_name}'
                                if key not in metrics_dict and pd.notna(value):
                                    metrics_dict[key] = float(value)
                                    
                    except Exception as e:
                        logger.warning(f"Error reading cell-level metrics from {csv_file}: {e}")
