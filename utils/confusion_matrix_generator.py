"""
Confusion matrix heatmap generator for label transfer (annotation) evaluations.

Scans experiment directories and creates confusion matrix heatmaps for each
transfer strategy (even, random, difficult).
"""
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logs_ import get_logger

logger = get_logger()


class ConfusionMatrixGenerator:
    """
    Generate confusion matrix heatmaps for label transfer evaluations.
    
    Scans all method/dataset combinations in the output directory and creates
    confusion matrix heatmaps for each transfer strategy (even, random, difficult).
    
    Attributes:
        output_base_dir: Base directory to scan for confusion matrix files
        dpi: Resolution for saved figures
    """
    
    def __init__(
        self,
        output_base_dir: Path,
        dpi: int = 150
    ):
        """
        Initialize confusion matrix generator.
        
        Args:
            output_base_dir: Base directory to scan for confusion matrix files
            dpi: Resolution for saved figures
        """
        self.output_base_dir = Path(output_base_dir)
        self.dpi = dpi
    
    def _extract_strategy_from_filename(self, filename: str) -> str:
        """
        Extract strategy name from confusion matrix filename.
        
        Args:
            filename: Filename stem (without extension)
        
        Returns:
            Strategy name: 'even', 'random', 'random_mean', 'random_foldN', 'difficult', or 'unknown'
        """
        if 'random' in filename:
            if 'mean' in filename:
                return 'random_mean'
            else:
                # Extract fold number
                fold_match = [s for s in filename.split('_') if s.startswith('fold')]
                if fold_match:
                    return f"random_{fold_match[0]}"
                else:
                    return 'random'
        elif 'even' in filename:
            return 'even'
        elif 'difficult' in filename:
            return 'difficult'
        else:
            return 'unknown'
    
    def _find_label_transfer_directories(self) -> List[Path]:
        """
        Find all label_transfer task directories to scan.
        
        Returns:
            List of paths to label_transfer directories
        """
        # Try task/method/dataset structure first
        task_dirs = list(self.output_base_dir.glob('label_transfer'))
        if not task_dirs:
            # Check if we're already in a task directory
            if 'label_transfer' in str(self.output_base_dir):
                task_dirs = [self.output_base_dir]
            else:
                # Try to find any directory that might contain label_transfer results
                task_dirs = [d for d in self.output_base_dir.iterdir() 
                           if d.is_dir() and 'label_transfer' in str(d)]
        
        return task_dirs
    
    def _select_best_confusion_matrix_file(
        self,
        strategy: str,
        files: List[Path]
    ) -> Tuple[Path, str]:
        """
        Select the best confusion matrix file for a strategy.
        
        For random strategy, prefers mean file if available.
        
        Args:
            strategy: Strategy name
            files: List of candidate files
        
        Returns:
            Tuple of (selected_file, strategy_label)
        """
        # For random strategy, prefer mean if available
        if strategy == 'random_mean' or (strategy == 'random' and len(files) > 1):
            # Use mean file if available
            mean_file = [f for f in files if 'mean' in f.name]
            if mean_file:
                return mean_file[0], 'random (mean across folds)'
            else:
                # Use first file
                return files[0], strategy
        else:
            return files[0], strategy
    
    def _create_heatmap(
        self,
        cm_df: pd.DataFrame,
        method_name: str,
        dataset_name: str,
        strategy_label: str,
        save_path: Path
    ) -> None:
        """
        Create and save a confusion matrix heatmap.
        
        Args:
            cm_df: Confusion matrix DataFrame
            method_name: Name of the method
            dataset_name: Name of the dataset
            strategy_label: Label for the transfer strategy
            save_path: Path to save the plot
        """
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(cm_df) * 0.8), max(8, len(cm_df) * 0.7)))
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
        
        # Create heatmap with both raw counts and normalized percentages
        sns.heatmap(
            cm_normalized,
            annot=cm_df,  # Show raw counts as annotations
            fmt='d',  # Integer format for annotations
            cmap='Blues',
            cbar_kws={'label': 'Normalized (row %)'},
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(
            f'Confusion Matrix: {method_name} - {dataset_name}\n'
            f'Strategy: {strategy_label}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Predicted Cell Type', fontsize=12)
        ax.set_ylabel('True Cell Type', fontsize=12)
        
        # Rotate labels if there are many classes
        if len(cm_df) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def generate_heatmaps(self, plots_dir: Optional[Path] = None) -> List[Path]:
        """
        Generate confusion matrix heatmaps for all found experiments.
        
        Scans all method/dataset combinations in the output directory and creates
        confusion matrix heatmaps for each transfer strategy (even, random, difficult).
        
        Saves plots to dataset-specific directories: task/method/dataset/plots/evaluations/
        
        Args:
            plots_dir: Deprecated - plots are now saved to dataset-specific directories.
                      Kept for backward compatibility but ignored.
            
        Returns:
            List of paths to saved heatmap files
        """
        saved_paths = []
        
        # Find label_transfer directories
        task_dirs = self._find_label_transfer_directories()
        
        if not task_dirs:
            logger.info("No label_transfer task directory found. Skipping confusion matrix heatmaps.")
            return saved_paths
        
        for task_dir in task_dirs:
            logger.info(f"Scanning {task_dir} for confusion matrices...")
            
            # Find all method directories
            for method_dir in task_dir.iterdir():
                if not method_dir.is_dir() or method_dir.name.startswith('.'):
                    continue
                
                method_name = method_dir.name
                
                # Find all dataset directories
                for dataset_dir in method_dir.iterdir():
                    if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                        continue
                    
                    dataset_name = dataset_dir.name
                    metrics_dir = dataset_dir / 'metrics'
                    
                    if not metrics_dir.exists():
                        continue
                    
                    # Find confusion matrix files
                    cm_files = list(metrics_dir.glob('annotation_*_confusion_matrix*.csv'))
                    
                    if not cm_files:
                        continue
                    
                    logger.info(f"Found {len(cm_files)} confusion matrix files for {method_name}/{dataset_name}")
                    
                    # Create dataset-specific plots directory: task/method/dataset/plots/evaluations/
                    dataset_plots_dir = dataset_dir / 'plots' / 'evaluations'
                    dataset_plots_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Group by strategy (even, random, difficult)
                    strategies = {}
                    for cm_file in cm_files:
                        strategy = self._extract_strategy_from_filename(cm_file.stem)
                        
                        if strategy not in strategies:
                            strategies[strategy] = []
                        strategies[strategy].append(cm_file)
                    
                    # Generate heatmap for each strategy
                    for strategy, files in strategies.items():
                        cm_file, strategy_label = self._select_best_confusion_matrix_file(strategy, files)
                        
                        try:
                            # Load confusion matrix
                            cm_df = pd.read_csv(cm_file, index_col=0)
                            
                            # Create filename for saved plot
                            # Use strategy name directly (e.g., 'even', 'difficult', 'random_mean', 'random_fold0')
                            plot_filename = f'confusion_matrix_{strategy}.png'
                            plot_path = dataset_plots_dir / plot_filename
                            
                            # Create and save heatmap
                            self._create_heatmap(
                                cm_df, method_name, dataset_name, strategy_label, plot_path
                            )
                            
                            saved_paths.append(plot_path)
                            logger.info(f"Saved confusion matrix heatmap to {plot_path}")
                            
                        except Exception as e:
                            logger.warning(f"Error generating confusion matrix heatmap for {cm_file}: {e}")
                            continue
        
        return saved_paths
