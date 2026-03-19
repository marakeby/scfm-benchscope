"""
Metric definitions and utilities for results summarization.

Defines metric directionality, ranges, categories, and utility functions
for metric normalization and comparison.
"""
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# METRIC DIRECTIONALITY DEFINITIONS
# =============================================================================

# Metrics where LOWER values indicate BETTER performance
# All other metrics are assumed to be "higher is better"
LOWER_IS_BETTER_METRICS: Set[str] = {
    # Batch effects: BRAS, CiLISI_batch (all higher is better)
    'rmse',  # Drug response: RMSE lower is better
}


# =============================================================================
# METRIC RANGE DEFINITIONS
# =============================================================================

# Define known ranges for metrics for proper normalization
# Format: metric_name -> (min_value, max_value)
# All batch effect metrics are stored in original range and direction.
# iLISI/cLISI: stored as raw median; range is dataset-dependent (use data min/max for plots).
METRIC_RANGES: Dict[str, Tuple[float, float]] = {
    # Batch effects metrics (global_score = mean of these two normalized)
    'bras': (0.0, 1.0),              # BRAS batch removal score (scib_metrics), higher is better
    'CiLISI_batch': (0.0, 1.0),      # Normalized per cell-type iLISI: (raw - 1) / (n_batches - 1), higher = better mixing

    # Biological signal metrics (scib_metrics Leiden only)
    'nmi': (0.0, 1.0),               # NMI with Leiden (scib_metrics), higher is better
    'ari': (-1.0, 1.0),              # ARI with Leiden (scib_metrics), higher is better
    
    # Classification/Annotation metrics
    'AUC': (0.0, 1.0),
    'AUPRC': (0.0, 1.0),
    'F1': (0.0, 1.0),
    'Accuracy': (0.0, 1.0),
    'Precision': (0.0, 1.0),
    'Recall': (0.0, 1.0),

    # Drug response metrics (per split/model)
    'pearson_r': (-1.0, 1.0),
    'r2': (-1.0, 1.0),
    'rmse': (0.0, 1.0),
}


# =============================================================================
# METRIC CATEGORY DEFINITIONS (for organized reporting)
# =============================================================================

METRIC_CATEGORIES: Dict[str, List[str]] = {
    'batch_effects': [
        'bras', 'CiLISI_batch'
    ],
    'biological_signal': [
        'nmi', 'ari',
    ],
    'classification': [
        'AUC', 'AUPRC', 'F1', 'Accuracy', 'Precision', 'Recall'
    ],
    'annotation': [
        'F1', 'Accuracy', 'Precision', 'Recall'
    ],
    'drug_response': [
        'pearson_r', 'r2', 'rmse'
    ],
}


# =============================================================================
# COLOR PALETTES FOR VISUALIZATION
# =============================================================================

# Colorblind-friendly palette for methods
METHOD_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# Paired palette for tasks
TASK_COLORS = [
    '#a6cee3',  # light blue
    '#1f78b4',  # dark blue
    '#b2df8a',  # light green
    '#33a02c',  # dark green
    '#fb9a99',  # light red
    '#e31a1c',  # dark red
    '#fdbf6f',  # light orange
    '#ff7f00',  # dark orange
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_lower_better(metric_name: str) -> bool:
    """
    Check if a metric is "lower is better".
    
    Args:
        metric_name: Full metric name (may include prefixes like 'batch_effects_')
    
    Returns:
        True if lower values are better for this metric
    """
    # Check direct match
    if metric_name in LOWER_IS_BETTER_METRICS:
        return True
    
    # Check if any lower-is-better metric is contained in the name
    for lower_metric in LOWER_IS_BETTER_METRICS:
        if lower_metric in metric_name:
            return True
    
    return False


def get_metric_range(metric_name: str) -> Optional[Tuple[float, float]]:
    """
    Get the known range for a metric.
    
    Args:
        metric_name: Full metric name (may include prefixes)
    
    Returns:
        Tuple of (min, max) if known, None otherwise
    """
    # Check direct match first
    if metric_name in METRIC_RANGES:
        return METRIC_RANGES[metric_name]
    
    # Check if any known range metric is contained in the name
    for range_metric, range_vals in METRIC_RANGES.items():
        if range_metric in metric_name:
            return range_vals
    
    return None
