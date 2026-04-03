

import os
import pandas as pd

import os
from os import listdir
from os.path import isfile, join, dirname, basename
from matplotlib import pyplot as plt
import numpy as np
import math




model_name_map={
'hvg': 'HVG',
'pca': 'PCA',
'scgpt': 'scGPT', 
'scgpt_cancer': 'scGPT [cancer]',
'scvi':'scVI',
'scvi_donor_id':'scVI',
'scfoundation':'scFoundation',
'scimilarity':'SCimilarity',
'cellplm':'CellPLM',
'nicheformer': 'Nicheformer',
'gf-6L-30M-i2048': 'GF-V1',
'gf-6L-30M-i2048_continue': 'GF-V1 [continue]',
'Geneformer-V2-104M_CLcancer': 'GF-V2 [cancer]',
'Geneformer-V2-104M': 'GF-V2',
'Geneformer-V2-104M_continue': 'GF-V2 [continue]',
'Geneformer-V2-316M': 'GF-V2-Deep',
'gf-6L-30M-i2048_finetune': 'GF-V1 [finetune]',
'Geneformer-V2-104M_finetune': 'GF-V2 [finetune]',

}



experiment_name_map=dict(pre_post ='Treatment Naive vs Anti PD1',
                         brca_full_pre_post ='Treatment Naive vs Anti PD1',
                         brca_pre_post ='Treatment Naive vs Anti PD1',
                         chemo= 'Treatment Naive vs Neoadjuvant Chemo',
                         brca_full_chemo= 'Treatment Naive vs Neoadjuvant Chemo',
                         brca_chemo= 'Treatment Naive vs Neoadjuvant Chemo',
                         luad2 = 'Treatment Naive vs TKI treated',
                        # luad22 = 'Treatment Naive vs TKI treated',
                         luad1 = 'Early stage vs Late stage',
                        # luad11 = 'Early stage vs Late stage',
                         outcome ='T-cell exhaustion',
                         brca_full_outcome ='T-cell exhaustion',
                         brca_outcome ='T-cell exhaustion',
                         subtype= 'ER+ vs TNBC',
                         brca_full_subtype= 'ER+ vs TNBC', 
                        brca_subtype= 'ER+ vs TNBC', 
                        brca_cell_type= 'BRCA Cell Type')


def collect_cv_metrics(experiment_dirs):
    """
    Collects metrics from MIL, Vote, and Avg CSVs across experiment directories.

    Args:
        experiment_dirs (list): List of experiment directory names.

    Returns:
        dict: Dictionary with keys 'mil', 'vote', and 'avg' containing respective DataFrames.
    """
    mil_results = []
    vote_results = []
    avg_results = []

    for exp in experiment_dirs:
        mil_path = os.path.join(exp, 'cv', 'mil_cv_metrics.csv')
        vote_path = os.path.join(exp, 'cv', 'vote_cv_metrics.csv')
        vote_path2 = os.path.join(exp, 'cv', 'cls_vote_cv_metrics.csv')
        avg_path = os.path.join(exp, 'cv', 'avg_cv_metrics.csv')
        
        if os.path.exists(vote_path) :
            vote_df = pd.read_csv(vote_path)
            vote_df.columns = ['Metrics', 'model', 'fold']
            vote_df['experiment'] = os.path.basename(exp)
            vote_results.append(vote_df)
        
        if os.path.exists(vote_path2) :
            vote_df = pd.read_csv(vote_path2)
            vote_df.columns = ['Metrics', 'model', 'fold']
            vote_df['experiment'] = os.path.basename(exp)
            vote_results.append(vote_df)

        if os.path.exists(avg_path):
            avg_df = pd.read_csv(avg_path)
            avg_df.columns = ['Metrics', 'model', 'fold']
            avg_df['experiment'] = os.path.basename(exp)
            avg_results.append(avg_df)

        if os.path.exists(mil_path):
            mil_df = pd.read_csv(mil_path)
            mil_df.columns = ['Metrics', 'model', 'fold']
            mil_df['experiment'] = os.path.basename(exp)
            mil_results.append(mil_df)
        else:
            print(f"Missing files in {exp}")

    results = {
        "mil": pd.concat(mil_results, ignore_index=True) if mil_results else pd.DataFrame(),
        "vote": pd.concat(vote_results, ignore_index=True) if vote_results else pd.DataFrame(),
        "avg": pd.concat(avg_results, ignore_index=True) if avg_results else pd.DataFrame(),
    }

    return results

def map_groups(exp):
    exp = exp.lower()
    if 'gf' in exp:
        return 'Geneformer'
    elif 'geneformer' in exp:
        return 'Geneformer'
    elif 'scfoundation' in exp:
        return 'Other'
    elif 'scimilarity' in exp:
        return 'Other'
    elif 'scgpt' in exp:
        return 'scGPT'
    elif 'cellplm' in exp:
        return 'Other'
    
    elif any(x in exp for x in ['hvg', 'pca', 'scvi']):
        return 'Baseline'
    else:
        return 'Other'  # optional fallback


def plot_cv_metrics(mil_df_all, vote_df_all, labels, x_label, metric, show_error_bars=True, plot_type='errorbar'):
    """
    Plots the average value of a specified metric across folds for each experiment,
    with optional error bars or bar plot, sorted within groups and spaced between groups.
    Adds background highlighting for each group.

    Args:
        mil_df_all (pd.DataFrame): DataFrame with MIL metrics and 'experiment' & 'fold' columns.
        vote_df_all (pd.DataFrame): DataFrame with Vote metrics and 'experiment' & 'fold' columns.
        metric (str): The name of the metric to plot (e.g., 'AUC').
        show_error_bars (bool): Whether to show error bars (used with 'errorbar' plot_type).
        plot_type (str): Type of plot, either 'errorbar' or 'bar'.
    """

    # Filter for selected metric
    mil_metric = mil_df_all[mil_df_all['Metrics'] == metric]
    vote_metric = vote_df_all[vote_df_all['Metrics'] == metric]

    # Aggregate stats
    mil_stats = mil_metric.groupby(['experiment', 'group'])['model'].agg(['mean', 'std']).reset_index()
    vote_stats = vote_metric.groupby(['experiment', 'group'])['model'].agg(['mean', 'std']).reset_index()
    # avg_stats = vote_metric.groupby(['experiment', 'group'])['model'].agg(['mean', 'std']).reset_index()
    
    # Sort within group by mean
    mil_stats['sort_key'] = mil_stats.groupby('group')['mean'].transform(lambda x: x.rank(method='first'))
    mil_stats = mil_stats.sort_values(by=['group', 'sort_key'])
    vote_stats = vote_stats.set_index('experiment').loc[mil_stats['experiment'].values].reset_index()
    # avg_stats = avg_stats.set_index('experiment').loc[mil_stats['experiment'].values].reset_index()


    # Create spacing between groups
    unique_groups = mil_stats['group'].unique()
    group_to_exps = {g: mil_stats[mil_stats['group'] == g] for g in unique_groups}

    x_labels, group_labels = [], []
    x_positions = []
    xpos = 0
    spacing = 1.0  # space between groups
    group_bounds = []

    for g in unique_groups:
        exps = group_to_exps[g]
        start_x = xpos
        for _, row in exps.iterrows():
            x_labels.append(row['experiment'])
            group_labels.append(g)
            x_positions.append(xpos)
            xpos += 1
        end_x = xpos - 1
        group_bounds.append((start_x - 0.5, end_x + 0.5))
        xpos += spacing  # add space after group

    x = np.array(x_positions)
    mil_means = mil_stats['mean'].values
    mil_stds = mil_stats['std'].values
    vote_means = vote_stats['mean'].values
    vote_stds = vote_stats['std'].values

    # Plot
    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Add alternating background colors for groups
    for i, (xmin, xmax) in enumerate(group_bounds):
        # ax.axvspan(xmin, xmax, color='lightgray' if i % 2 == 0 else 'whitesmoke', alpha=0.3)

        ax.axvspan(xmin, xmax, color='whitesmoke' )

    offset = 0.2
    width = 0.35

    if plot_type == 'errorbar':
        plt.errorbar(x - offset, mil_means, yerr=mil_stds if show_error_bars else None,
                     fmt='o', label=labels[0], capsize=5 if show_error_bars else 0)
        plt.errorbar(x + offset, vote_means, yerr=vote_stds if show_error_bars else None,
                     fmt='s', label=labels[1], capsize=5 if show_error_bars else 0)
    elif plot_type == 'bar':
        plt.bar(x - width/2, mil_means, yerr=mil_stds if show_error_bars else None,
                width=width, label=labels[0], capsize=5 if show_error_bars else 0)
        plt.bar(x + width/2, vote_means, yerr=vote_stds if show_error_bars else None,
                width=width, label=labels[1], capsize=5 if show_error_bars else 0)

    # Add group names
    for i, g in enumerate(unique_groups):
        idxs = [j for j, grp in enumerate(group_labels) if grp == g]
        if idxs:
            mid = np.mean([x[j] for j in idxs])
            plt.text(mid, 1.1, g, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.xlabel(x_label)
    plt.ylabel(metric)
    # plt.title(f'{metric} Across Experiments ({"Bar" if plot_type=="bar" else "Mean ± Std"})')
    plt.ylim(0, 1.05)
    # plt.legend()
    plt.legend(bbox_to_anchor=(-.1, -.1), loc='lower left', borderaxespad=0.)
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    
    # plt.show()
    return fig

def load_metrics_from_folder(base_path):
    """
    Reads cls_metrics_vote.csv and cls_metrics_mil.csv from each subfolder in the base path.
    Returns a dictionary with classifier names as keys and a tuple of DataFrames (mil_df, vote_df) as values.
    """
    classifier_data = {}
    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            mil_path = os.path.join(subfolder_path, "cls_metrics_mil.csv")
            vote_path = os.path.join(subfolder_path, "cls_metrics_vote.csv")
            avg_path = os.path.join(subfolder_path, "cls_metrics_avg_expr.csv")
            if os.path.exists(mil_path) and os.path.exists(vote_path):
                mil_df = pd.read_csv(mil_path)
                vote_df = pd.read_csv(vote_path)
                avg_df = pd.read_csv(avg_path)
                classifier_data[subfolder] = (mil_df, vote_df, avg_df)
    return classifier_data

def plot_radar_for_classifiers(classifier_data, columns=3):
    """
    Plots radar diagrams for each classifier using their metrics.
    The plots are arranged in a grid with the specified number of columns.
    """
    n_classifiers = len(classifier_data)
    rows = math.ceil(n_classifiers / columns)

    fig, axs = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows), subplot_kw=dict(polar=True))
    axs = axs.flatten()  # Flatten in case it's a 2D array

    for i, (classifier, (mil_df, vote_df, avg_df)) in enumerate(classifier_data.items()):
        ax = axs[i]

        metrics = mil_df['Metrics'].tolist()
        mil_scores = mil_df['randomforest'].tolist()
        vote_scores = vote_df['randomforest'].tolist()
        avg_scores = avg_df['randomforest'].tolist()

        # Check for consistent lengths
        if not (len(metrics) == len(mil_scores) == len(vote_scores)):
            print(f"Skipping {classifier} due to mismatched lengths:")
            print(f"Metrics: {len(metrics)}, MIL: {len(mil_scores)}, Vote: {len(vote_scores)}")
            continue
        noise_std =.005
        mil_scores = [x + np.random.normal(0, noise_std) for x in mil_scores]
        vote_scores = [x + np.random.normal(0, noise_std) for x in vote_scores]
        avg_scores = [x + np.random.normal(0, noise_std) for x in avg_scores]
        
        # Close the radar chart
        metrics.append(metrics[0])
        mil_scores.append(mil_scores[0])
        vote_scores.append(vote_scores[0])
        avg_scores.append(avg_scores[0])
        
        angles = np.linspace(0, 2 * np.pi, len(metrics)).tolist()

        ax.plot(angles, mil_scores, label='MIL', linewidth=2, color='tab:blue')
        # ax.fill(angles, mil_scores, alpha=0.1, color='tab:blue')

        ax.plot(angles, vote_scores, label='Vote', linewidth=2, color='tab:orange')
        # ax.fill(angles, vote_scores, alpha=0.1, color='tab:orange')

        ax.plot(angles, avg_scores, label='Avg.', linewidth=2, color='tab:green')
        # ax.fill(angles, avg_scores, alpha=0.1, color='tab:green')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1])
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        ax.set_title(f'{classifier}', size=14, pad=20)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.legend(['MIL', 'Vote', 'Avg.'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    # plt.show()
    return fig


def get_embedding_metrics(experiment_dirs):

    unsupervised_metrics= []
    for dir_ in experiment_dirs:
        
        # dir_path = join(base_dir, dir_)
        model  = basename(dir_)
        files = [f for f in listdir(dir_) if isfile(join(dir_, f))]



        # unsupervised metrics
        if 'embedding_metrics.csv' in files:
            f = join(dir_, 'embedding_metrics.csv')
            df = pd.read_csv(f, index_col=0)
            df.columns=[model]
            unsupervised_metrics.append(df)
    
    un_df = pd.concat(unsupervised_metrics, axis=1)
    
    
    return un_df

def assign_group(row):
    index_name = row.name.lower()  # ensure case-insensitive matching

    if 'gf' in index_name:
        return 'geneformer'
    elif 'scgpt' in index_name:
        return 'scgpt'
    elif any(x in index_name for x in ['hvg', 'pca', 'scvi']):
        return 'baseline'
    else:
        return 'other'  # optional fallback


def plot_groups(data_df, col = 'ASW_label'):
    assert col in data_df.columns
    assert 'group' in data_df.columns
    fig, ax = plt.subplots()
    labels = []
    xticks=[]
    shift = 0
    gap=1
    bar_width = 0.8
    groups= sorted(data_df.group.unique())
    for g in groups:
        subset = data_df[data_df.group ==g].copy().sort_values(col)
        values = subset[col].values
        x_pos = shift+ np.arange(len(values))
        bar = ax.bar(x_pos, values, width=bar_width, label=g)

        labels.extend(subset.index)
        xticks.extend(x_pos)
        shift = shift+len(values) + gap
        print(shift)

    ax.legend(groups)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1.05)
    
    plt.title(col)
    plt.xticks(rotation=90)
    return fig
