from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from out import OUT_DIR


def load_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def visualize_fit_process():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'my_fit_logs.csv')))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)

    # TRAIN VISUALIZATION
    sns.lineplot(ax=ax[0], data=df, x='epoch', y='train/fold_loss')
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

    # VALIDATION_VISUALIZATION
    df_melted = df.melt(id_vars=['epoch', 'fold'], value_vars=['val/fold_loss', 'val/fold_loss_binary'],
                        var_name='val_loss_type', value_name='loss')
    sns.lineplot(ax=ax[1], data=df_melted, x='epoch', y='loss', hue='val_loss_type', markers=True)
    ax[1].set_title("Validation Loss")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))
    fig.savefig(str(Path(OUT_DIR, 'figures', 'fit_process.jpg')))


def visualize_val_metrics():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'my_val_metrics_logs.csv')))
    fig, ax = plt.subplots(figsize=(10, 10))
    df_melted = df.melt(
        id_vars=['epoch'],
        value_vars=['val/mean_val_loss', 'val/mean_auroc', 'val/mean_auprc', 'val/mean_auroc_binary',
                    'val/mean_auprc_binary', 'val/f1x', 'val/f1x_binary'],
        var_name='metric_type', value_name='metrics'
    )
    sns.lineplot(data=df_melted, x='epoch', y='metrics', hue='metric_type', markers=True)
    ax.set_title('KFold Cross Validation Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))
    fig.savefig(str(Path(OUT_DIR, 'figures', 'val_metrics.jpg')))


def visualize_test_metrics():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'my_test_metrics_logs.csv')))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(ax=ax, data=df, legend=True)
    ax.set_title('Test Metrics')
    ax.set_ylabel('Value')
    ax.set_xticks([])
    for v in ax.containers:
        ax.bar_label(v)

    fig.savefig(str(Path(OUT_DIR, 'figures', 'test_metrics.jpg')))


def visualize_results():
    visualize_fit_process()
    visualize_val_metrics()
    visualize_test_metrics()


if __name__ == '__main__':
    visualize_results()
