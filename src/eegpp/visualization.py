from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from out import OUT_DIR
from src.eegpp import params


def load_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def visualize_fit_process():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'fit_logs.csv')))
    fig, ax = plt.subplots(1, 2, figsize=(10 * 2, 5))
    plt.subplots_adjust(wspace=0.3)

    # TRAINING LOSS
    sns.lineplot(ax=ax[0], data=df, x='epoch', y='train/epoch_loss')
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

    # VALIDATION CRITERIA
    sns.lineplot(ax=ax[1], data=df, x='epoch', y=f'val/epoch_{params.CRITERIA}', color='orange')
    ax[1].set_title("Validation Criteria")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

    fig.savefig(str(Path(OUT_DIR, 'figures', 'fit_process.jpg')))


def visualize_fit_process_backup():
    """
        df = load_csv(str(Path(OUT_DIR, 'logs', 'fit_logs.csv')))
        fig, ax = plt.subplots(1, 3, figsize=(10 * 3, 5))
        plt.subplots_adjust(wspace=0.3)

        # TRAIN LOSS
        sns.lineplot(ax=ax[0], data=df, x='epoch', y='train/epoch_loss')
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

        # VALIDATION LOSS
        df_melted = df.melt(id_vars=['epoch', 'fold'], value_vars=['val/epoch_loss', 'val/epoch_loss_binary'],
                            var_name='val_loss_type', value_name='loss')
        sns.lineplot(ax=ax[1], data=df_melted, x='epoch', y='loss', hue='val_loss_type', markers=True)
        ax[1].set_title("Validation Loss")
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

        # VALIDATION METRICS
        df_melted = df.melt(
            id_vars=['epoch', 'fold'],
            value_vars=['val/epoch_auroc', 'val/epoch_auprc', 'val/epoch_auroc_binary', 'val/epoch_auprc_binary',
                        'val/epoch_f1x', 'val/epoch_f1x_binary'],
            var_name='metric_type', value_name='metrics'
        )
        sns.barplot(ax=ax[2], data=df_melted, x='fold', y='metrics', hue='metric_type')
        ax[2].set_title("Validation Metrics")
        ax[2].set_xlabel('Fold')
        ax[2].set_ylabel('Value')
        ax[2].set_xticks(np.arange(0, max(df['fold'].unique()) + 1))
        # for v in ax[2].containers:
        #     ax[2].bar_label(v)

        fig.savefig(str(Path(OUT_DIR, 'figures', 'fit_process.jpg')))
    """


def visualize_test_metrics():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'test_logs.csv')))
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
    visualize_test_metrics()


if __name__ == '__main__':
    visualize_results()
