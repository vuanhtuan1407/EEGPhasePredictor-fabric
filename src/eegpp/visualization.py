from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from out import OUT_DIR


def load_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def visualize_fit_process():
    fig, ax = plt.subplots()
    df = load_csv(str(Path(OUT_DIR, 'logs', 'my_fit_logs.csv')))
    df_melted = df.melt(id_vars=['epoch', 'fold'], value_vars=['train/fold_loss', 'val/fold_loss'],
                        var_name='loss_type', value_name='loss')
    sns.lineplot(ax=ax, data=df_melted, x='epoch', y='loss', hue='loss_type', markers=True)
    ax.set_title("Training/Validation Loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim(top=2)
    fig.savefig(str(Path(OUT_DIR, 'figures', 'fit_process.jpg')))


def visualize_val_metrics():
    df = load_csv(str(Path(OUT_DIR, 'logs', 'my_val_metrics_logs.csv')))
    fig, ax = plt.subplots(figsize=(10, 10))
    df_melted = df.melt(
        id_vars=['epoch'],
        value_vars=['val/mean_val_loss', 'val/mean_auroc', 'val/mean_auprc', 'val/mean_auroc_binary',
                    'val/mean_auprc_binary', 'val/metric', 'val/metric_binary'],
        var_name='metric_type', value_name='metrics'
    )
    sns.lineplot(data=df_melted, x='epoch', y='metrics', hue='metric_type', markers=True)
    ax.set_title('KFold Cross Validation Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
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
