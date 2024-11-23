import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .data import DUMP_DATA_FILES
from .out import OUT_DIR
from . import params
from .utils.data_utils import LABEL_DICT


def load_csv(filepath):
    data = pd.read_csv(filepath)
    return data


def visualize_fit_process(model_type):
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
    ax[1].set_title(f"Validation Criteria ({params.CRITERIA.upper()})")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].set_xticks(np.arange(0, max(df['epoch'].unique()) + 1))

    fig.savefig(str(Path(OUT_DIR, 'figures', f'{model_type}_fit_process.jpg')))


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


def visualize_test_metrics(model_type):
    df = load_csv(str(Path(OUT_DIR, 'logs', 'test_logs.csv')))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(ax=ax, data=df, legend=True)
    ax.set_title('Test Metrics')
    ax.set_ylabel('Value')
    ax.set_xticks([])
    for v in ax.containers:
        ax.bar_label(v)

    fig.savefig(str(Path(OUT_DIR, 'figures', f'{model_type}_test_metrics.jpg')))


def visualize_results(model_type):
    visualize_fit_process(model_type)
    visualize_test_metrics(model_type)


def visualize_train_signal(window_size=3):
    fig, axes = plt.subplots(len(DUMP_DATA_FILES['train']), 3, figsize=(10 * 3 * window_size, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, data_file in enumerate(DUMP_DATA_FILES['train']):
        print(f'Load data in {data_file}')
        start_datetime, eeg, emg, mot, lbs, mxs = joblib.load(data_file)
        s_idx = 0
        while True:
            random.seed(time.time())
            s_idx = random.randint(0, len(start_datetime) - window_size - 1)
            if lbs[s_idx] // 2 != lbs[s_idx + 1] // 2 or lbs[s_idx + 1] // 2 != lbs[s_idx + 2] // 2:
                break

        # v_dt = np.array(start_datetime[s_idx: s_idx + window_size])
        v_eeg = np.array(eeg[s_idx:s_idx + window_size])
        v_emg = np.array(emg[s_idx:s_idx + window_size])
        v_mot = np.array(mot[s_idx:s_idx + window_size])
        tmp_lbs = np.array(lbs[s_idx:s_idx + window_size])
        v_lbs = []
        for lb in tmp_lbs:
            lbs = np.array([LABEL_DICT[lb]])
            lbs = np.tile(lbs, 1024)
            v_lbs.append(lbs)
        v_lbs = np.concatenate(v_lbs)

        v_data = {
            "lbs": v_lbs,
            "eeg": np.reshape(v_eeg, (-1)),
            "emg": np.reshape(v_emg, (-1)),
            "mot": np.reshape(v_mot, (-1)),
        }
        df = pd.DataFrame(data=v_data)

        # EEG
        sns.lineplot(ax=axes[i, 0], data=df['eeg'])
        axes[i, 0].set_title('EEG')
        axes[i, 0].set_ylabel('microVolt')
        axes[i, 0].set_xticks(np.arange(1024 * window_size)[::1024])
        axes[i, 0].set_xticklabels(v_lbs[::1024])
        axes[i, 0].xaxis.grid(True)
        # axes[i, 0].set_ylim(mxs[0])
        # axes[i, 0].set_xlabel('Datetime')

        # EMG
        sns.lineplot(ax=axes[i, 1], data=df['emg'])
        axes[i, 1].set_title('EMG')
        axes[i, 1].set_ylabel('microVolt')
        axes[i, 1].set_xticks(np.arange(1024 * window_size)[::1024])
        axes[i, 1].set_xticklabels(v_lbs[::1024])
        axes[i, 1].xaxis.grid(True)
        # axes[i, 1].set_ylim(mxs[1])
        # axes[i, 1].set_xlabel('Datetime')

        # MOT
        sns.lineplot(ax=axes[i, 2], data=df['mot'])
        axes[i, 2].set_title('MOT')
        axes[i, 2].set_ylabel('microVolt')
        axes[i, 2].set_xticks(np.arange(1024 * window_size)[::1024])
        axes[i, 2].set_xticklabels(v_lbs[::1024])
        axes[i, 2].xaxis.grid(True)
        # axes[i, 2].set_ylim(mxs[2])
        # axes[i, 2].set_xlabel('Datetime')

    fig.savefig(str(Path(OUT_DIR, 'figures', 'train_signal.jpg')))


def visualize_labels():
    fig, axes = plt.subplots(1, len(DUMP_DATA_FILES['train']) + 1, figsize=(10 * 4, 10))
    plt.subplots_adjust(wspace=0.3)

    all_num_samples = np.zeros(len(LABEL_DICT) - 1)
    all_total = 0
    labels = list(LABEL_DICT.values())[:-1]
    for i, data_file in enumerate(DUMP_DATA_FILES['train']):
        print(f'Load data in {data_file}')
        _, _, _, _, lbs, _ = joblib.load(data_file)
        total = 0
        num_samples = np.zeros(len(LABEL_DICT) - 1)
        other = 0
        for lb in lbs:
            if lb == -1:
                other += 1
                continue
            num_samples[lb] += 1
            all_num_samples[lb] += 1
            total += 1
            all_total += 1
        color_palette = sns.color_palette("colorblind")
        axes[i].pie([f"{(num_samples[i] * 100 / total)}" for i in range(len(labels))], labels=labels,
                    autopct='%.2f%%', colors=color_palette)
        axes[i].set_title(f'Label distribution in file {i + 1}')
        print(f"Other labels {other} sample(s)")

    color_palette = sns.color_palette("colorblind")
    axes[-1].pie([f"{(all_num_samples[i] * 100 / all_total)}" for i in range(len(labels))], labels=labels,
                 autopct='%.2f%%', colors=color_palette)
    axes[-1].set_title(f'Label distribution in all files')

    fig.savefig(str(Path(OUT_DIR, 'figures', 'labels_analysis.jpg')))


if __name__ == '__main__':
    pass
    # visualize_results()
    # visualize_train_signal()
    # visualize_labels()
