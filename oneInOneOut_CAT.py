# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Load libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange
import math
import random

import math
import random
# ADD THESE LINES:
from scipy.integrate import trapezoid as trapz
from scipy.spatial.distance import euclidean
try:
    from fastdtw import fastdtw
except ImportError:
    print("Warning: fastdtw not installed. DTW metrics will be skipped.")
    # Dummy function to prevent crash if library is missing
    def fastdtw(x, y, dist): return 0, None

# Config
GPU_ID = 0
DATA_PATH = 'train_data.csv'
OUTPUT_DIR = 'Origin_Exports'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

OBS_WINDOW = 10
BATCH_SIZE = 64
NUM_EPOCHS = 1000
OPTIMIZER = 'AdamW' # 'AdamW' | 'SOAP'
LR = 1e-4
SEED = 24
DROP_RATE = 0.2
NUM_BLOCKS = 3
LATENT_DIM = 128
NUM_HEADS = 4
# Augmentation settings (train set only)
AUGMENT_TRAIN = True
AUG_NUM_COPIES = 4  # number of synthetic copies per original sequence
AUG_METHOD = 'timewarp'  # 'timewarp' | 'mixup' | 'scale' | 'noise' | None

MIXUP_ALPHA = 1.0  # Controls interpolation strength in mixup augmentation (0=no mixing, 1=equal weighting)
GAIN_RANGE = (0.9, 1.1)
WARP_STRENGTH = 0.1
AUG_NOISE_SCALE = 0.02  # used only when AUG_METHOD == 'noise'
AUG_NOISE_MIN_STD = 1e-3  # absolute floor to avoid zero-noise sequences

SHOW_PER_RAT_TABLES = False


# Explicit train/validation rat splits.
TRAIN_RATS = [
    'Rat_01',
    'Rat_02',
    'Rat_03',
    'Rat_04',
    'Rat_05',
    'Rat_06',
    
    
    
]
VAL_RATS = [
    'Rat_07',
    'Rat_08',
    'Rat_09',
    
]

COLORS = {
    'myoglobin': '#2E86AB',
    'troponin': '#A23B72',
    'ck': '#F18F01',
}

BIOMARKER_CONFIGS = {
    'myoglobin': {'column': 'Myoglobin', 'label': 0},
    'troponin': {'column': 'Troponin', 'label': 1},
    'ck': {'column': 'CK', 'label': 2},
}

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(SEED)

df = pd.read_csv(DATA_PATH)

# Find all unique Rat_IDs in the dataset
unique_rats = df['Rat_ID'].unique()
print(f"Number of different rats: {len(unique_rats)}")
print(f"Rat IDs: {unique_rats}")

def group_rat_data(df):
    rat_data_local = {}
    for rid in df['Rat_ID'].unique():
        sub = df[df['Rat_ID'] == rid]
        rat_data_local[rid] = {
            key: sub[[cfg['column'], 'Time(min)']].copy()
            for key, cfg in BIOMARKER_CONFIGS.items()
        }
    return rat_data_local

# Extract and visualize the data
rat_data = group_rat_data(df)
if SHOW_PER_RAT_TABLES:
    for rid in unique_rats:
        print(f"\n--- Data for {rid} ---")
        for key, cfg in BIOMARKER_CONFIGS.items():
            print(f"\n{cfg['column']} data:")
            print(rat_data[rid][key])

def plot_all_rats_time_series(rat_ids, rat_data_map, biomarker_cfgs, colors_map, tmin, tmax):
    # --- DATA EXPORT FOR ORIGIN (FIGURE 1) ---
    export_rows = []
    for rid in rat_ids:
        for key, cfg in biomarker_cfgs.items():
            series = rat_data_map[rid][key]
            # series has columns: [ColumnName, 'Time(min)']
            val_col = cfg['column']
            for t, v in zip(series['Time(min)'], series[val_col]):
                export_rows.append({
                    'RatID': rid,
                    'Biomarker': val_col,  # Categorical for coloring
                    'Time': t,             # X-axis
                    'Value': v             # Y-axis
                })
    
    csv_path = f"{OUTPUT_DIR}/Figure1_AllRats_TimeSeries.csv"
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    # -------------------------------

    # (Keep your existing plotting code below exactly as it was)
    fig, ax = plt.subplots(figsize=(12, 6))
    handled = set()
    for rid in rat_ids:
        for key, cfg in biomarker_cfgs.items():
            series = rat_data_map[rid][key]
            label = cfg['column'] if cfg['column'] not in handled else None
            ax.plot(
                series['Time(min)'],
                series[cfg['column']],
                color=colors_map.get(key, '#555555'),
                linewidth=1.5,
                label=label,
                alpha=0.8,
            )
            handled.add(cfg['column'])
    ax.set_title('Biomarker Levels Over Time (all rats)')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Concentration')
    ax.set_xlim(tmin, tmax)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_distribution_histograms(train_ids, val_ids, biomarker_cfgs, colors_map):
    fig, axes = plt.subplots(1, len(biomarker_cfgs), figsize=(5 * len(biomarker_cfgs), 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, cfg) in zip(axes, sorted(biomarker_cfgs.items(), key=lambda kv: kv[1]['label'])):
        train_vals = np.concatenate(
            [rat_data[rid][key][cfg['column']].values for rid in train_ids]
        )
        val_vals = np.concatenate(
            [rat_data[rid][key][cfg['column']].values for rid in val_ids]
        )
        ax.hist(train_vals, bins=30, alpha=0.6, density=True, color=colors_map.get(key, '#555555'), label='Train')
        ax.hist(val_vals, bins=30, alpha=0.6, density=True, histtype='step', color=colors_map.get(key, '#000000'), label='Validation')
        ax.set_title(cfg['column'])
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Density')
        ax.legend()
    fig.suptitle('Train vs Validation Distributions')
    plt.tight_layout()
    plt.show()

def plot_train_histograms(obs_list, future_list, labels, title):
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        print(f"No sequences available for {title} histogram.")
        return
    obs_arr = np.stack(obs_list)
    fut_arr = np.stack(future_list)

    export_rows = []
    # Map numerical labels back to string names for easier reading in Origin
    label_map = {cfg['label']: cfg['column'] for cfg in BIOMARKER_CONFIGS.values()}
    
    for i in range(len(labels)):
        # Concatenate observed and future for the full distribution
        full_seq = np.concatenate([obs_arr[i], fut_arr[i]])
        b_name = label_map[labels[i]]
        
        for val in full_seq:
            export_rows.append({
                'Biomarker': b_name,
                'Value': val
            })

    # Clean filename from title
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    csv_path = f"{OUTPUT_DIR}/Figure3_Histogram_{safe_title}.csv"
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    fig, axes = plt.subplots(1, len(BIOMARKER_CONFIGS), figsize=(5 * len(BIOMARKER_CONFIGS), 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, cfg) in zip(axes, sorted(BIOMARKER_CONFIGS.items(), key=lambda kv: kv[1]['label'])):
        mask = labels_arr == cfg['label']
        if not np.any(mask):
            ax.set_axis_off()
            continue
        values = np.concatenate([obs_arr[mask].ravel(), fut_arr[mask].ravel()])
        ax.hist(values, bins=30, density=True, color=COLORS.get(key, '#555555'), alpha=0.75)
        ax.set_title(cfg['column'])
        ax.set_xlabel('Concentration')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_train_time_series(obs_list, future_list, labels, metadata, title):
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        print(f"No sequences available for {title} time-series.")
        return
    obs_arr = np.stack(obs_list)
    fut_arr = np.stack(future_list)

    export_rows = []
    
    for i in range(len(labels)):
        meta = metadata[i]
        # Determine if augmented
        is_aug = meta.get('augmented', False)
        aug_type = meta.get('aug_type', 'Original')
        
        # Get Time Axis
        # Note: We need to reconstruct the time axis for each specific sequence
        base_meta = metadata[meta['source_idx']] if is_aug else meta
        rat_id = base_meta['rat_id']
        biomarker = base_meta['biomarker']
        full_time_ref = rat_data[rat_id][biomarker]['Time(min)'].values
        
        # Values
        full_vals = np.concatenate([obs_arr[i], fut_arr[i]])
        # Match time length
        current_times = full_time_ref[:len(full_vals)]
        
        for t, v in zip(current_times, full_vals):
            export_rows.append({
                'SequenceID': i,          # Unique ID for this specific curve
                'RatID': rat_id,
                'Biomarker': BIOMARKER_CONFIGS[biomarker]['column'],
                'Type': 'Augmented' if is_aug else 'Original',
                'AugMethod': aug_type,
                'Time': t,
                'Value': v
            })
            
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    csv_path = f"{OUTPUT_DIR}/Figure4_TimeSeries_{safe_title}.csv"
    
    # Writing in chunks or optimizing is better for huge data, 
    # but for this dataset size, direct write is fine.
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    fig, axes = plt.subplots(1, len(BIOMARKER_CONFIGS), figsize=(5 * len(BIOMARKER_CONFIGS), 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, cfg) in zip(axes, sorted(BIOMARKER_CONFIGS.items(), key=lambda kv: kv[1]['label'])):
        ax.set_title(cfg['column'])
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Concentration')
        idxs = np.where(labels_arr == cfg['label'])[0]
        if idxs.size == 0:
            ax.set_axis_off()
            continue
        base_color = COLORS.get(key, '#555555')
        legend_used = {'Original': False, 'Augmented': False}
        for idx in idxs:
            meta = metadata[idx]
            base_meta = metadata[meta['source_idx']] if meta.get('augmented') else meta
            rat_id = base_meta['rat_id']
            biomarker = base_meta['biomarker']
            times = rat_data[rat_id][biomarker]['Time(min)'].values
            obs_vals = obs_arr[idx]
            fut_vals = fut_arr[idx]
            full_vals = np.concatenate([obs_vals, fut_vals])
            full_times = times[:full_vals.shape[0]]
            linestyle = '--' if meta.get('augmented') else '-'
            alpha = 0.6 if meta.get('augmented') else 0.8
            label = 'Augmented' if meta.get('augmented') else 'Original'
            plot_label = label if not legend_used[label] else None
            legend_used[label] = True
            ax.plot(full_times, full_vals, linestyle=linestyle, color=base_color, alpha=alpha, label=plot_label)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Split data into train and validation sets based on the explicit IDs above
train_rats = [rat for rat in TRAIN_RATS if rat in unique_rats]
val_rats = [rat for rat in VAL_RATS if rat in unique_rats]

missing_train = sorted(set(TRAIN_RATS) - set(train_rats))
missing_val = sorted(set(VAL_RATS) - set(val_rats))
if missing_train or missing_val:
    raise ValueError(
        f"Missing requested rats in dataset - train: {missing_train}, val: {missing_val}"
    )

print(f"Training rats: {train_rats}")
print(f"Validation rats: {val_rats}")

time_min, time_max = df['Time(min)'].min(), df['Time(min)'].max()
plot_all_rats_time_series(unique_rats, rat_data, BIOMARKER_CONFIGS, COLORS, time_min, time_max)
plot_distribution_histograms(train_rats, val_rats, BIOMARKER_CONFIGS, COLORS)

biomarkers = [
    cfg['column'] for _, cfg in sorted(BIOMARKER_CONFIGS.items(), key=lambda kv: kv[1]['label'])
]
print("Summary Statistics:")
print("=" * 50)

# %%
for biomarker in biomarkers:
    print(f"\n{biomarker}:")
    
    # Collect train and val data
    train_values = []
    val_values = []
    
    for rat_id in train_rats:
        biomarker_data = rat_data[rat_id][biomarker.lower()]
        train_values.extend(biomarker_data[biomarker].values)
    
    for rat_id in val_rats:
        biomarker_data = rat_data[rat_id][biomarker.lower()]
        val_values.extend(biomarker_data[biomarker].values)
    
    train_values = np.array(train_values)
    val_values = np.array(val_values)
    
    print(f"Train - Mean: {train_values.mean():.2f}, Std: {train_values.std():.2f}, "
          f"Min: {train_values.min():.2f}, Max: {train_values.max():.2f}")
    print(f"Validation - Mean: {val_values.mean():.2f}, Std: {val_values.std():.2f}, "
          f"Min: {val_values.min():.2f}, Max: {val_values.max():.2f}")


# Prepare the dataset
num_biomarkers = len(BIOMARKER_CONFIGS)
label_to_display = {cfg['label']: cfg['column'] for cfg in BIOMARKER_CONFIGS.values()}


def compute_biomarker_stats(obs_array, future_array, labels):
    stats = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        combined = np.concatenate([obs_array[mask], future_array[mask]], axis=1)
        mean = combined.mean()
        std = combined.std()
        if std < 1e-6:
            std = 1.0
        stats[int(label)] = {"mean": float(mean), "std": float(std)}
    return stats


def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def collect_observed_times(metadata_list):
    times = []
    for meta in metadata_list:
        base_meta = metadata_list[meta['source_idx']] if meta.get('augmented') else meta
        rat_id = base_meta['rat_id']
        biomarker_key = base_meta['biomarker']
        time_series = rat_data[rat_id][biomarker_key]['Time(min)'].values
        times.append(time_series[:OBS_WINDOW])
    return np.asarray(times, dtype=np.float32)


def build_sequence_lists(rats, time_grid=None):
    obs_list = []
    future_list = []
    labels = []
    metadata = []
    for rat_id in rats:
        for biomarker_key, cfg in BIOMARKER_CONFIGS.items():
            biomarker_df = rat_data[rat_id][biomarker_key]
            value_series = biomarker_df[cfg['column']].values
            time_series = biomarker_df['Time(min)'].values
            if len(value_series) <= OBS_WINDOW:
                continue
            candidate_grid = time_series[OBS_WINDOW:]
            if time_grid is None:
                time_grid = candidate_grid
            else:
                if len(candidate_grid) != len(time_grid) or not np.allclose(candidate_grid, time_grid):
                    raise ValueError(
                        f"Inconsistent prediction window for {rat_id} - {cfg['column']}. "
                        "Ensure all sequences share the same time grid."
                    )
            obs_list.append(value_series[:OBS_WINDOW])
            future_list.append(value_series[OBS_WINDOW:])
            labels.append(cfg['label'])
            metadata.append({'rat_id': rat_id, 'biomarker': biomarker_key})
    return obs_list, future_list, labels, metadata, time_grid


def augment_sequences_with_noise(obs_list, future_list, labels, metadata):
    if not AUGMENT_TRAIN or AUG_NUM_COPIES <= 0:
        return obs_list, future_list, labels, metadata

    augmented_obs = list(obs_list)
    augmented_future = list(future_list)
    augmented_labels = list(labels)
    augmented_metadata = list(metadata)

    for idx, (obs_seq, future_seq, label, meta) in enumerate(zip(obs_list, future_list, labels, metadata)):
        base_series = np.concatenate([obs_seq, future_seq])
        base_std = np.std(base_series)
        noise_std = max(base_std * AUG_NOISE_SCALE, AUG_NOISE_MIN_STD)

        for aug_idx in range(1, AUG_NUM_COPIES + 1):
            obs_noise = np.random.normal(0.0, noise_std, size=obs_seq.shape)
            future_noise = np.random.normal(0.0, noise_std, size=future_seq.shape)

            obs_aug = obs_seq + obs_noise
            future_aug = future_seq + future_noise

            augmented_obs.append(obs_aug)
            augmented_future.append(future_aug)
            augmented_labels.append(label)

            meta_copy = dict(meta)
            meta_copy['augmented'] = True
            meta_copy['aug_type'] = 'noise'
            meta_copy['aug_index'] = aug_idx
            meta_copy['source_idx'] = idx
            augmented_metadata.append(meta_copy)

    return augmented_obs, augmented_future, augmented_labels, augmented_metadata


def augment_sequences_mixup(obs_list, future_list, labels, metadata):
    if AUG_NUM_COPIES <= 0:
        return obs_list, future_list, labels, metadata

    augmented_obs = list(obs_list)
    augmented_future = list(future_list)
    augmented_labels = list(labels)
    augmented_metadata = list(metadata)

    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    alpha = max(MIXUP_ALPHA, 1e-6)

    for idx, (obs_seq, future_seq, label, meta) in enumerate(zip(obs_list, future_list, labels, metadata)):
        candidates = label_to_indices[label]
        for aug_idx in range(1, AUG_NUM_COPIES + 1):
            if len(candidates) > 1:
                partner_idx = idx
                while partner_idx == idx:
                    partner_idx = int(np.random.choice(candidates))
            else:
                partner_idx = idx

            lam = np.random.beta(alpha, alpha)
            obs_partner = obs_list[partner_idx]
            future_partner = future_list[partner_idx]

            obs_aug = lam * obs_seq + (1.0 - lam) * obs_partner
            future_aug = lam * future_seq + (1.0 - lam) * future_partner

            augmented_obs.append(obs_aug)
            augmented_future.append(future_aug)
            augmented_labels.append(label)

            meta_copy = dict(meta)
            meta_copy['augmented'] = True
            meta_copy['aug_type'] = 'mixup'
            meta_copy['aug_index'] = aug_idx
            meta_copy['source_idx'] = idx
            meta_copy['partner_idx'] = partner_idx
            meta_copy['mixup_lambda'] = float(lam)
            augmented_metadata.append(meta_copy)

    return augmented_obs, augmented_future, augmented_labels, augmented_metadata


def augment_sequences_gain(obs_list, future_list, labels, metadata):
    if AUG_NUM_COPIES <= 0:
        return obs_list, future_list, labels, metadata

    augmented_obs = list(obs_list)
    augmented_future = list(future_list)
    augmented_labels = list(labels)
    augmented_metadata = list(metadata)

    low, high = GAIN_RANGE

    for idx, (obs_seq, future_seq, label, meta) in enumerate(zip(obs_list, future_list, labels, metadata)):
        for aug_idx in range(1, AUG_NUM_COPIES + 1):
            scale = np.random.uniform(low, high)
            obs_aug = obs_seq * scale
            future_aug = future_seq * scale

            augmented_obs.append(obs_aug)
            augmented_future.append(future_aug)
            augmented_labels.append(label)

            meta_copy = dict(meta)
            meta_copy['augmented'] = True
            meta_copy['aug_type'] = 'gain'
            meta_copy['aug_index'] = aug_idx
            meta_copy['source_idx'] = idx
            meta_copy['gain_scale'] = float(scale)
            augmented_metadata.append(meta_copy)

    return augmented_obs, augmented_future, augmented_labels, augmented_metadata


def augment_sequences_timewarp(obs_list, future_list, labels, metadata):
    if AUG_NUM_COPIES <= 0:
        return obs_list, future_list, labels, metadata

    augmented_obs = list(obs_list)
    augmented_future = list(future_list)
    augmented_labels = list(labels)
    augmented_metadata = list(metadata)

    for idx, (obs_seq, future_seq, label, meta) in enumerate(zip(obs_list, future_list, labels, metadata)):
        rat_id = meta['rat_id']
        biomarker_key = meta['biomarker']
        biomarker_df = rat_data[rat_id][biomarker_key]
        time_series = biomarker_df['Time(min)'].values

        total_len = len(obs_seq) + len(future_seq)
        if len(time_series) != total_len:
            time_series = np.linspace(0.0, float(total_len - 1), total_len)

        total_duration = time_series[-1] - time_series[0]
        if total_duration <= 0:
            continue

        u = (time_series - time_series[0]) / total_duration
        base_values = np.concatenate([obs_seq, future_seq])

        for aug_idx in range(1, AUG_NUM_COPIES + 1):
            coeff = np.random.uniform(-WARP_STRENGTH, WARP_STRENGTH)
            warp = u + coeff * u * (1.0 - u)
            warp = np.clip(warp, 0.0, 1.0)
            warp[0] = 0.0
            warp[-1] = 1.0

            warp = np.maximum.accumulate(warp)
            if warp[-1] == 0.0:
                continue
            warp /= warp[-1]

            values_warped = np.interp(warp, u, base_values)
            obs_aug = values_warped[:len(obs_seq)]
            future_aug = values_warped[len(obs_seq):]

            augmented_obs.append(obs_aug)
            augmented_future.append(future_aug)
            augmented_labels.append(label)

            meta_copy = dict(meta)
            meta_copy['augmented'] = True
            meta_copy['aug_type'] = 'timewarp'
            meta_copy['aug_index'] = aug_idx
            meta_copy['source_idx'] = idx
            meta_copy['warp_coef'] = float(coeff)
            augmented_metadata.append(meta_copy)

    return augmented_obs, augmented_future, augmented_labels, augmented_metadata


def augment_sequences(obs_list, future_list, labels, metadata):
    if not AUGMENT_TRAIN or AUG_NUM_COPIES <= 0:
        return obs_list, future_list, labels, metadata

    method = (AUG_METHOD or 'none').lower()
    if method == 'noise':
        return augment_sequences_with_noise(obs_list, future_list, labels, metadata)
    if method == 'mixup':
        return augment_sequences_mixup(obs_list, future_list, labels, metadata)
    if method == 'scale':
        return augment_sequences_gain(obs_list, future_list, labels, metadata)
    if method == 'timewarp':
        return augment_sequences_timewarp(obs_list, future_list, labels, metadata)

    print(f"Unknown augmentation method '{AUG_METHOD}'. Skipping augmentation.")
    return obs_list, future_list, labels, metadata


train_obs_list, train_future_list, train_labels, train_metadata, x_train = build_sequence_lists(train_rats)
val_obs_list, val_future_list, val_labels, val_metadata, x_val = build_sequence_lists(val_rats, time_grid=x_train)

plot_train_histograms(train_obs_list, train_future_list, train_labels, 'Train Data (Original)')
plot_train_time_series(train_obs_list, train_future_list, train_labels, train_metadata, 'Train Time Series (Original)')

original_train_count = len(train_obs_list)
if AUGMENT_TRAIN:
    print(
        f"Augmenting training sequences with method='{AUG_METHOD}' and copies per sequence={AUG_NUM_COPIES}"
    )
train_obs_list, train_future_list, train_labels, train_metadata = augment_sequences(
    train_obs_list,
    train_future_list,
    train_labels,
    train_metadata,
)
if AUGMENT_TRAIN:
    print(f"Training sequences: original={original_train_count}, augmented_total={len(train_obs_list)}")

plot_train_histograms(train_obs_list, train_future_list, train_labels, 'Train Data (Augmented)')
plot_train_time_series(train_obs_list, train_future_list, train_labels, train_metadata, 'Train Time Series (Augmented)')

if not train_obs_list:
    raise ValueError("No training sequences available after including all biomarkers.")
if not val_obs_list:
    raise ValueError("No validation sequences available after including all biomarkers.")

y_train = np.stack(train_future_list)
y_val = np.stack(val_future_list)
z_obs_train = np.stack(train_obs_list)
z_obs_val = np.stack(val_obs_list)
z_biomarker_train = np.array(train_labels)
z_biomarker_val = np.array(val_labels)

z_biomarker_train_onehot = to_one_hot(z_biomarker_train, num_biomarkers)
z_biomarker_val_onehot = to_one_hot(z_biomarker_val, num_biomarkers)

missing_val_labels = sorted(set(z_biomarker_val.tolist()) - set(z_biomarker_train.tolist()))
if missing_val_labels:
    raise ValueError(
        f"Validation set contains biomarker labels absent from training data: {missing_val_labels}"
    )

print("\nSequences per biomarker (train):")
for label in sorted(label_to_display):
    count = int(np.sum(z_biomarker_train == label))
    print(f"  {label_to_display[label]}: {count}")

print("\nSequences per biomarker (validation):")
for label in sorted(label_to_display):
    count = int(np.sum(z_biomarker_val == label))
    print(f"  {label_to_display[label]}: {count}")


# Normalize the data
x_train_raw = x_train.copy()
x_val_raw = x_val.copy()
y_train_raw = y_train.copy()
y_val_raw = y_val.copy()
z_obs_train_raw = z_obs_train.copy()
z_obs_val_raw = z_obs_val.copy()

biomarker_stats = compute_biomarker_stats(z_obs_train_raw, y_train_raw, z_biomarker_train)

z_obs_train_norm = np.full_like(z_obs_train_raw, fill_value=np.nan, dtype=np.float32)
y_train_norm = np.full_like(y_train_raw, fill_value=np.nan, dtype=np.float32)
z_obs_val_norm = np.full_like(z_obs_val_raw, fill_value=np.nan, dtype=np.float32)
y_val_norm = np.full_like(y_val_raw, fill_value=np.nan, dtype=np.float32)

for label, stats in biomarker_stats.items():
    mean = stats['mean']
    std = stats['std']

    train_mask = z_biomarker_train == label
    if np.any(train_mask):
        z_obs_train_norm[train_mask] = ((z_obs_train_raw[train_mask] - mean) / std).astype(np.float32)
        y_train_norm[train_mask] = ((y_train_raw[train_mask] - mean) / std).astype(np.float32)

    val_mask = z_biomarker_val == label
    if np.any(val_mask):
        z_obs_val_norm[val_mask] = ((z_obs_val_raw[val_mask] - mean) / std).astype(np.float32)
        y_val_norm[val_mask] = ((y_val_raw[val_mask] - mean) / std).astype(np.float32)

missing_mask = np.isnan(z_obs_val_norm).any(axis=1)
if np.any(missing_mask):
    missing_labels = sorted(set(z_biomarker_val[missing_mask].tolist()))
    raise ValueError(f"Missing biomarker stats for validation labels: {missing_labels}")

# Replace any potential lingering NaNs (e.g., sequences normalized but numerical precision issues) with zeros
z_obs_train_norm = np.nan_to_num(z_obs_train_norm, copy=False)
y_train_norm = np.nan_to_num(y_train_norm, copy=False)
z_obs_val_norm = np.nan_to_num(z_obs_val_norm, copy=False)
y_val_norm = np.nan_to_num(y_val_norm, copy=False)

x_train_mean, x_train_std = x_train_raw.mean(), x_train_raw.std()
if x_train_std == 0:
    x_train_std = 1.0
x_train_norm = (x_train_raw - x_train_mean) / x_train_std
x_val_norm = (x_val_raw - x_train_mean) / x_train_std

z_time_train_raw = collect_observed_times(train_metadata)
z_time_val_raw = collect_observed_times(val_metadata)
z_time_train_norm = ((z_time_train_raw - x_train_mean) / x_train_std).astype(np.float32)
z_time_val_norm = ((z_time_val_raw - x_train_mean) / x_train_std).astype(np.float32)

# Build the z_tokens for the training and validation sets
def build_z_tokens(obs_norm, biomarker_onehot, time_norm):
    obs_tokens = obs_norm[..., None].astype(np.float32)  # (batch, obs_window, 1)
    time_tokens = time_norm[..., None].astype(np.float32)
    biomarker_tokens = np.repeat(biomarker_onehot[:, None, :], obs_tokens.shape[1], axis=1).astype(np.float32)
    return np.concatenate([obs_tokens, time_tokens, biomarker_tokens], axis=-1).astype(np.float32)


z_train_tokens = build_z_tokens(z_obs_train_norm, z_biomarker_train_onehot, z_time_train_norm)
z_val_tokens = build_z_tokens(z_obs_val_norm, z_biomarker_val_onehot, z_time_val_norm)

print("Data shapes:")
print(f"x_train: {x_train_norm.shape}, x_val: {x_val_norm.shape}")
print(f"y_train: {y_train_norm.shape}, y_val: {y_val_norm.shape}")
print(f"z_obs_train: {z_obs_train_norm.shape}, z_obs_val: {z_obs_val_norm.shape}")
print(f"z_biomarker_train: {z_biomarker_train_onehot.shape}, z_biomarker_val: {z_biomarker_val_onehot.shape}")
print(f"z_train_tokens: {z_train_tokens.shape}, z_val_tokens: {z_val_tokens.shape}")

x_train_ts = torch.tensor(x_train_norm, dtype=torch.float32).to(device)
y_train_ts = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(-1).to(device)
z_train_ts = torch.tensor(z_train_tokens, dtype=torch.float32).to(device)

x_val_ts = torch.tensor(x_val_norm, dtype=torch.float32).to(device)
y_val_ts = torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(-1).to(device)
z_val_ts = torch.tensor(z_val_tokens, dtype=torch.float32).to(device)

# %%
# print(f'First 5 rows of x_train_ts: {x_train_ts[:5]}')
# print(f'First 5 rows of y_train_ts: {y_train_ts[:5]}')
# print(f'First 5 rows of z_train_ts: {z_train_ts[:5]}')
# print(f'First 5 rows of x_val_ts: {x_val_ts[:5]}')
# print(f'First 5 rows of y_val_ts: {y_val_ts[:5]}')
# print(f'First 5 rows of z_val_ts: {z_val_ts[:5]}')
print('Train data shapes:')
print(f'x_train_ts: {x_train_ts.shape}')
print(f'y_train_ts: {y_train_ts.shape}')
print(f'z_train_ts: {z_train_ts.shape}')
print(f'x_val_ts: {x_val_ts.shape}')
print(f'y_val_ts: {y_val_ts.shape}')
print(f'z_val_ts: {z_val_ts.shape}')

# %%
# Build model
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, activation, use_norms):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_norms = use_norms

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_z = nn.LayerNorm(d_model)
        
        # Initialize the attribute to prevent errors if accessed before forward
        self.last_attn_weights = None 

    def update_dropout_rate(self, new_rate):
        self.dropout_rate = new_rate

        self.cross_attn.dropout = new_rate

        for module in self.ffn:
            if isinstance(module, nn.Dropout):
                module.p = new_rate

        self.dropout.p = new_rate

    def forward(self, x, z_embedding):
        if self.use_norms:
            x_norm = self.norm_cross(x)
            z_context = self.norm_z(z_embedding)
        else:
            x_norm = x
            z_context = z_embedding

        # --- MODIFIED SECTION START ---
        # We set need_weights=True to capture the attention map
        cross_attn_output, attn_weights = self.cross_attn(
            query=x_norm,
            key=z_context,
            value=z_context,
            need_weights=True,  # <--- Essential change
        )
        # Store weights for visualization later
        # attn_weights shape: (Batch, Target_Seq(x), Source_Seq(z))
        # Note: PyTorch MultiheadAttention averages weights over heads if average_attn_weights is True (default)
        self.last_attn_weights = attn_weights 
        # --- MODIFIED SECTION END ---

        x = x + self.dropout(cross_attn_output)

        residual = x
        if self.use_norms:
            x_norm = self.norm_ffn(x)
        else:
            x_norm = x
        x = residual + self.dropout(self.ffn(x_norm))

        return x


class CrossAttnTransformer(nn.Module):
    def __init__(
        self,
        problem_dim,
        output_dim,
        z_dim,
        num_blocks=3,
        latent_dim=128,
        num_heads=4,
        dropout_rate=0.1,
        activation=nn.GELU(),
        use_norms=True,
    ):
        super().__init__()

        self.problem_dim = problem_dim
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_norms = use_norms

        self.x_embedding = nn.Linear(self.problem_dim, self.latent_dim)
        self.z_embedding = nn.Linear(self.z_dim, self.latent_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(self.latent_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    d_model=self.latent_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_norms=use_norms,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_projection = nn.Linear(self.latent_dim, self.output_dim)

    def update_dropout_rate(self, new_rate):
        self.dropout_rate = new_rate
        for block in self.transformer_blocks:
            block.update_dropout_rate(new_rate)

    def forward(self, x, z):
        x_emb = self.x_embedding(x)
        pos_enc_x = self.positional_encoding(x).to(x_emb.device)
        x_emb = x_emb + pos_enc_x.to(x_emb.dtype)
        z_emb = self.z_embedding(z)

        for block in self.transformer_blocks:
            x_emb = block(x_emb, z_emb)

        return self.final_projection(x_emb)

    
train_dataset = torch.utils.data.TensorDataset(z_train_ts, y_train_ts)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
val_dataset = torch.utils.data.TensorDataset(z_val_ts, y_val_ts)
val_loader = DataLoader(val_dataset, batch_size=len(z_val_ts), shuffle=False)

# %%
model = CrossAttnTransformer(
    problem_dim=1,
    output_dim=1,
    z_dim=z_train_ts.shape[-1],
    num_blocks=NUM_BLOCKS,
    latent_dim=LATENT_DIM,
    num_heads=NUM_HEADS,
    dropout_rate=DROP_RATE,
    activation=nn.GELU(),
    use_norms=True,
).to(device)

if OPTIMIZER == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200) # Add this
elif OPTIMIZER == 'SOAP':
    from soap import SOAP
    optimizer = SOAP(model.parameters(), lr=LR)


num_epochs = NUM_EPOCHS
train_loss_history = []
val_loss_history = []

pbar = trange(num_epochs, desc="Training", unit="epoch")
# Training loop
for epoch in pbar:
    model.train()
    train_loss = 0.0

    for z_t_ts, y_t_ts in train_loader:
        optimizer.zero_grad()

        # Expand the x-grid → (batch, N_steps, 1)
        x_expanded = (x_train_ts.unsqueeze(0).unsqueeze(-1).expand(z_t_ts.shape[0], -1, -1))
        y_pred = model(x_expanded, z_t_ts)

        # Loss on the future prediction
        loss = F.mse_loss(y_pred, y_t_ts)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for z_v_ts, y_v_ts in val_loader:
            x_exp = (x_val_ts.unsqueeze(0).unsqueeze(-1).expand(z_v_ts.shape[0], -1, -1))
            yp = model(x_exp, z_v_ts)
            
            loss_v = F.mse_loss(yp, y_v_ts)
            val_loss += loss_v.item()

    val_loss /= len(val_loader)
    # Update the learning rate based on the average loss
    scheduler.step(val_loss)  # <--- THIS GOES HERE (Outside the loop)
    val_loss_history.append(val_loss)

    if epoch % 100 == 0:
        pbar.set_postfix(
            train_loss=f"{train_loss:.4e}",
            val_loss=f"{val_loss:.4e}"
        )

# %%
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
#plt.loglog()
plt.show()
# --- DATA EXPORT FOR ORIGIN (FIGURE 5) ---
loss_df = pd.DataFrame({
    'Epoch': range(1, len(train_loss_history) + 1),
    'Train_Loss': train_loss_history,
    'Validation_Loss': val_loss_history
})
csv_path = f"{OUTPUT_DIR}/Figure5_LossCurve.csv"
loss_df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
# -----------------------------------------

# === PASTE THE NEW FUNCTION HERE ===
def calculate_advanced_metrics(y_true, y_pred, time_grid, threshold=0.5):
    metrics = {}
    
    # 1. AUC Error
    auc_true = trapz(y_true, time_grid)
    auc_pred = trapz(y_pred, time_grid)
    metrics['AUC_Diff'] = auc_pred - auc_true
    metrics['AUC_Rel_Error'] = abs(metrics['AUC_Diff']) / (auc_true + 1e-6)

    # 2. DTW (Check if shape matches)
    # Reshape for fastdtw
    distance, path = fastdtw(y_true.reshape(-1,1), y_pred.reshape(-1,1), dist=euclidean)
    metrics['DTW_Distance'] = distance

    # 3. Time-to-Threshold (TTE)
    true_cross = np.where(y_true > threshold)[0]
    pred_cross = np.where(y_pred > threshold)[0]
    if len(true_cross) > 0 and len(pred_cross) > 0:
        metrics['TTE_Error'] = time_grid[pred_cross[0]] - time_grid[true_cross[0]]
    else:
        metrics['TTE_Error'] = None

    # 4. Peak Errors
    metrics['Peak_Amp_Error'] = np.max(y_pred) - np.max(y_true)
    metrics['Peak_Time_Error'] = time_grid[np.argmax(y_pred)] - time_grid[np.argmax(y_true)]

    return metrics
# ===================================


# %%
def evaluate_and_plot(dataset, x_grid_ts, x_grid_raw, y_raw, z_labels, metadata, biomarker_stats_map, eval_name):
    print(f"Evaluating {eval_name} data...")
    model.eval()
    z_t_ts, _y_t_ts = dataset.tensors
    x_expanded = x_grid_ts.unsqueeze(0).unsqueeze(-1).expand(z_t_ts.shape[0], -1, -1)
    
    with torch.no_grad():
        y_pred_ts = model(x_expanded.to(device), z_t_ts.to(device))
    y_pred = y_pred_ts.detach().cpu().numpy()

    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 0]

    labels_arr = np.asarray(z_labels, dtype=np.int64)
    means = np.array([biomarker_stats_map[int(lbl)]['mean'] for lbl in labels_arr], dtype=y_pred.dtype)
    stds = np.array([biomarker_stats_map[int(lbl)]['std'] for lbl in labels_arr], dtype=y_pred.dtype)
    y_pred_denorm = y_pred * stds[:, None] + means[:, None]

    # Basic MSE calculation
    diff = y_raw - y_pred_denorm
    mse = np.mean(diff ** 2, axis=1)
    rel_l2 = np.linalg.norm(diff, axis=1) / np.linalg.norm(y_raw, axis=1)

    export_rows = []
    summary_metrics_rows = [] # New list for the advanced metrics

    for i in range(len(metadata)):
        meta = metadata[i]
        biomarker_name = BIOMARKER_CONFIGS[meta['biomarker']]['column']
        
        # --- NEW: Calculate Advanced Metrics ---
        # Define a threshold for TTE (e.g., 0.5 or custom per biomarker)
        adv_metrics = calculate_advanced_metrics(y_raw[i], y_pred_denorm[i], x_grid_raw, threshold=0.5)
        
        # Print to console (optional, or just print summary later)
        print(f"{meta['rat_id']} - {biomarker_name}: MSE={mse[i]:.4f}, DTW={adv_metrics['DTW_Distance']:.2f}, PeakErr={adv_metrics['Peak_Amp_Error']:.2f}")

        # Save for Summary CSV
        summary_row = {
            'RatID': meta['rat_id'],
            'Biomarker': biomarker_name,
            'SequenceIndex': i,
            'MSE': mse[i],
            'RL2E': rel_l2[i],
            **adv_metrics # Unpack all new metrics here
        }
        summary_metrics_rows.append(summary_row)

        # Save for Time-Series CSV (Figure 6/7)
        for t_idx, t_val in enumerate(x_grid_raw):
            export_rows.append({
                'RatID': meta['rat_id'],
                'Biomarker': biomarker_name,
                'SequenceIndex': i,
                'Time': t_val,
                'GroundTruth': y_raw[i, t_idx],
                'Prediction': y_pred_denorm[i, t_idx],
                'RL2E': rel_l2[i]
            })

    # Save the detailed Time Series (Existing)
    csv_name = f"Figure7_Predictions_{eval_name}.csv" if eval_name == "Validation" else f"Figure6_Predictions_{eval_name}.csv"
    csv_path = f"{OUTPUT_DIR}/{csv_name}"
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    # --- NEW: Save the Metrics Summary CSV ---
    metrics_csv_name = f"Metrics_Summary_{eval_name}.csv"
    pd.DataFrame(summary_metrics_rows).to_csv(f"{OUTPUT_DIR}/{metrics_csv_name}", index=False)
    print(f"Saved {OUTPUT_DIR}/{metrics_csv_name} (Contains AUC, DTW, TTE)")

    # -----------------------------------------

    seq_count = len(metadata)
    if seq_count:
        n_cols = min(3, seq_count)
        n_rows = math.ceil(seq_count / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis('off')
        for idx in range(seq_count):
            ax = axes[idx]
            ax.axis('on')
            meta = metadata[idx]
            display_name = BIOMARKER_CONFIGS[meta['biomarker']]['column']
            ax.plot(x_grid_raw, y_raw[idx, :].flatten(), 'o-', label='Ground Truth')
            ax.plot(x_grid_raw, y_pred_denorm[idx, :].flatten(), 'x-', label='Prediction')
            ax.set_xlabel("Time(min)")
            ax.set_ylabel("Concentration")
            ax.set_title(f"{meta['rat_id']} - {display_name} - RL2E={rel_l2[idx]:.4f}")
            ax.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No {eval_name.lower()} sequences available for plotting.")
    
# Evaluate Train and Validation
evaluate_and_plot(
    dataset=train_dataset,
    x_grid_ts=x_train_ts,
    x_grid_raw=x_train_raw,
    y_raw=y_train_raw,
    z_labels=z_biomarker_train,
    metadata=train_metadata,
    biomarker_stats_map=biomarker_stats,
    eval_name="Train",
)

evaluate_and_plot(
    dataset=val_dataset,
    x_grid_ts=x_val_ts,
    x_grid_raw=x_val_raw,
    y_raw=y_val_raw,
    z_labels=z_biomarker_val,
    metadata=val_metadata,
    biomarker_stats_map=biomarker_stats,
    eval_name="Validation",
)