import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import ttest_ind
import mne

# === Load features and metadata ===
features_dir = "features_per_subject"
df = pd.concat([
    pd.read_csv(os.path.join(features_dir, f))
    for f in os.listdir(features_dir) if f.endswith(".csv")
])

meta = pd.read_csv("dataset/participants.tsv", sep="\t")
df = df.merge(meta, left_on="subject_id", right_on="participant_id")

# === Pivot data to get power features per band ===
power_df = df.pivot_table(
    index=["subject_id", "channel", "Group"],
    columns="band",
    values="mean"
).reset_index()

power_df.columns.name = None
power_df = power_df.rename(columns={
    'delta': 'delta_power',
    'theta': 'theta_power',
    'alpha': 'alpha_power',
    'beta': 'beta_power',
    'gamma': 'gamma_power'
})

# === Compute ratio features ===
power_df["alpha_theta"] = power_df["alpha_power"] / power_df["theta_power"]
power_df["beta_theta"] = power_df["beta_power"] / power_df["theta_power"]

# === EEG bands and groups ===
features = [
    "delta_power", "theta_power", "alpha_power",
    "beta_power", "gamma_power", "alpha_theta", "beta_theta"
]
band_names = [
    "Delta", "Theta", "Alpha",
    "Beta", "Gamma", "Alpha/Theta", "Beta/Theta"
]
groups = ['C', 'A', 'F']

# === Load electrode layout ===
raw = mne.io.read_raw_eeglab("dataset/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set", preload=True)
channels = [ch for ch in raw.info["ch_names"] if ch in power_df["channel"].unique()]

# === Compute average log values for each band and group ===
log_avg_values = {band: {g: [] for g in groups} for band in features}

for band in features:
    for g in groups:
        values = power_df[(power_df["Group"] == g)].pivot(index="subject_id", columns="channel", values=band)
        group_avg = values.mean(axis=1)
        log_avg_values[band][g] = np.log(group_avg.dropna())

# === Plot histograms ===
for band, name in zip(features, band_names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    fig.suptitle(f'{name}: CN vs AD vs FTD')
    ax1.hist(log_avg_values[band]['C'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('CN')
    ax2.hist(log_avg_values[band]['A'], bins=20, color='salmon', edgecolor='black')
    ax2.set_title('AD')
    ax3.hist(log_avg_values[band]['F'], bins=20, color='lightgreen', edgecolor='black')
    ax3.set_title('FTD')
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Log Power')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
