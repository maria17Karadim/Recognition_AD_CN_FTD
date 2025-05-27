import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import ttest_ind
import seaborn as sns
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

# === Rename columns for consistency ===
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

# === EEG bands to use ===
features = [
    "delta_power", "theta_power", "alpha_power",
    "beta_power", "gamma_power", "alpha_theta", "beta_theta"
]
band_names = [
    "Delta", "Theta", "Alpha",
    "Beta", "Gamma", "Alpha/Theta", "Beta/Theta"
]

# === Load electrode layout from an example EEG file ===
raw = mne.io.read_raw_eeglab("dataset/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set", preload=True)
channels = [ch for ch in raw.info["ch_names"] if ch in power_df["channel"].unique()]

# === Compute and plot P-value topomaps (CN vs AD) with Cohen's d ===
p_vals_all = []
cohens_d_all = []
print("\n\U0001F4CA Minimum p-values per EEG band:")

for band, band_name in zip(features, band_names):
    band_p_vals = []
    band_d_vals = []
    for ch in channels:
        group_c = power_df[(power_df["channel"] == ch) & (power_df["Group"] == "C")][band]
        group_a = power_df[(power_df["channel"] == ch) & (power_df["Group"] == "A")][band]

        if len(group_c) > 1 and len(group_a) > 1:
            _, p = ttest_ind(group_c, group_a, nan_policy='omit')
            mean_diff = np.mean(group_c) - np.mean(group_a)
            pooled_std = np.sqrt(((np.std(group_c)**2) + (np.std(group_a)**2)) / 2)
            d = mean_diff / pooled_std if pooled_std != 0 else 0
        else:
            p = 1.0
            d = 0

        band_p_vals.append(p)
        band_d_vals.append(d)

    band_p_vals = np.nan_to_num(band_p_vals, nan=1.0)
    p_vals_all.append(band_p_vals)
    cohens_d_all.append(band_d_vals)
    print(f"{band_name}: min p = {np.min(band_p_vals):.4f}")

# === Plot P-value topomaps ===
vlim = [0, 1]
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
x = 0

for i in range(2):
    for j in range(4):
        if x < len(p_vals_all):
            vals = np.array(p_vals_all[x])
            masked_vals = np.ma.masked_where(vals > 0.05, vals)
            im, _ = mne.viz.plot_topomap(
                masked_vals, raw.info, show=False, axes=axs[i][j],
                names=channels, cmap='viridis', vlim=vlim)
            axs[i][j].set_title(band_names[x])
        else:
            axs[i][j].axis("off")
        x += 1

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=vlim[0], vmax=vlim[1]))
sm.set_array(np.linspace(*vlim))
fig.colorbar(sm, cax=cbar_ax)
fig.suptitle("P-values: CN vs AD (per EEG Band)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()

# === Compute average feature topomaps (CN, AD, FTD) ===
groups = ['C', 'A', 'F']
avg_bands = []

for feat in features:
    band_group_values = []
    for g in groups:
        tmp = power_df[(power_df["Group"] == g)].pivot(index="subject_id", columns="channel", values=feat)
        band_group_values.append(tmp.mean().reindex(channels).values)
    avg_bands.append(band_group_values)

avg_bands = np.array(avg_bands)

# === Plot average EEG band power per group ===
fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(10, 14))
for i in range(7):
    vlim = [np.nanmin(avg_bands[i]), np.nanmax(avg_bands[i])]
    for j in range(3):
        im, _ = mne.viz.plot_topomap(
            avg_bands[i][j], raw.info, axes=axs[i][j],
            show=False, names=channels, cmap="viridis", vlim=vlim)

    cbar_ax = fig.add_axes([0.92, 0.07 + 0.13*i, 0.015, 0.1])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=colors.Normalize(vmin=vlim[0], vmax=vlim[1]))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    if i == 1:
        cbar_ax.set_ylabel("Power Score", size="large")

group_labels = ['CN', 'AD', 'FTD']
for ax, col in zip(axs[0], group_labels):
    ax.set_title(col, fontsize=12)

for ax, row in zip(axs[:, 0], band_names):
    ax.annotate(row, xy=(0, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=12)

fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.suptitle("Average EEG Feature Topomaps by Group", fontsize=16)
plt.show()

# === Plot histograms of features by group ===
num_features = len(features)
cols = 3
rows = int(np.ceil(num_features / cols))
fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))

for i, feature in enumerate(features):
    ax = axs[i // cols][i % cols] if rows > 1 else axs[i % cols]
    subset = power_df[[feature, "Group"]].dropna()

    sns.histplot(
        data=subset,
        x=feature,
        hue="Group",
        ax=ax,
        kde=False,
        stat="density",
        element="step",
        common_norm=False
    )

    ax.set_title(f"Distribution of {band_names[i]}", fontsize=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

# Turn off any unused axes
for j in range(i + 1, rows * cols):
    fig.delaxes(axs[j // cols][j % cols] if rows > 1 else axs[j % cols])

fig.tight_layout()
fig.suptitle("Histograms of EEG Features by Group", fontsize=16)
plt.subplots_adjust(top=0.95)
plt.show()
