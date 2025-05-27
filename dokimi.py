import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import entropy
import antropy as ant
import networkx as nx
import traceback
from mne_connectivity import spectral_connectivity_epochs

mne.set_log_level("WARNING")

# === Configuration ===
base_dir = "dataset/derivatives"
output_dir = "features_per_subject"
os.makedirs(output_dir, exist_ok=True)

# === Load participant info ===
participants_df = pd.read_csv("dataset/participants.tsv", sep="\t")

# Frequency bands
freq_bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

for i in range(1, 89):
    subj_id = f"sub-{i:03}"
    eeg_file = os.path.join(base_dir, subj_id, "eeg", f"{subj_id}_task-eyesclosed_eeg.set")

    if not os.path.exists(eeg_file):
        print(f"❌ Missing file: {eeg_file}")
        continue

    print(f"✅ Processing {subj_id}...")

    try:
        print("🔄 Loading EEG file...")
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

        print("🔄 Cropping to 60 seconds...")
        raw.crop(tmax=60.0)

        print("🔄 Selecting EEG channels...")
        raw.pick("eeg")

        print("🔄 Filtering 0.5-45 Hz...")
        raw.filter(0.5, 45.0, method='iir', iir_params=dict(ftype='butter', order=4), phase='zero')

        print("🔄 Setting average reference...")
        raw.set_eeg_reference('average')

        print("🔄 Creating epochs...")
        epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=5.0, preload=True)

        print("🔄 Computing PSD...")
        psd_obj = epochs.compute_psd(fmin=0.5, fmax=45.0, method='welch', n_fft=256, verbose=False)
        psds = psd_obj.get_data() * 1e12
        freqs = psd_obj.freqs
        ch_names = epochs.info['ch_names']
        epoch_data = epochs.get_data()

        print("🔄 Computing connectivity...")
        con_methods = ['coh', 'plv', 'imcoh']
        con_results = {
            m: spectral_connectivity_epochs(
                epochs, method=m, mode='fourier', sfreq=raw.info['sfreq'],
                fmin=0.5, fmax=45.0, faverage=True, verbose=False
            ).get_data()
            for m in con_methods
        }

        print("✅ EEG loaded. Extracting features...")
        rows = []
        for ep_idx in range(psds.shape[0]):
            epoch_psd = psds[ep_idx]
            epoch_raw = epoch_data[ep_idx]

            for band_name, (fmin, fmax) in freq_bands.items():
                band_mask = (freqs >= fmin) & (freqs < fmax)
                band_psd = epoch_psd[:, band_mask]
                band_raw = epoch_raw.copy()

                total_power = np.sum(band_psd, axis=1, keepdims=True)
                prob_dists = band_psd / (total_power + 1e-12)
                entropy_vals = entropy(prob_dists, base=2, axis=1)
                ap_en = [ant.app_entropy(ch) for ch in band_raw]
                samp_en = [ant.sample_entropy(ch) for ch in band_raw]
                perm_en = [ant.perm_entropy(ch, normalize=True) for ch in band_raw]

                corr_matrix = np.corrcoef(band_raw)
                G = nx.from_numpy_array(np.abs(corr_matrix))
                try:
                    clustering = nx.average_clustering(G)
                    path_len = nx.average_shortest_path_length(G)
                    efficiency = nx.global_efficiency(G)
                except:
                    clustering = path_len = efficiency = np.nan

                try:
                    rand_G = nx.erdos_renyi_graph(len(G.nodes), 0.5)
                    rand_cluster = nx.average_clustering(rand_G)
                    rand_path = nx.average_shortest_path_length(rand_G)
                    gamma = clustering / rand_cluster if rand_cluster else np.nan
                    lam = path_len / rand_path if rand_path else np.nan
                    small_world = gamma / lam if lam else np.nan
                except:
                    small_world = np.nan

                for ch_idx, ch_name in enumerate(ch_names):
                    row = {
                        "subject_id": subj_id,
                        "epoch_idx": ep_idx,
                        "band": band_name,
                        "channel": ch_name,
                        "entropy": entropy_vals[ch_idx],
                        "ap_entropy": ap_en[ch_idx],
                        "sample_entropy": samp_en[ch_idx],
                        "perm_entropy": perm_en[ch_idx],
                        "sync_clustering": clustering,
                        "sync_path_length": path_len,
                        "sync_efficiency": efficiency,
                        "sync_small_worldness": small_world,
                        "mean": np.mean(band_raw[ch_idx]),
                        "variance": np.var(band_raw[ch_idx]),
                        "group": participants_df.loc[participants_df['participant_id'] == subj_id, 'Group'].values[0]
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f"features_{subj_id}.csv"), index=False)
        print(f"📁 Saved to features_{subj_id}.csv")

    except Exception as e:
        print(f"❌ Error processing {subj_id}: {e}")
        traceback.print_exc()
