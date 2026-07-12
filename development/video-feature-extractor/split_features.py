import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

CORPUS       = "MSVD"
DATASET_DIR  = f"/mnt/d/__VC-Transformer-KG/data/{CORPUS}"
METADATA_DIR = f"{DATASET_DIR}/metadata"
FEATURES_DIR = f"{DATASET_DIR}/features"

# Load the train, validation, and test DataFrames
train_df = pd.read_json(f"{METADATA_DIR}/msvd_train.json")
val_df   = pd.read_json(f"{METADATA_DIR}/msvd_val.json")
test_df  = pd.read_json(f"{METADATA_DIR}/msvd_test.json")
train_video_ids = set(train_df["video_id"].tolist())
val_video_ids   = set(val_df["video_id"].tolist())
test_video_ids  = set(test_df["video_id"].tolist())

# === Define the feature type and paths ===
for feature_type in ["Blip2QFormerMeanKF"]:
    print(f"========== Processing feature type: {feature_type} ==========")
    total_feature_path = f"{FEATURES_DIR}/{CORPUS}_{feature_type}.hdf5"
    train_feature_path = f"{FEATURES_DIR}/{CORPUS}_{feature_type}_train.hdf5"
    val_feature_path   = f"{FEATURES_DIR}/{CORPUS}_{feature_type}_val.hdf5"
    test_feature_path  = f"{FEATURES_DIR}/{CORPUS}_{feature_type}_test.hdf5"

    train_features = {}
    test_features = {}
    val_features = {}

    # Split the features into train, validation, and test sets
    #   based on the video IDs in the respective DataFrames
    print("Splitting features into train, validation, and test sets...")
    with h5py.File(total_feature_path, "r") as f:
        assert len(f.keys()) == 1970
        for key in tqdm(f.keys(), desc="Splitting features"):
            assert len(f[key].shape) == 2
            if   key in train_video_ids: train_features[key] = f[key][:]
            elif key in val_video_ids:   val_features[key] = f[key][:]
            elif key in test_video_ids:  test_features[key] = f[key][:]
            else: raise ValueError(f"Unexpected video ID: {key}")

    with h5py.File(train_feature_path, "w") as f:
        for key, value in train_features.items():
            f.create_dataset(key, data=value)
    with h5py.File(val_feature_path, "w") as f:
        for key, value in val_features.items():
            f.create_dataset(key, data=value)
    with h5py.File(test_feature_path, "w") as f:
        for key, value in test_features.items():
            f.create_dataset(key, data=value)

    # Verify the splits
    with h5py.File(train_feature_path, "r") as f:
        print(f"Number of training videos: {len(f.keys())}")
    with h5py.File(val_feature_path, "r") as f:
        print(f"Number of validation videos: {len(f.keys())}")
    with h5py.File(test_feature_path, "r") as f:
        print(f"Number of test videos: {len(f.keys())}")

    # === Analyze the distribution of the number of GOPs in the train set ===
    print(f"\n===== GOP count distribution ({CORPUS} train set, {feature_type}) =====")
    with h5py.File(train_feature_path, "r") as f:
        # Each dataset has shape (NUM_GOP, d) -> reading .shape does NOT load the data
        num_gops = np.array([f[key].shape[0] for key in f.keys()])

    print(f"Number of videos : {len(num_gops)}")
    print(f"Min / Max        : {num_gops.min()} / {num_gops.max()}")
    print(f"Mean +/- Std     : {num_gops.mean():.2f} +/- {num_gops.std():.2f}")
    print(f"Total GOPs       : {num_gops.sum()}")

    print("\nPercentiles of GOP count:")
    for p in [50, 75, 80, 85, 90, 95]:
        print(f"  P{p:<3}: {np.percentile(num_gops, p):.1f}")

    # Text histogram of the GOP count distribution
    print("\nHistogram of GOP count:")
    counts, bin_edges = np.histogram(num_gops, bins=20)
    max_count = counts.max()
    for count, lo, hi in zip(counts, bin_edges[:-1], bin_edges[1:]):
        bar = "#" * int(np.ceil(count / max_count * 50))
        print(f"  [{lo:6.1f}, {hi:6.1f}): {count:5d} {bar}")

    # Also save a histogram figure next to the feature files (if matplotlib is available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(num_gops, bins=30, edgecolor="black", alpha=0.75)
        for p in [50, 75, 90, 95]:
            v = np.percentile(num_gops, p)
            ax.axvline(v, linestyle="--", linewidth=1, label=f"P{p} = {v:.1f}")
        ax.set_xlabel("Number of GOPs per video")
        ax.set_ylabel("Number of videos")
        ax.set_title(f"GOP count distribution - {CORPUS} train set ({feature_type})")
        ax.legend()
        fig.tight_layout()
        fig_path = f"{FEATURES_DIR}/{CORPUS}_{feature_type}_train_gop_hist.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved histogram figure to: {fig_path}")
    except ImportError:
        print("\nmatplotlib not available -- skipped saving the histogram figure.")
