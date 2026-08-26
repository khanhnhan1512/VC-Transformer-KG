# DiBiDec

**Video Captioning via Direct Bidirectional Decoding with GOP-Based Multimodal Features**

DiBiDec (Direct Bidirectional Decoder) integrates pre-trained multimodal features (appearance, semantic, and motion) extracted at the Group-of-Pictures (GOP) level directly into a bidirectional Transformer decoder, without an intermediate fusion encoder.

This codebase is developed upon [BTKG](https://github.com/nickchen121/BTKG/tree/main).

## Project Structure

```
root/
├── config.py                       # Configuration (datasets, model, training)
├── train.py                        # Training and evaluation entry point
├── utils.py                        # Training loop, evaluation, metrics
├── requirements.txt                # Python dependencies
├── models/
│   ├── abd_transformer.py          # DiBiDec model
│   └── label_smoothing.py          # Label smoothing loss
├── loader/
│   ├── data_loader_fusion.py       # Base dataset, vocabulary, corpus classes
│   ├── MSVD.py                     # MSVD dataset loader
│   ├── MSRVTT.py                   # MSR-VTT dataset loader
│   ├── VATEX.py                    # VATEX dataset loader
│   └── transform.py                # Caption and feature transformations
├── test_video_ids/                 # Test-set video IDs used for evaluation
│   ├── msvd_test_video_ids.txt     # 670 video IDs
│   ├── msrvtt_test_video_ids.txt   # 2,990 video IDs
│   └── vatex_test_video_ids.txt    # 5,808 video IDs (accessible subset)
└── data/                           # Datasets (downloaded separately, see below)
    ├── MSVD/
    ├── MSRVTT/
    └── VATEX/
```

## Installation

**Requirements:** Python 3.10+, CUDA 11.8

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The `data/` folder is not included in this repository due to its large size. Download the pre-extracted features and metadata from the links below:

| Dataset | Download |
|---------|----------|
| MSVD | [Link](https://www.kaggle.com/datasets/vmphat/bidect-msvd-dataset) |
| MSR-VTT | [Link](https://www.kaggle.com/datasets/vmphat/bidect-msrvtt-dataset) |
| VATEX | [Link](https://www.kaggle.com/datasets/vmphat/bidect-vatex-dataset) |

After downloading, place the files so that the directory structure matches:

```
data/
├── MSVD/
│   ├── features/       # .hdf5 feature files
│   └── metadata/       # .csv caption files
├── MSRVTT/
│   ├── features/       # .hdf5 feature files
│   └── metadata/       # .json caption files
└── VATEX/
    ├── features/       # .hdf5 feature files
    └── metadata/       # .json caption files
```

## Test Video IDs

The `test_video_ids/` folder lists the exact video IDs used for testing on each benchmark, one ID per line:

| Dataset | File | # Videos |
|---------|------|----------|
| MSVD | `test_video_ids/msvd_test_video_ids.txt` | 670 |
| MSR-VTT | `test_video_ids/msrvtt_test_video_ids.txt` | 2,990 |
| VATEX | `test_video_ids/vatex_test_video_ids.txt` | 5,808 |

For VATEX, some videos from the original public test split are no longer publicly available. Evaluation is therefore performed on the accessible subset of 5,808 videos (out of the official 6,000-video test set). These IDs are released so that the reported VATEX results can be reproduced and compared on the same subset.

## Usage

### Configure

Edit `config.py` to select the dataset and features:

```python
class TrainConfig:
    corpus = "MSRVTT"  # Options: "MSVD", "MSRVTT", "VATEX"
```

```python
class FeatureConfig:
    model = "newBlip2ClsKF+newImgCapBlip2KF+newMViTv2"
```

### Train

```bash
python train.py
```

Checkpoints are saved to `./checkpoints/` and logs to `./logs/`.

## Acknowledgements

This project is built upon [BTKG](https://github.com/nickchen121/BTKG/tree/main). We thank the original authors for their work.
