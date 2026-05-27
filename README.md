# [BiDecT]

An Encoder-free Bidirectional Decoder Transformer with GOP-based multimodal representations for video captioning.

This codebase is developed upon [BTKG](https://github.com/nickchen121/BTKG/tree/main).

## Project Structure

```
root/
├── config.py                       # Configuration (datasets, model, training)
├── train.py                        # Training and evaluation entry point
├── utils.py                        # Training loop, evaluation, metrics
├── requirements.txt                # Python dependencies
├── models/
│   ├── abd_transformer.py          # BiDecT model
│   └── label_smoothing.py          # Label smoothing loss
├── loader/
│   ├── data_loader_fusion.py       # Base dataset, vocabulary, corpus classes
│   ├── MSVD.py                     # MSVD dataset loader
│   ├── MSRVTT.py                   # MSR-VTT dataset loader
│   ├── VATEX.py                    # VATEX dataset loader
│   └── transform.py                # Caption and feature transformations
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
