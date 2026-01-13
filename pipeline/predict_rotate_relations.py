"""
Predict relations for object pairs detected on keyframes using a trained RotatE model.
Designed to run easily on Kaggle (CLI). Example Kaggle cell:

!pip install openke-py
!python predict_rotate_relations.py \
    --hdf5_path /kaggle/working/pipeline/result/MSVD_DinoObjects.hdf5 \
    --entity2id /kaggle/working/OPENKE_file/entity2id.txt \
    --relation2id /kaggle/working/OPENKE_file/relation2id.txt \
    --checkpoint /kaggle/working/openke_rotate_ckpt/rotate.ckpt \
    --output_csv /kaggle/working/dino_relation_preds.csv
"""
import argparse
import json
import os
from typing import Dict, Iterable, Iterator, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from openke.module.model import RotatE


def load_id_map(path: str) -> Dict[str, int]:
    """Load a name->id map from OpenKE-style files.

    Supports both formats:
    - With header count on the first line
    - Without header
    Lines are typically: <name>\t<id>
    """
    mp: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if lines and lines[0].isdigit():
        lines = lines[1:]

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        name, idx_str = parts[0], parts[1]
        try:
            mp[name] = int(idx_str)
        except ValueError:
            continue
    return mp


def build_inverse_map(mp: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in mp.items()}


def get_frame_lengths(hdf5_path: str) -> Dict[str, int]:
    """Return number of frames per video to preallocate prediction slots."""
    frame_lengths: Dict[str, int] = {}
    with h5py.File(hdf5_path, "r") as f:
        for vid in f.keys():
            frame_lengths[vid] = len(f[vid])
    return frame_lengths


def write_hdf5(path: str, rel_store: Dict[str, List[List[Dict[str, object]]]]) -> None:
    """Write predictions into HDF5: each video is a dataset of JSON-encoded frames."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        for vid, frames in rel_store.items():
            json_frames = [json.dumps(frame_preds) for frame_preds in frames]
            f.create_dataset(vid, data=np.array(json_frames, dtype=str_dtype), compression="gzip")


def iter_object_pairs(
    hdf5_path: str,
    ent2id: Dict[str, int],
    score_threshold: float,
    permutations: bool = True,
) -> Iterator[Tuple[str, int, str, str]]:
    """Yield (video_id, frame_idx, head_label, tail_label) for valid pairs."""
    with h5py.File(hdf5_path, "r") as f:
        for vid in f.keys():
            frames = f[vid][()]
            for frame_idx, frame_json in enumerate(frames):
                if isinstance(frame_json, bytes):
                    frame_json = frame_json.decode("utf-8")
                objs = [o for o in json.loads(frame_json) if o.get("score", 0) >= score_threshold]
                # Keep only objects that exist in KG
                objs = [o for o in objs if o.get("label") in ent2id]
                if len(objs) < 2:
                    continue
                indices = range(len(objs))
                pairs = (
                    ((i, j) for i in indices for j in indices if i != j)
                    if permutations
                    else ((i, j) for i in indices for j in indices if i < j)
                )
                for i, j in pairs:
                    yield vid, frame_idx, objs[i]["label"], objs[j]["label"]


def score_relations(
    model: RotatE,
    device: torch.device,
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    head_label: str,
    tail_label: str,
    topk: int,
) -> List[Tuple[str, float]]:
    rel_tot = len(rel2id)
    h_id = ent2id[head_label]
    t_id = ent2id[tail_label]

    heads = torch.full((rel_tot,), h_id, dtype=torch.long, device=device)
    tails = torch.full((rel_tot,), t_id, dtype=torch.long, device=device)
    rels = torch.arange(rel_tot, device=device, dtype=torch.long)
    
    # OpenKE expects dict format with batch_h, batch_t, batch_r
    data = {
        'batch_h': heads,
        'batch_t': tails,
        'batch_r': rels,
        'mode': 'normal'
    }

    with torch.no_grad():
        # OpenKE RotatE.forward returns: margin - distance, so HIGHER is BETTER.
        scores = model.forward(data)

    k = min(topk, rel_tot)
    top_scores, top_indices = torch.topk(scores, k=k)

    inv_rel = build_inverse_map(rel2id)
    return [(inv_rel[int(rel_id)], float(score)) for score, rel_id in zip(top_scores, top_indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RotatE relation prediction for DINO objects")
    parser.add_argument(
        "--hdf5_path",
        default="pipeline/result/MSVD_DinoObjects.hdf5",
        help="Path to HDF5 with detected objects per keyframe.",
    )
    parser.add_argument(
        "--entity2id",
        default="OPENKE_file/entity2id.txt",
        help="Path to entity2id.txt used for training RotatE.",
    )
    parser.add_argument(
        "--relation2id",
        default="OPENKE_file/relation2id.txt",
        help="Path to relation2id.txt used for training RotatE.",
    )
    parser.add_argument(
        "--checkpoint",
        default="openke_rotate_ckpt/rotate.ckpt",
        help="Path to trained RotatE checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--dim", type=int, default=256, help="Embedding dim used when training (must match checkpoint)."
    )
    parser.add_argument(
        "--gamma", type=float, default=6.0, help="Gamma used when training (must match checkpoint)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=2.0, help="Epsilon used when training (must match checkpoint)."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.35,
        help="Discard detected objects with score below this value.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many top relations to keep per object pair.",
    )
    parser.add_argument(
        "--permutations",
        action="store_true",
        help="Use ordered pairs (head, tail) permutations; default is unordered combos.",
    )
    parser.add_argument(
        "--output_csv",
        default="pipeline/result/dino_relation_preds.csv",
        help="Where to save the predictions CSV.",
    )
    parser.add_argument(
        "--output_hdf5",
        default="pipeline/result/dino_relation_preds.hdf5",
        help="Where to save predictions in HDF5 (per video, per frame JSON list).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection (auto picks cuda if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ent2id = load_id_map(args.entity2id)
    rel2id = load_id_map(args.relation2id)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[info] device={device} | entities={len(ent2id)} | relations={len(rel2id)}")

    model = RotatE(ent_tot=len(ent2id), rel_tot=len(rel2id), dim=args.dim, margin=args.gamma, epsilon=args.epsilon)
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    rows: List[Dict[str, object]] = []
    frame_lengths = get_frame_lengths(args.hdf5_path)
    rel_store: Dict[str, List[List[Dict[str, object]]]] = {
        vid: [[] for _ in range(frame_lengths[vid])] for vid in frame_lengths
    }
    total_pairs = 0
    for vid, frame_idx, head_lbl, tail_lbl in iter_object_pairs(
        args.hdf5_path, ent2id, args.score_threshold, permutations=args.permutations
    ):
        total_pairs += 1
        top_rels = score_relations(
            model,
            device,
            ent2id,
            rel2id,
            head_lbl,
            tail_lbl,
            args.topk,
        )
        for rel, score in top_rels:
            if vid in rel_store and frame_idx < len(rel_store[vid]):
                rel_store[vid][frame_idx].append(
                    {"head": head_lbl, "tail": tail_lbl, "relation": rel, "score": score}
                )
            rows.append(
                {
                    "video_id": vid,
                    "frame_idx": frame_idx,
                    "head": head_lbl,
                    "tail": tail_lbl,
                    "relation": rel,
                    "score": score,
                }
            )
        if total_pairs % 5000 == 0:
            print(f"[progress] processed {total_pairs} pairs")

    if not rows:
        print("[warn] no predictions were produced; check thresholds and ID maps")
        return

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"[done] pairs={total_pairs} | saved CSV -> {args.output_csv}")

    if args.output_hdf5:
        write_hdf5(args.output_hdf5, rel_store)
        print(f"[done] saved HDF5 -> {args.output_hdf5}")


if __name__ == "__main__":
    main()
