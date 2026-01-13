"""
Simple script to test RotatE model with a few head-tail pairs.
Designed for quick validation of the trained RotatE model.
"""
import argparse
import torch
from openke.module.model import RotatE
from typing import Dict, List, Tuple


def load_id_map(path: str) -> Dict[str, int]:
    """Load a name->id map from OpenKE-style files.

    Supports both formats:
    - With header count on the first line (common in OpenKE benchmarks)
    - Without header
    Lines are typically: <name>\t<id>
    """
    mp: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Drop an optional first-line count
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
    parser = argparse.ArgumentParser(description="Simple RotatE relation prediction for testing")
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
        "--topk",
        type=int,
        default=3,
        help="How many top relations to keep per object pair.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection (auto picks cuda if available).",
    )
    parser.add_argument(
        "--head",
        type=str,
        required=True,
        help="Head entity label for testing.",
    )
    parser.add_argument(
        "--tail",
        type=str,
        required=True,
        help="Tail entity label for testing.",
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

    # Use the provided head and tail labels
    head_label = args.head
    tail_label = args.tail

    if head_label not in ent2id or tail_label not in ent2id:
        missing = []
        if head_label not in ent2id:
            missing.append(f"head='{head_label}'")
        if tail_label not in ent2id:
            missing.append(f"tail='{tail_label}'")
        print(f"[error] Entity not found in entity2id: {', '.join(missing)}")
        # Quick suggestions
        q = (head_label if head_label not in ent2id else tail_label).lower()
        candidates = [e for e in ent2id.keys() if q in e.lower()]
        if candidates:
            print("[hint] Similar entities:")
            for e in candidates[:20]:
                print("  -", e)
        return
    
    top_rels = score_relations(
        model,
        device,
        ent2id,
        rel2id,
        head_label,
        tail_label,
        args.topk,
    )
    print(f"\nTop relations for ({head_label}, {tail_label}):")
    for rel, score in top_rels:
        print(f"  {rel}: {score:.4f}")


if __name__ == "__main__":
    main()