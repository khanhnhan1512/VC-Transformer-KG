"""
augment_kg_data.py - Data augmentation for small Knowledge Graph datasets

Techniques:
1. Inverse Relations: (h, r, t) -> (t, r_inverse, h)
   - "man play guitar" -> "guitar played_by man"
2. Self-loops removal: Remove (e, r, e) patterns
3. Frequency-based filtering: Keep only relations with enough samples

Usage:
    python knowledge/augment_kg_data.py --input_dir OPENKE_file --output_dir OPENKE_augmented
"""

import os
import argparse
from collections import Counter
from typing import Set, Tuple, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Augment KG data for small datasets")
    parser.add_argument("--input_dir", default="OPENKE_file", help="Input OPENKE directory")
    parser.add_argument("--output_dir", default="OPENKE_augmented", help="Output directory")
    parser.add_argument("--add_inverse", action="store_true", default=True,
                        help="Add inverse relations (h,r,t) -> (t,r_inv,h)")
    parser.add_argument("--min_rel_freq", type=int, default=10,
                        help="Minimum frequency for a relation to be kept")
    parser.add_argument("--min_ent_freq", type=int, default=3,
                        help="Minimum frequency for an entity to be kept")
    parser.add_argument("--remove_self_loops", action="store_true", default=True,
                        help="Remove triplets where head == tail")
    return parser.parse_args()


def load_id_mapping(filepath: str) -> Dict[str, int]:
    """Load entity2id or relation2id file."""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:] if lines[0].strip().isdigit() else lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_triplets(filepath: str) -> Set[Tuple[int, int, int]]:
    """Load triplets from train2id.txt format."""
    triplets = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) == 3:
                h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                triplets.add((h, t, r))
    return triplets


def save_triplets(triplets: Set[Tuple[int, int, int]], filepath: str):
    """Save triplets in OpenKE format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{len(triplets)}\n")
        for h, t, r in triplets:
            f.write(f"{h} {t} {r}\n")


def save_id_mapping(mapping: Dict[str, int], filepath: str):
    """Save entity2id or relation2id file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{len(mapping)}\n")
        for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{idx}\n")


def add_inverse_relations(
    triplets: Set[Tuple[int, int, int]],
    rel2id: Dict[str, int],
    id2rel: Dict[int, str]
) -> Tuple[Set[Tuple[int, int, int]], Dict[str, int]]:
    """
    Add inverse relations: (h, r, t) -> (t, r_inverse, h)
    
    This doubles the training data and helps model learn bidirectional patterns.
    """
    new_triplets = set(triplets)
    new_rel2id = dict(rel2id)
    
    # Create inverse relation IDs
    max_rel_id = max(rel2id.values())
    inverse_map = {}  # original_rel_id -> inverse_rel_id
    
    for rel_name, rel_id in rel2id.items():
        inv_name = f"{rel_name}_inv"
        inv_id = max_rel_id + 1 + rel_id
        new_rel2id[inv_name] = inv_id
        inverse_map[rel_id] = inv_id
    
    # Add inverse triplets
    for h, t, r in triplets:
        if h != t:  # Skip self-loops
            inv_r = inverse_map[r]
            new_triplets.add((t, h, inv_r))
    
    return new_triplets, new_rel2id


def filter_by_frequency(
    triplets: Set[Tuple[int, int, int]],
    min_rel_freq: int,
    min_ent_freq: int
) -> Set[Tuple[int, int, int]]:
    """Filter triplets by entity and relation frequency."""
    
    # Count frequencies
    rel_counts = Counter(r for _, _, r in triplets)
    ent_counts = Counter()
    for h, t, _ in triplets:
        ent_counts[h] += 1
        ent_counts[t] += 1
    
    # Filter
    valid_rels = {r for r, c in rel_counts.items() if c >= min_rel_freq}
    valid_ents = {e for e, c in ent_counts.items() if c >= min_ent_freq}
    
    filtered = {
        (h, t, r) for h, t, r in triplets
        if r in valid_rels and h in valid_ents and t in valid_ents
    }
    
    print(f"  Relations kept: {len(valid_rels)} (threshold: {min_rel_freq})")
    print(f"  Entities kept: {len(valid_ents)} (threshold: {min_ent_freq})")
    print(f"  Triplets kept: {len(filtered)} / {len(triplets)}")
    
    return filtered


def remove_self_loops(triplets: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
    """Remove triplets where head == tail."""
    filtered = {(h, t, r) for h, t, r in triplets if h != t}
    removed = len(triplets) - len(filtered)
    print(f"  Self-loops removed: {removed}")
    return filtered


def remap_ids(
    triplets: Set[Tuple[int, int, int]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int]
) -> Tuple[Set[Tuple[int, int, int]], Dict[str, int], Dict[str, int]]:
    """Remap entity and relation IDs to be contiguous after filtering."""
    
    # Find used IDs
    used_ents = set()
    used_rels = set()
    for h, t, r in triplets:
        used_ents.add(h)
        used_ents.add(t)
        used_rels.add(r)
    
    # Create reverse mappings
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    
    # Create new contiguous mappings
    new_ent2id = {}
    new_rel2id = {}
    ent_remap = {}  # old_id -> new_id
    rel_remap = {}
    
    for new_id, old_id in enumerate(sorted(used_ents)):
        if old_id in id2ent:
            new_ent2id[id2ent[old_id]] = new_id
            ent_remap[old_id] = new_id
    
    for new_id, old_id in enumerate(sorted(used_rels)):
        if old_id in id2rel:
            new_rel2id[id2rel[old_id]] = new_id
            rel_remap[old_id] = new_id
    
    # Remap triplets
    new_triplets = set()
    for h, t, r in triplets:
        if h in ent_remap and t in ent_remap and r in rel_remap:
            new_triplets.add((ent_remap[h], ent_remap[t], rel_remap[r]))
    
    return new_triplets, new_ent2id, new_rel2id


def split_data(
    triplets: Set[Tuple[int, int, int]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1
) -> Tuple[Set, Set, Set]:
    """Split triplets into train/valid/test sets."""
    triplet_list = list(triplets)
    import random
    random.shuffle(triplet_list)
    
    n = len(triplet_list)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    
    train = set(triplet_list[:n_train])
    valid = set(triplet_list[n_train:n_train + n_valid])
    test = set(triplet_list[n_train + n_valid:])
    
    return train, valid, test


def main():
    args = parse_args()
    
    print(f"[config] Input: {args.input_dir}")
    print(f"[config] Output: {args.output_dir}")
    print(f"[config] Add inverse: {args.add_inverse}")
    print(f"[config] Min relation freq: {args.min_rel_freq}")
    print(f"[config] Min entity freq: {args.min_ent_freq}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("[1] Loading data...")
    ent2id = load_id_mapping(os.path.join(args.input_dir, "entity2id.txt"))
    rel2id = load_id_mapping(os.path.join(args.input_dir, "relation2id.txt"))
    id2rel = {v: k for k, v in rel2id.items()}
    
    train = load_triplets(os.path.join(args.input_dir, "train2id.txt"))
    valid = load_triplets(os.path.join(args.input_dir, "valid2id.txt"))
    test = load_triplets(os.path.join(args.input_dir, "test2id.txt"))
    
    all_triplets = train | valid | test
    print(f"  Original: {len(train)} train, {len(valid)} valid, {len(test)} test")
    print(f"  Total unique: {len(all_triplets)}")
    print(f"  Entities: {len(ent2id)}, Relations: {len(rel2id)}")
    print()
    
    # Step 1: Remove self-loops
    if args.remove_self_loops:
        print("[2] Removing self-loops...")
        all_triplets = remove_self_loops(all_triplets)
        print()
    
    # Step 2: Add inverse relations
    if args.add_inverse:
        print("[3] Adding inverse relations...")
        original_count = len(all_triplets)
        all_triplets, rel2id = add_inverse_relations(all_triplets, rel2id, id2rel)
        print(f"  Triplets: {original_count} -> {len(all_triplets)} (+{len(all_triplets) - original_count})")
        print(f"  Relations: {len(id2rel)} -> {len(rel2id)}")
        print()
    
    # Step 3: Filter by frequency
    if args.min_rel_freq > 0 or args.min_ent_freq > 0:
        print("[4] Filtering by frequency...")
        all_triplets = filter_by_frequency(all_triplets, args.min_rel_freq, args.min_ent_freq)
        print()
    
    # Step 4: Remap IDs
    print("[5] Remapping IDs...")
    all_triplets, ent2id, rel2id = remap_ids(all_triplets, ent2id, rel2id)
    print(f"  Final entities: {len(ent2id)}")
    print(f"  Final relations: {len(rel2id)}")
    print(f"  Final triplets: {len(all_triplets)}")
    print()
    
    # Step 5: Split data
    print("[6] Splitting data...")
    train, valid, test = split_data(all_triplets, train_ratio=0.8, valid_ratio=0.1)
    print(f"  Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    print()
    
    # Save
    print("[7] Saving...")
    save_id_mapping(ent2id, os.path.join(args.output_dir, "entity2id.txt"))
    save_id_mapping(rel2id, os.path.join(args.output_dir, "relation2id.txt"))
    save_triplets(train, os.path.join(args.output_dir, "train2id.txt"))
    save_triplets(valid, os.path.join(args.output_dir, "valid2id.txt"))
    save_triplets(test, os.path.join(args.output_dir, "test2id.txt"))
    
    print(f"\nDone! Output saved to {args.output_dir}")
    
    # Print relation distribution
    print("\n[info] Top 15 relations in augmented data:")
    rel_counts = Counter(r for _, _, r in train)
    id2rel_new = {v: k for k, v in rel2id.items()}
    for rel_id, count in rel_counts.most_common(15):
        rel_name = id2rel_new.get(rel_id, f"UNK-{rel_id}")
        pct = count / len(train) * 100
        print(f"  {rel_name}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
