"""
init_embeddings_glove.py - Initialize entity embeddings with GloVe pretrained vectors

This helps knowledge graph models converge faster and perform better on small datasets
by leveraging semantic information from pretrained word embeddings.

Usage:
    python knowledge/init_embeddings_glove.py \
        --glove_path data/glove.6B/glove.6B.100d.txt \
        --entity2id_path OPENKE_file/entity2id.txt \
        --output_path OPENKE_file/entity_init_emb.npy \
        --dim 100
"""

import argparse
import numpy as np
from typing import Dict
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize embeddings with GloVe")
    parser.add_argument("--glove_path", default="data/glove.6B/glove.6B.100d.txt",
                        help="Path to GloVe embeddings file")
    parser.add_argument("--entity2id_path", default="OPENKE_file/entity2id.txt",
                        help="Path to entity2id.txt")
    parser.add_argument("--output_path", default="OPENKE_file/entity_init_emb.npy",
                        help="Output path for numpy array")
    parser.add_argument("--dim", type=int, default=100,
                        help="GloVe dimension (50, 100, 200, or 300)")
    return parser.parse_args()


def load_glove(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings into a dictionary."""
    print(f"[info] Loading GloVe from {glove_path}...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
            
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} vectors...")
    
    print(f"[info] Loaded {len(embeddings)} GloVe vectors")
    return embeddings


def load_entity2id(filepath: str) -> Dict[str, int]:
    """Load entity2id mapping."""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Skip header if first line is just a number
        start = 1 if lines[0].strip().isdigit() else 0
        for line in lines[start:]:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def get_entity_embedding(entity: str, glove: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    """
    Get embedding for an entity.
    
    Strategy:
    1. Try exact match (lowercase)
    2. Try each word in multi-word entity
    3. Average available word vectors
    4. Fall back to random initialization
    """
    entity_lower = entity.lower().strip()
    
    # Exact match
    if entity_lower in glove:
        return glove[entity_lower]
    
    # Split multi-word entity and average
    words = entity_lower.replace('_', ' ').replace('-', ' ').split()
    vectors = []
    
    for word in words:
        if word in glove:
            vectors.append(glove[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    
    # Random initialization (normalized)
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def main():
    args = parse_args()
    
    # Check GloVe file exists
    if not os.path.exists(args.glove_path):
        # Try alternative paths
        alt_paths = [
            f"data/glove.6B/glove.6B.{args.dim}d.txt",
            f"glove.6B.{args.dim}d.txt",
            f"/kaggle/input/glove6b/glove.6B.{args.dim}d.txt",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                args.glove_path = alt
                break
        else:
            print(f"[error] GloVe file not found: {args.glove_path}")
            print("[hint] Download from: https://nlp.stanford.edu/data/glove.6B.zip")
            return
    
    # Load GloVe
    glove = load_glove(args.glove_path)
    glove_dim = len(next(iter(glove.values())))
    print(f"[info] GloVe dimension: {glove_dim}")
    
    # Load entities
    entity2id = load_entity2id(args.entity2id_path)
    num_entities = len(entity2id)
    print(f"[info] Number of entities: {num_entities}")
    
    # Create embedding matrix
    embeddings = np.zeros((num_entities, glove_dim), dtype=np.float32)
    
    found = 0
    partial = 0
    random_init = 0
    
    for entity, idx in entity2id.items():
        entity_lower = entity.lower()
        
        if entity_lower in glove:
            embeddings[idx] = glove[entity_lower]
            found += 1
        else:
            vec = get_entity_embedding(entity, glove, glove_dim)
            embeddings[idx] = vec
            
            # Check if it was partial match or random
            words = entity_lower.replace('_', ' ').replace('-', ' ').split()
            if any(w in glove for w in words):
                partial += 1
            else:
                random_init += 1
    
    print(f"\n[stats] Embedding initialization:")
    print(f"  Exact match: {found} ({found/num_entities*100:.1f}%)")
    print(f"  Partial match: {partial} ({partial/num_entities*100:.1f}%)")
    print(f"  Random init: {random_init} ({random_init/num_entities*100:.1f}%)")
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms
    
    # Save
    np.save(args.output_path, embeddings)
    print(f"\n[save] Embeddings saved to {args.output_path}")
    print(f"[save] Shape: {embeddings.shape}")
    
    # Print sample
    print("\n[sample] First 5 entities:")
    id2entity = {v: k for k, v in entity2id.items()}
    for i in range(min(5, num_entities)):
        entity = id2entity.get(i, f"UNK-{i}")
        vec_sample = embeddings[i][:5]
        print(f"  {i}: {entity} -> [{', '.join(f'{x:.3f}' for x in vec_sample)}...]")


if __name__ == "__main__":
    main()
