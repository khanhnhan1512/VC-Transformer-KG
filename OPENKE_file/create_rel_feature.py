"""
Tạo rel_feature HDF5 từ Pre-trained Embeddings (GloVe)
Input: 
    - ent_msvd.txt, rel_msvd.txt (entity/relation vocabulary)
    - Video-caption mapping hoặc object detection results
    - GloVe embeddings
Output:
    - MSVD_rel_train.hdf5, MSVD_rel_val.hdf5, MSVD_rel_test.hdf5
"""

import os
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

# ============== CONFIGURATION ==============
GLOVE_PATH = '/kaggle/input/glove-6b/glove.6B/glove.6B.300d.txt'
OPENKE_DIR = '/kaggle/working/OPENKE_file'
METADATA_DIR = '/kaggle/input/btkg-msvd-dataset/MSVD/metadata'
OUTPUT_DIR = '/kaggle/working/features'

EMBEDDING_DIM = 300
NUM_FRAMES = 28  # Số frames per video (từ config)

# ============== LOAD GLOVE ==============
def load_glove(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings"""
    print(f"Loading GloVe from {glove_path}...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings

# ============== LOAD ENTITY/RELATION VOCABULARY ==============
def load_vocabulary(ent_path: str, rel_path: str) -> Tuple[List[str], List[str]]:
    """Load entity và relation vocabulary"""
    entities = []
    with open(ent_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                main = line.split('|')[0].strip()
                entities.append(main)
    
    relations = []
    with open(rel_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                main = line.split('|')[0].strip()
                relations.append(main)
    
    return entities, relations

# ============== CREATE ENTITY EMBEDDINGS ==============
def create_entity_embeddings(
    entities: List[str], 
    glove: Dict[str, np.ndarray]
) -> np.ndarray:
    """Tạo embedding matrix cho entities"""
    embeddings = np.zeros((len(entities), EMBEDDING_DIM), dtype=np.float32)
    
    for idx, entity in enumerate(entities):
        words = entity.lower().split()
        vectors = [glove[w] for w in words if w in glove]
        if vectors:
            embeddings[idx] = np.mean(vectors, axis=0)
    
    return embeddings

def create_relation_embeddings(
    relations: List[str],
    glove: Dict[str, np.ndarray]
) -> np.ndarray:
    """Tạo embedding matrix cho relations"""
    embeddings = np.zeros((len(relations), EMBEDDING_DIM), dtype=np.float32)
    
    for idx, relation in enumerate(relations):
        words = relation.lower().split()
        vectors = [glove[w] for w in words if w in glove]
        if vectors:
            embeddings[idx] = np.mean(vectors, axis=0)
    
    return embeddings

# ============== LOAD VIDEO-TRIPLE MAPPING ==============
def load_video_triples(total_word_path: str, total_id_path: str) -> Dict[str, List[Tuple]]:
    """
    Map video IDs to their knowledge triples
    Hiện tại chưa có mapping video->triple, nên dùng global average
    """
    # Đọc tất cả valid triples
    triples = []
    with open(total_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                s, o, r = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((s, o, r))
    
    return triples

# ============== CREATE REL_FEATURE ==============
def create_rel_feature_for_video(
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    triples: List[Tuple],
    num_frames: int = 28
) -> np.ndarray:
    """
    Tạo rel_feature cho một video
    
    Phương pháp: 
    - Aggregate embeddings từ tất cả triples liên quan
    - Repeat cho mỗi frame
    
    Returns: (num_frames, 300)
    """
    if not triples:
        return np.zeros((num_frames, EMBEDDING_DIM), dtype=np.float32)
    
    # Aggregate triple embeddings
    triple_features = []
    for s, o, r in triples:
        if s < len(entity_emb) and o < len(entity_emb) and r < len(relation_emb):
            # Combine: subject + object + relation
            feat = (entity_emb[s] + entity_emb[o] + relation_emb[r]) / 3.0
            triple_features.append(feat)
    
    if triple_features:
        avg_feature = np.mean(triple_features, axis=0)
    else:
        avg_feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Repeat for each frame
    rel_feature = np.tile(avg_feature, (num_frames, 1))
    
    return rel_feature.astype(np.float32)

# ============== MAIN: CREATE HDF5 FILES ==============
def create_rel_feature_hdf5(
    split: str,  # 'train', 'val', 'test'
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    triples: List[Tuple],
    metadata_dir: str,
    output_dir: str
):
    """Tạo file HDF5 cho một split"""
    
    # Đọc danh sách video IDs từ metadata
    csv_path = os.path.join(metadata_dir, f'{split}.csv')
    df = pd.read_csv(csv_path)
    video_ids = df['VideoID'].unique().tolist()
    
    print(f"\nCreating rel_feature for {split} split ({len(video_ids)} videos)...")
    
    # Tạo output file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'MSVD_rel_{split}.hdf5')
    
    with h5py.File(output_path, 'w') as hf:
        for vid in video_ids:
            # Tạo feature cho video
            # Hiện tại dùng global triples (chưa có per-video mapping)
            rel_feat = create_rel_feature_for_video(
                entity_emb, relation_emb, triples, NUM_FRAMES
            )
            
            # Lưu vào HDF5
            hf.create_dataset(str(vid), data=rel_feat)
    
    print(f"Saved: {output_path}")
    print(f"  - Videos: {len(video_ids)}")
    print(f"  - Feature shape per video: ({NUM_FRAMES}, {EMBEDDING_DIM})")

# ============== MAIN ==============
def main():
    print("="*60)
    print("Creating rel_feature HDF5 files")
    print("="*60)
    
    # Load GloVe
    glove = load_glove(GLOVE_PATH)
    
    # Load vocabulary
    ent_path = os.path.join(OPENKE_DIR, 'ent_msvd.txt')
    rel_path = os.path.join(OPENKE_DIR, 'rel_msvd.txt')
    
    # Fallback to original files if MSVD-specific not found
    if not os.path.exists(ent_path):
        ent_path = os.path.join(OPENKE_DIR, 'ent.txt')
    if not os.path.exists(rel_path):
        rel_path = os.path.join(OPENKE_DIR, 'rel.txt')
    
    entities, relations = load_vocabulary(ent_path, rel_path)
    print(f"Loaded {len(entities)} entities, {len(relations)} relations")
    
    # Create embeddings
    entity_emb = create_entity_embeddings(entities, glove)
    relation_emb = create_relation_embeddings(relations, glove)
    print(f"Entity embeddings: {entity_emb.shape}")
    print(f"Relation embeddings: {relation_emb.shape}")
    
    # Load triples
    total_id_path = os.path.join(OPENKE_DIR, 'total_id.txt')
    total_word_path = os.path.join(OPENKE_DIR, 'total_word.txt')
    triples = load_video_triples(total_word_path, total_id_path)
    print(f"Loaded {len(triples)} triples")
    
    # Create HDF5 for each split
    for split in ['train', 'val', 'test']:
        create_rel_feature_hdf5(
            split, entity_emb, relation_emb, triples,
            METADATA_DIR, OUTPUT_DIR
        )
    
    print("\n" + "="*60)
    print("DONE! Created:")
    print(f"  - {OUTPUT_DIR}/MSVD_rel_train.hdf5")
    print(f"  - {OUTPUT_DIR}/MSVD_rel_val.hdf5")
    print(f"  - {OUTPUT_DIR}/MSVD_rel_test.hdf5")
    print("="*60)

if __name__ == '__main__':
    main()