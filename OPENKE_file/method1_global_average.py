"""
================================================================================
PHƯƠNG PHÁP 1: Global Average (V1 Original)
================================================================================

Vấn đề: Tất cả video có CÙNG MỘT embedding vì dùng global average từ tất cả triples.

Pipeline:
    total_id.txt → Load all triples → Average all → Same embedding for all videos

Input:
    - total_id.txt: tất cả triples dạng (entity1_id, entity2_id, relation_id)
    - entity2id.txt, relation2id.txt
    - GloVe embeddings

Output:
    - MSVD_rel_{train,val,test}.hdf5: shape (num_videos, 28, 300)
    - Tất cả videos có CÙNG embedding!

Kết quả: ~0.1% unique embeddings ❌
"""

import os
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ============== CONFIGURATION ==============
# Kaggle paths
GLOVE_PATH = '/kaggle/input/glove-6b/glove.6B/glove.6B.300d.txt'
OPENKE_DIR = '/kaggle/working/OPENKE_file'
METADATA_DIR = '/kaggle/input/btkg-msvd-dataset/MSVD/metadata'
OUTPUT_DIR = '/kaggle/working/features_v1'

EMBEDDING_DIM = 300
NUM_FRAMES = 28

# ============== FUNCTIONS ==============
def load_glove(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings"""
    print(f"Loading GloVe from {glove_path}...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == EMBEDDING_DIM:
                embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings

def load_vocabulary(ent_path: str, rel_path: str) -> Tuple[List[str], List[str]]:
    """Load entity và relation vocabulary (chỉ lấy main word, không có synonyms)"""
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

def create_embeddings(words: List[str], glove: Dict[str, np.ndarray]) -> np.ndarray:
    """Tạo embedding matrix từ vocabulary"""
    embeddings = np.zeros((len(words), EMBEDDING_DIM), dtype=np.float32)
    
    for idx, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in glove:
            embeddings[idx] = glove[word_lower]
        else:
            # Multi-word: average
            tokens = word_lower.split()
            vecs = [glove[t] for t in tokens if t in glove]
            if vecs:
                embeddings[idx] = np.mean(vecs, axis=0)
    
    return embeddings

def load_all_triples(total_id_path: str) -> List[Tuple[int, int, int]]:
    """Load TẤT CẢ triples từ total_id.txt"""
    triples = []
    with open(total_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                e1, e2, r = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((e1, e2, r))
    return triples

def create_global_feature(
    triples: List[Tuple[int, int, int]],
    entity_emb: np.ndarray,
    relation_emb: np.ndarray
) -> np.ndarray:
    """
    Tạo GLOBAL feature từ TẤT CẢ triples
    ĐÂY LÀ VẤN ĐỀ: 1 feature cho tất cả videos!
    """
    if not triples:
        return np.zeros((NUM_FRAMES, EMBEDDING_DIM), dtype=np.float32)
    
    features = []
    for e1, e2, r in triples:
        if e1 < len(entity_emb) and e2 < len(entity_emb) and r < len(relation_emb):
            # Combine: (entity1 + entity2 + relation) / 3
            feat = (entity_emb[e1] + entity_emb[e2] + relation_emb[r]) / 3.0
            features.append(feat)
    
    if features:
        global_feature = np.mean(features, axis=0)
    else:
        global_feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Repeat cho mỗi frame
    return np.tile(global_feature, (NUM_FRAMES, 1)).astype(np.float32)

def create_hdf5_files(
    global_feature: np.ndarray,
    metadata_dir: str,
    output_dir: str
):
    """Tạo HDF5 files - TẤT CẢ videos dùng CÙNG feature!"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(metadata_dir, f'{split}.csv')
        df = pd.read_csv(csv_path)
        video_ids = df['VideoID'].unique().tolist()
        
        output_path = os.path.join(output_dir, f'MSVD_rel_{split}.hdf5')
        
        with h5py.File(output_path, 'w') as hf:
            for vid in video_ids:
                # CÙNG feature cho TẤT CẢ videos!
                hf.create_dataset(str(vid), data=global_feature)
        
        print(f"Saved: {output_path} ({len(video_ids)} videos)")

# ============== MAIN ==============
def main():
    print("="*60)
    print("V1: Global Average Method")
    print("⚠️  WARNING: All videos will have the SAME embedding!")
    print("="*60)
    
    # Load GloVe
    glove = load_glove(GLOVE_PATH)
    
    # Load vocabulary
    ent_path = os.path.join(OPENKE_DIR, 'ent.txt')
    rel_path = os.path.join(OPENKE_DIR, 'rel.txt')
    entities, relations = load_vocabulary(ent_path, rel_path)
    print(f"Entities: {len(entities)}, Relations: {len(relations)}")
    
    # Create embeddings
    entity_emb = create_embeddings(entities, glove)
    relation_emb = create_embeddings(relations, glove)
    
    # Load ALL triples
    total_id_path = os.path.join(OPENKE_DIR, 'total_id.txt')
    triples = load_all_triples(total_id_path)
    print(f"Total triples: {len(triples)}")
    
    # Create GLOBAL feature (same for all videos)
    global_feature = create_global_feature(triples, entity_emb, relation_emb)
    print(f"Global feature shape: {global_feature.shape}")
    
    # Create HDF5 files
    create_hdf5_files(global_feature, METADATA_DIR, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DONE! But all videos have the SAME embedding ❌")
    print("="*60)

if __name__ == '__main__':
    main()
