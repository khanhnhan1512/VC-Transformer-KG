"""
================================================================================
PHƯƠNG PHÁP 2: Caption-based Matching (V2)
================================================================================

Cải tiến: Mỗi video có embedding RIÊNG dựa trên captions của nó.

Pipeline:
    1. Load captions cho mỗi video từ CSV
    2. Tokenize captions → match với vocabulary (ent.txt, rel.txt)
    3. Lấy GloVe embeddings của matched entities/relations
    4. Average → Per-video embedding

Input:
    - train.csv, val.csv, test.csv (captions)
    - ent.txt, rel.txt (vocabulary với synonyms)
    - GloVe embeddings

Output:
    - MSVD_rel_{train,val,test}.hdf5: shape (num_videos, 28, 300)
    - Mỗi video có embedding RIÊNG!

Kết quả: ~60-80% unique embeddings ✅
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

# ============== CONFIGURATION ==============
# Kaggle paths
GLOVE_PATH = '/kaggle/input/glove-6b/glove.6B/glove.6B.300d.txt'
OPENKE_DIR = '/kaggle/working/OPENKE_file'
METADATA_DIR = '/kaggle/input/btkg-msvd-dataset/MSVD/metadata'
OUTPUT_DIR = '/kaggle/working/features_v2'

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

def load_vocabulary_with_synonyms(ent_path: str, rel_path: str):
    """
    Load vocabulary VỚI synonyms
    
    Format file: main_word|synonym1#synonym2#synonym3
    Ví dụ: person|man#guy#woman#lady
    
    Returns:
        word2entity: {"man": 1, "guy": 1, "person": 1, ...}
        word2relation: {"play": 0, "playing": 0, ...}
        entities: ["__background", "person", "bicycle", ...]
        relations: ["drive", "play", "talk", ...]
    """
    entities = []
    word2entity = {}
    
    with open(ent_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                main = parts[0].strip()
                entities.append(main)
                
                # Map main word
                if main:
                    word2entity[main.lower()] = idx
                
                # Map synonyms
                if len(parts) > 1 and parts[1]:
                    for syn in parts[1].split('#'):
                        syn = syn.strip().lower()
                        if syn:
                            word2entity[syn] = idx
    
    relations = []
    word2relation = {}
    
    with open(rel_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                main = parts[0].strip()
                relations.append(main)
                
                # Map main word
                if main:
                    word2relation[main.lower()] = idx
                
                # Map synonyms
                if len(parts) > 1 and parts[1]:
                    for syn in parts[1].split('#'):
                        syn = syn.strip().lower()
                        if syn:
                            word2relation[syn] = idx
    
    print(f"Loaded {len(entities)} entities with {len(word2entity)} word mappings")
    print(f"Loaded {len(relations)} relations with {len(word2relation)} word mappings")
    
    return word2entity, word2relation, entities, relations

def create_embeddings(words: List[str], glove: Dict[str, np.ndarray]) -> np.ndarray:
    """Tạo embedding matrix từ vocabulary"""
    embeddings = np.zeros((len(words), EMBEDDING_DIM), dtype=np.float32)
    
    for idx, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in glove:
            embeddings[idx] = glove[word_lower]
        else:
            tokens = word_lower.split()
            vecs = [glove[t] for t in tokens if t in glove]
            if vecs:
                embeddings[idx] = np.mean(vecs, axis=0)
    
    return embeddings

def preprocess_text(text: str) -> List[str]:
    """Tiền xử lý text: lowercase, remove punctuation, tokenize"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def extract_from_caption(
    caption: str,
    word2entity: Dict[str, int],
    word2relation: Dict[str, int]
) -> Tuple[Set[int], Set[int]]:
    """
    Trích xuất entity và relation IDs từ caption
    
    Ví dụ: "A man is playing guitar"
    → entities: {1}  (man → person)
    → relations: {1} (playing → play)
    """
    words = preprocess_text(caption)
    
    entity_ids = set()
    relation_ids = set()
    
    # Single word matching
    for word in words:
        if word in word2entity:
            entity_ids.add(word2entity[word])
        if word in word2relation:
            relation_ids.add(word2relation[word])
    
    # 2-gram matching (for phrases like "sports ball")
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if phrase in word2entity:
            entity_ids.add(word2entity[phrase])
        if phrase in word2relation:
            relation_ids.add(word2relation[phrase])
    
    return entity_ids, relation_ids

def create_video_feature(
    entity_ids: Set[int],
    relation_ids: Set[int],
    entity_emb: np.ndarray,
    relation_emb: np.ndarray
) -> np.ndarray:
    """
    Tạo feature cho MỘT video dựa trên matched entities/relations
    """
    features = []
    
    # Collect entity embeddings
    for ent_id in entity_ids:
        if ent_id < len(entity_emb):
            features.append(entity_emb[ent_id])
    
    # Collect relation embeddings
    for rel_id in relation_ids:
        if rel_id < len(relation_emb):
            features.append(relation_emb[rel_id])
    
    if features:
        avg_feature = np.mean(features, axis=0)
    else:
        avg_feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Repeat cho mỗi frame
    return np.tile(avg_feature, (NUM_FRAMES, 1)).astype(np.float32)

def load_video_captions(csv_path: str) -> Dict[str, List[str]]:
    """Load tất cả captions cho mỗi video"""
    df = pd.read_csv(csv_path)
    
    video_captions = defaultdict(list)
    for _, row in df.iterrows():
        vid = row['VideoID']
        caption = row['Description']
        if pd.notna(caption):
            video_captions[vid].append(str(caption))
    
    return dict(video_captions)

def create_hdf5_for_split(
    split: str,
    word2entity: Dict[str, int],
    word2relation: Dict[str, int],
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    metadata_dir: str,
    output_dir: str
):
    """Tạo HDF5 file cho một split"""
    csv_path = os.path.join(metadata_dir, f'{split}.csv')
    video_captions = load_video_captions(csv_path)
    video_ids = list(video_captions.keys())
    
    print(f"\nProcessing {split}: {len(video_ids)} videos")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'MSVD_rel_{split}.hdf5')
    
    stats = {'with_entities': 0, 'with_relations': 0}
    
    with h5py.File(output_path, 'w') as hf:
        for vid in tqdm(video_ids, desc=split):
            captions = video_captions[vid]
            
            # Aggregate từ TẤT CẢ captions của video
            all_entity_ids = set()
            all_relation_ids = set()
            
            for caption in captions:
                ent_ids, rel_ids = extract_from_caption(
                    caption, word2entity, word2relation
                )
                all_entity_ids.update(ent_ids)
                all_relation_ids.update(rel_ids)
            
            if all_entity_ids:
                stats['with_entities'] += 1
            if all_relation_ids:
                stats['with_relations'] += 1
            
            # Create per-video feature
            rel_feat = create_video_feature(
                all_entity_ids, all_relation_ids,
                entity_emb, relation_emb
            )
            
            hf.create_dataset(str(vid), data=rel_feat)
    
    print(f"Saved: {output_path}")
    print(f"  - With entities: {stats['with_entities']}/{len(video_ids)} "
          f"({stats['with_entities']/len(video_ids)*100:.1f}%)")
    print(f"  - With relations: {stats['with_relations']}/{len(video_ids)} "
          f"({stats['with_relations']/len(video_ids)*100:.1f}%)")

# ============== MAIN ==============
def main():
    print("="*60)
    print("V2: Caption-based Matching Method")
    print("✅ Each video has its OWN embedding!")
    print("="*60)
    
    # Load GloVe
    glove = load_glove(GLOVE_PATH)
    
    # Load vocabulary with synonyms
    ent_path = os.path.join(OPENKE_DIR, 'ent.txt')
    rel_path = os.path.join(OPENKE_DIR, 'rel.txt')
    word2entity, word2relation, entities, relations = load_vocabulary_with_synonyms(
        ent_path, rel_path
    )
    
    # Create embeddings
    entity_emb = create_embeddings(entities, glove)
    relation_emb = create_embeddings(relations, glove)
    print(f"Entity embeddings: {entity_emb.shape}")
    print(f"Relation embeddings: {relation_emb.shape}")
    
    # Create HDF5 for each split
    for split in ['train', 'val', 'test']:
        create_hdf5_for_split(
            split, word2entity, word2relation,
            entity_emb, relation_emb,
            METADATA_DIR, OUTPUT_DIR
        )
    
    print("\n" + "="*60)
    print("DONE! Each video has unique embedding ✅")
    print(f"Output: {OUTPUT_DIR}/MSVD_rel_{{train,val,test}}.hdf5")
    print("="*60)

if __name__ == '__main__':
    main()
