"""
================================================================================
PHƯƠNG PHÁP 3: TransE Training + Per-Video Matching (SOTA)
================================================================================

Phương pháp tốt nhất, gồm 2 bước:
1. Training TransE embeddings từ Knowledge Graph (thay GloVe)
2. Per-video matching với weighted aggregation

Ưu điểm so với V2:
- TransE embeddings học được relational structure: h + r ≈ t
- Weighted aggregation: entities quan trọng hơn relations
- Domain-specific embeddings thay vì general-purpose GloVe

Pipeline:
    Step 1: Train TransE
        total_id.txt → TransE Model → entity_emb.npy, relation_emb.npy
    
    Step 2: Create Features
        captions + TransE embeddings → per-video features

Kết quả dự kiến: ~70%+ unique embeddings ✅✅
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

# ============== CONFIGURATION ==============
# Kaggle paths
GLOVE_PATH = '/kaggle/input/glove-6b/glove.6B/glove.6B.300d.txt'
OPENKE_DIR = '/kaggle/working/OPENKE_file'
METADATA_DIR = '/kaggle/input/btkg-msvd-dataset/MSVD/metadata'
OUTPUT_DIR = '/kaggle/working/features_sota'

EMBEDDING_DIM = 300
NUM_FRAMES = 28

# TransE hyperparameters
TRANSE_EPOCHS = 500
TRANSE_BATCH_SIZE = 256
TRANSE_LR = 0.01
TRANSE_MARGIN = 1.0

# Feature creation options
USE_TRANSE = True           # True: dùng TransE, False: dùng GloVe
ENTITY_WEIGHT = 0.7         # Trọng số cho entities
RELATION_WEIGHT = 0.3       # Trọng số cho relations

# ============== STEP 1: TRANSE TRAINING ==============
class TransEModel(nn.Module):
    """
    TransE: Translation-based Embedding
    
    Idea: h + r ≈ t (head + relation ≈ tail)
    
    Ví dụ: embedding("man") + embedding("play") ≈ embedding("guitar")
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int, margin: float = 1.0):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, dim)
        self.rel_embeddings = nn.Embedding(num_relations, dim)
        self.margin = margin
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.ent_embeddings.weight.data = nn.functional.normalize(
                self.ent_embeddings.weight.data, dim=1
            )
    
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score function: ||h + r - t||
        Lower score = better triple
        """
        h_emb = self.ent_embeddings(h)
        r_emb = self.rel_embeddings(r)
        t_emb = self.ent_embeddings(t)
        
        return torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
    
    def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        Margin-based ranking loss:
        L = max(0, margin + score(pos) - score(neg))
        """
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


class TripleDataset(Dataset):
    """Dataset cho TransE training"""
    def __init__(self, triples: List[Tuple[int, int, int]], num_entities: int):
        self.triples = triples
        self.num_entities = num_entities
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, t, r = self.triples[idx]
        
        # Negative sampling: corrupt head or tail randomly
        if np.random.random() < 0.5:
            neg_h = np.random.randint(self.num_entities)
            neg_t = t
        else:
            neg_h = h
            neg_t = np.random.randint(self.num_entities)
        
        return (
            torch.tensor(h), torch.tensor(r), torch.tensor(t),
            torch.tensor(neg_h), torch.tensor(r), torch.tensor(neg_t)
        )


def load_triples_for_training(openke_dir: str) -> Tuple[List, int, int]:
    """Load triples và đếm entities/relations"""
    # Đếm entities và relations
    num_entities = 0
    num_relations = 0
    
    ent_path = os.path.join(openke_dir, 'entity2id.txt')
    rel_path = os.path.join(openke_dir, 'relation2id.txt')
    
    if os.path.exists(ent_path):
        with open(ent_path, 'r') as f:
            num_entities = len(f.readlines())
    
    if os.path.exists(rel_path):
        with open(rel_path, 'r') as f:
            num_relations = len(f.readlines())
    
    # Load triples
    triples = []
    
    # Thử train2id.txt trước (nếu đã chạy segmentation.py)
    train_path = os.path.join(openke_dir, 'train2id.txt')
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, t, r))
    else:
        # Fallback to total_id.txt
        total_path = os.path.join(openke_dir, 'total_id.txt')
        if os.path.exists(total_path):
            with open(total_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                        triples.append((h, t, r))
    
    return triples, num_entities, num_relations


def train_transe(openke_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Training TransE model
    
    Returns:
        entity_embeddings: (num_entities, dim)
        relation_embeddings: (num_relations, dim)
    """
    print("\n" + "="*60)
    print("STEP 1: Training TransE")
    print("="*60)
    
    # Load data
    triples, num_entities, num_relations = load_triples_for_training(openke_dir)
    
    if not triples:
        print("No triples found for training!")
        return None, None
    
    print(f"Training data:")
    print(f"  - Entities: {num_entities}")
    print(f"  - Relations: {num_relations}")
    print(f"  - Triples: {len(triples)}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Device: {device}")
    
    model = TransEModel(num_entities, num_relations, EMBEDDING_DIM, TRANSE_MARGIN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRANSE_LR)
    
    dataset = TripleDataset(triples, num_entities)
    dataloader = DataLoader(dataset, batch_size=TRANSE_BATCH_SIZE, shuffle=True)
    
    # Training loop
    print(f"\nTraining for {TRANSE_EPOCHS} epochs...")
    
    for epoch in range(TRANSE_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            loss = model.loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings after each update
            with torch.no_grad():
                model.ent_embeddings.weight.data = nn.functional.normalize(
                    model.ent_embeddings.weight.data, dim=1
                )
            
            total_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{TRANSE_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Extract embeddings
    entity_emb = model.ent_embeddings.weight.data.cpu().numpy()
    relation_emb = model.rel_embeddings.weight.data.cpu().numpy()
    
    # Save embeddings
    np.save(os.path.join(openke_dir, 'entity_embeddings_transe.npy'), entity_emb)
    np.save(os.path.join(openke_dir, 'relation_embeddings_transe.npy'), relation_emb)
    
    print(f"\nTransE training completed!")
    print(f"  - Entity embeddings: {entity_emb.shape}")
    print(f"  - Relation embeddings: {relation_emb.shape}")
    
    return entity_emb, relation_emb


# ============== STEP 2: PER-VIDEO FEATURE CREATION ==============
def load_glove(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings (fallback nếu không có TransE)"""
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
    """Load vocabulary với synonyms"""
    entities = []
    word2entity = {}
    
    with open(ent_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                main = parts[0].strip()
                entities.append(main)
                
                if main:
                    word2entity[main.lower()] = idx
                
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
                
                if main:
                    word2relation[main.lower()] = idx
                
                if len(parts) > 1 and parts[1]:
                    for syn in parts[1].split('#'):
                        syn = syn.strip().lower()
                        if syn:
                            word2relation[syn] = idx
    
    return word2entity, word2relation, entities, relations


def create_glove_embeddings(words: List[str], glove: Dict[str, np.ndarray]) -> np.ndarray:
    """Tạo embeddings từ GloVe (fallback)"""
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
    """Tiền xử lý text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def extract_from_caption(
    caption: str,
    word2entity: Dict[str, int],
    word2relation: Dict[str, int]
) -> Tuple[Set[int], Set[int]]:
    """Trích xuất entity và relation IDs từ caption"""
    words = preprocess_text(caption)
    
    entity_ids = set()
    relation_ids = set()
    
    for word in words:
        if word in word2entity:
            entity_ids.add(word2entity[word])
        if word in word2relation:
            relation_ids.add(word2relation[word])
    
    # 2-gram matching
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if phrase in word2entity:
            entity_ids.add(word2entity[phrase])
        if phrase in word2relation:
            relation_ids.add(word2relation[phrase])
    
    return entity_ids, relation_ids


def create_video_feature_weighted(
    entity_ids: Set[int],
    relation_ids: Set[int],
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    entity_weight: float = 0.7,
    relation_weight: float = 0.3
) -> np.ndarray:
    """
    Tạo feature với WEIGHTED aggregation
    
    Feature = entity_weight * mean(entity_embs) + relation_weight * mean(relation_embs)
    
    Tại sao weighted?
    - Entities (objects) quan trọng hơn cho video understanding
    - Relations (actions) bổ sung ngữ cảnh
    """
    # Collect entity embeddings
    entity_features = []
    for ent_id in entity_ids:
        if ent_id < len(entity_emb):
            entity_features.append(entity_emb[ent_id])
    
    # Collect relation embeddings
    relation_features = []
    for rel_id in relation_ids:
        if rel_id < len(relation_emb):
            relation_features.append(relation_emb[rel_id])
    
    # Weighted aggregation
    if entity_features:
        ent_mean = np.mean(entity_features, axis=0)
    else:
        ent_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    if relation_features:
        rel_mean = np.mean(relation_features, axis=0)
    else:
        rel_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Combine with weights
    if entity_features or relation_features:
        feature = entity_weight * ent_mean + relation_weight * rel_mean
    else:
        feature = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Repeat cho mỗi frame
    return np.tile(feature, (NUM_FRAMES, 1)).astype(np.float32)


def load_video_captions(csv_path: str) -> Dict[str, List[str]]:
    """Load captions grouped by video"""
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
    
    stats = {'with_entities': 0, 'with_relations': 0, 'total_ent': 0, 'total_rel': 0}
    
    with h5py.File(output_path, 'w') as hf:
        for vid in tqdm(video_ids, desc=split):
            captions = video_captions[vid]
            
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
            stats['total_ent'] += len(all_entity_ids)
            stats['total_rel'] += len(all_relation_ids)
            
            # Create weighted feature
            rel_feat = create_video_feature_weighted(
                all_entity_ids, all_relation_ids,
                entity_emb, relation_emb,
                ENTITY_WEIGHT, RELATION_WEIGHT
            )
            
            hf.create_dataset(str(vid), data=rel_feat)
    
    print(f"Saved: {output_path}")
    print(f"  - With entities: {stats['with_entities']}/{len(video_ids)} "
          f"({stats['with_entities']/len(video_ids)*100:.1f}%)")
    print(f"  - Avg entities/video: {stats['total_ent']/len(video_ids):.2f}")
    print(f"  - Avg relations/video: {stats['total_rel']/len(video_ids):.2f}")


# ============== MAIN ==============
def main():
    print("="*60)
    print("SOTA: TransE + Weighted Aggregation")
    print("="*60)
    
    # Load vocabulary
    ent_path = os.path.join(OPENKE_DIR, 'ent.txt')
    rel_path = os.path.join(OPENKE_DIR, 'rel.txt')
    word2entity, word2relation, entities, relations = load_vocabulary_with_synonyms(
        ent_path, rel_path
    )
    
    # Get embeddings
    if USE_TRANSE:
        # Check if TransE embeddings exist
        ent_emb_path = os.path.join(OPENKE_DIR, 'entity_embeddings_transe.npy')
        rel_emb_path = os.path.join(OPENKE_DIR, 'relation_embeddings_transe.npy')
        
        if os.path.exists(ent_emb_path) and os.path.exists(rel_emb_path):
            print("\nLoading existing TransE embeddings...")
            entity_emb = np.load(ent_emb_path)
            relation_emb = np.load(rel_emb_path)
        else:
            print("\nTraining new TransE embeddings...")
            entity_emb, relation_emb = train_transe(OPENKE_DIR)
            
            if entity_emb is None:
                print("TransE training failed! Falling back to GloVe...")
                USE_TRANSE = False
    
    if not USE_TRANSE:
        print("\nUsing GloVe embeddings...")
        glove = load_glove(GLOVE_PATH)
        entity_emb = create_glove_embeddings(entities, glove)
        relation_emb = create_glove_embeddings(relations, glove)
    
    print(f"\nEmbedding shapes:")
    print(f"  - Entity: {entity_emb.shape}")
    print(f"  - Relation: {relation_emb.shape}")
    print(f"\nAggregation weights:")
    print(f"  - Entity weight: {ENTITY_WEIGHT}")
    print(f"  - Relation weight: {RELATION_WEIGHT}")
    
    # Create features
    print("\n" + "="*60)
    print("STEP 2: Creating Per-Video Features")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        create_hdf5_for_split(
            split, word2entity, word2relation,
            entity_emb, relation_emb,
            METADATA_DIR, OUTPUT_DIR
        )
    
    print("\n" + "="*60)
    print("DONE! SOTA features created ✅✅")
    print(f"Output: {OUTPUT_DIR}/MSVD_rel_{{train,val,test}}.hdf5")
    print("="*60)


if __name__ == '__main__':
    main()
