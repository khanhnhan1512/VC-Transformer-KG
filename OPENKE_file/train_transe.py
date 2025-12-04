"""
SOTA Pipeline: Training TransE với OpenKE
==========================================

Bước này training Knowledge Graph Embeddings thay vì dùng GloVe.
TransE học relationship: head + relation ≈ tail

Requirements:
    pip install openke  # hoặc clone từ https://github.com/thunlp/OpenKE

Input files (từ pipeline knowledge/):
    - train2id.txt: training triples (format: head_id tail_id relation_id)
    - entity2id.txt: entity → id mapping
    - relation2id.txt: relation → id mapping

Output:
    - entity_embeddings.npy: (num_entities, embedding_dim)
    - relation_embeddings.npy: (num_relations, embedding_dim)
"""

import os
import numpy as np

# ============== CONFIGURATION ==============
# Kaggle paths
OPENKE_DIR = '/kaggle/working/OPENKE_file'
OUTPUT_DIR = '/kaggle/working/OPENKE_file'

EMBEDDING_DIM = 300  # Phải match với GloVe dim nếu muốn kết hợp
EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.01
MARGIN = 1.0

# ============== PREPARE DATA FOR OPENKE ==============
def prepare_openke_data():
    """
    Chuẩn bị data theo format OpenKE yêu cầu
    
    OpenKE cần:
    - benchmarks/<dataset>/train2id.txt: num_triples \n head tail relation
    - benchmarks/<dataset>/entity2id.txt: num_entities \n entity id
    - benchmarks/<dataset>/relation2id.txt: num_relations \n relation id
    """
    benchmark_dir = os.path.join(OPENKE_DIR, 'benchmarks', 'msvd')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Convert entity2id.txt
    ent2id = {}
    with open(os.path.join(OPENKE_DIR, 'entity2id.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(benchmark_dir, 'entity2id.txt'), 'w') as f:
        f.write(f"{len(lines)}\n")
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, idx = parts[0], int(parts[1])
                ent2id[entity] = idx
                f.write(f"{entity}\t{idx}\n")
    
    # Convert relation2id.txt
    rel2id = {}
    with open(os.path.join(OPENKE_DIR, 'relation2id.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(benchmark_dir, 'relation2id.txt'), 'w') as f:
        f.write(f"{len(lines)}\n")
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, idx = parts[0], int(parts[1])
                rel2id[relation] = idx
                f.write(f"{relation}\t{idx}\n")
    
    # Convert train2id.txt (head tail rel format)
    with open(os.path.join(OPENKE_DIR, 'train2id.txt'), 'r') as f:
        lines = f.readlines()
    
    triples = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            triples.append((h, t, r))
    
    with open(os.path.join(benchmark_dir, 'train2id.txt'), 'w') as f:
        f.write(f"{len(triples)}\n")
        for h, t, r in triples:
            f.write(f"{h}\t{t}\t{r}\n")
    
    print(f"Prepared OpenKE data:")
    print(f"  - Entities: {len(ent2id)}")
    print(f"  - Relations: {len(rel2id)}")
    print(f"  - Training triples: {len(triples)}")
    
    return len(ent2id), len(rel2id), benchmark_dir

# ============== TRAINING WITH OPENKE ==============
def train_transe_openke(num_entities: int, num_relations: int, benchmark_dir: str):
    """
    Training TransE với OpenKE framework
    """
    try:
        from openke.config import Trainer, Tester
        from openke.module.model import TransE
        from openke.module.loss import MarginLoss
        from openke.module.strategy import NegativeSampling
        from openke.data import TrainDataLoader, TestDataLoader
    except ImportError:
        print("OpenKE not installed. Using fallback method...")
        return train_transe_manual(num_entities, num_relations)
    
    # DataLoader
    train_dataloader = TrainDataLoader(
        in_path=benchmark_dir + "/",
        nbatches=100,
        threads=4,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )
    
    # Model
    transe = TransE(
        ent_tot=num_entities,
        rel_tot=num_relations,
        dim=EMBEDDING_DIM,
        p_norm=1,
        norm_flag=True
    )
    
    # Loss & Strategy
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=MARGIN),
        batch_size=train_dataloader.get_batch_size()
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=EPOCHS,
        alpha=LEARNING_RATE,
        use_gpu=True,
        opt_method="adam"
    )
    
    trainer.run()
    
    # Extract embeddings
    entity_embeddings = transe.ent_embeddings.weight.data.cpu().numpy()
    relation_embeddings = transe.rel_embeddings.weight.data.cpu().numpy()
    
    return entity_embeddings, relation_embeddings

# ============== FALLBACK: MANUAL TRANSE TRAINING ==============
def train_transe_manual(num_entities: int, num_relations: int):
    """
    Training TransE thủ công nếu không có OpenKE
    
    TransE: h + r ≈ t
    Loss: max(0, margin + ||h + r - t|| - ||h' + r - t'||)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    print("Training TransE manually...")
    
    class TransEModel(nn.Module):
        def __init__(self, num_ent, num_rel, dim, margin=1.0):
            super().__init__()
            self.ent_embeddings = nn.Embedding(num_ent, dim)
            self.rel_embeddings = nn.Embedding(num_rel, dim)
            self.margin = margin
            
            # Initialize
            nn.init.xavier_uniform_(self.ent_embeddings.weight)
            nn.init.xavier_uniform_(self.rel_embeddings.weight)
            
            # Normalize
            self.ent_embeddings.weight.data = nn.functional.normalize(
                self.ent_embeddings.weight.data, dim=1
            )
        
        def forward(self, h, r, t):
            h_emb = self.ent_embeddings(h)
            r_emb = self.rel_embeddings(r)
            t_emb = self.ent_embeddings(t)
            
            # Score: ||h + r - t||
            score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
            return score
        
        def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
            pos_score = self.forward(pos_h, pos_r, pos_t)
            neg_score = self.forward(neg_h, neg_r, neg_t)
            
            # Margin loss
            loss = torch.mean(torch.relu(self.margin + pos_score - neg_score))
            return loss
    
    class TripleDataset(Dataset):
        def __init__(self, triples, num_ent):
            self.triples = triples
            self.num_ent = num_ent
        
        def __len__(self):
            return len(self.triples)
        
        def __getitem__(self, idx):
            h, t, r = self.triples[idx]
            
            # Negative sampling: corrupt head or tail
            if np.random.random() < 0.5:
                neg_h = np.random.randint(self.num_ent)
                neg_t = t
            else:
                neg_h = h
                neg_t = np.random.randint(self.num_ent)
            
            return (
                torch.tensor(h), torch.tensor(r), torch.tensor(t),
                torch.tensor(neg_h), torch.tensor(r), torch.tensor(neg_t)
            )
    
    # Load triples
    triples = []
    train_path = os.path.join(OPENKE_DIR, 'train2id.txt')
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, t, r))
    else:
        # Fallback to total_id.txt
        total_path = os.path.join(OPENKE_DIR, 'total_id.txt')
        with open(total_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, t, r))
    
    print(f"Loaded {len(triples)} triples for training")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransEModel(num_entities, num_relations, EMBEDDING_DIM, MARGIN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    dataset = TripleDataset(triples, num_entities)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            loss = model.loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            with torch.no_grad():
                model.ent_embeddings.weight.data = nn.functional.normalize(
                    model.ent_embeddings.weight.data, dim=1
                )
            
            total_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Extract embeddings
    entity_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
    relation_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
    
    return entity_embeddings, relation_embeddings

# ============== SAVE EMBEDDINGS ==============
def save_embeddings(entity_emb: np.ndarray, relation_emb: np.ndarray):
    """Lưu embeddings"""
    ent_path = os.path.join(OUTPUT_DIR, 'entity_embeddings_transe.npy')
    rel_path = os.path.join(OUTPUT_DIR, 'relation_embeddings_transe.npy')
    
    np.save(ent_path, entity_emb)
    np.save(rel_path, relation_emb)
    
    print(f"\nSaved embeddings:")
    print(f"  - Entity embeddings: {ent_path} {entity_emb.shape}")
    print(f"  - Relation embeddings: {rel_path} {relation_emb.shape}")

# ============== MAIN ==============
def main():
    print("="*60)
    print("SOTA: Training TransE Knowledge Graph Embeddings")
    print("="*60)
    
    # Count entities and relations
    num_entities = 0
    num_relations = 0
    
    ent_path = os.path.join(OPENKE_DIR, 'entity2id.txt')
    rel_path = os.path.join(OPENKE_DIR, 'relation2id.txt')
    
    if os.path.exists(ent_path):
        with open(ent_path, 'r') as f:
            num_entities = len(f.readlines())
    
    if os.path.exists(rel_path):
        with open(rel_path, 'r') as f:
            num_relations = len(f.readlines())
    
    print(f"Knowledge Graph size:")
    print(f"  - Entities: {num_entities}")
    print(f"  - Relations: {num_relations}")
    
    # Train TransE
    entity_emb, relation_emb = train_transe_manual(num_entities, num_relations)
    
    # Save
    save_embeddings(entity_emb, relation_emb)
    
    print("\n" + "="*60)
    print("TransE training completed!")
    print("="*60)

if __name__ == '__main__':
    main()
