"""
================================================================================
HƯỚNG DẪN CHẠY TRÊN KAGGLE - COMPLETE GUIDE
================================================================================

Hướng dẫn đầy đủ để chạy các phương pháp tạo rel_feature trên Kaggle.
Copy các cells bên dưới vào Kaggle Notebook.

Datasets cần add:
- glove.6b (cho GloVe embeddings)
- btkg-msvd-dataset (cho MSVD data)

================================================================================
"""

# ============================================================
# CELL 1: Setup và Upload Files
# ============================================================
"""
# Tạo thư mục OPENKE_file
!mkdir -p /kaggle/working/OPENKE_file

# Upload ent.txt và rel.txt từ local
# Hoặc tạo trực tiếp:
"""

ENT_TXT_CONTENT = """person|man#guy#woman#lady#girl#boy#kid#child#baby#people
dog|puppy#dogs#hound
cat|kitten#cats#kitty
car|vehicle#automobile#cars
bicycle|bike#cycle
horse|pony#horses
ball|balls
guitar|guitars
piano|pianos#keyboard
food|meal#dish#dishes#eating
water|liquid#drink#drinks#drinking
table|desk#tables
chair|seat#chairs
phone|telephone#cellphone#mobile
book|books#reading
computer|laptop#pc#desktop
tv|television#screen#monitor
camera|cameras
glass|glasses#cup#cups
bottle|bottles
door|doors#gate
window|windows
bed|beds#sleeping
house|home#building
tree|trees#plant#plants
flower|flowers
road|street#path#highway
sky|skies
sun|sunshine
snow|ice#snowy
"""

REL_TXT_CONTENT = """play|playing#plays#played
eat|eating#eats#ate
drink|drinking#drinks#drank
drive|driving#drives#drove#ride#riding
walk|walking#walks#walked
run|running#runs#ran
sit|sitting#sits#sat
stand|standing#stands#stood
talk|talking#talks#talked#speak#speaking
watch|watching#watches#watched
read|reading#reads
write|writing#writes#wrote
cook|cooking#cooks#cooked
sleep|sleeping#sleeps#slept
swim|swimming#swims#swam
dance|dancing#dances#danced
sing|singing#sings#sang
jump|jumping#jumps#jumped
climb|climbing#climbs#climbed
throw|throwing#throws#threw
catch|catching#catches#caught
"""

# ============================================================
# CELL 2: Chọn phương pháp để chạy
# ============================================================
"""
# Uncomment phương pháp muốn chạy:

# Phương pháp 1: Global Average (V1 - không khuyến khích)
# %run /kaggle/working/OPENKE_file/method1_global_average.py

# Phương pháp 2: Caption Matching (V2 - khuyến khích cho quick test)
# %run /kaggle/working/OPENKE_file/method2_caption_matching.py

# Phương pháp 3: TransE + Weighted (SOTA - khuyến khích)
# %run /kaggle/working/OPENKE_file/method3_sota_transe.py

# Phương pháp 4: Full SOTA với Object Detection
# %run /kaggle/working/OPENKE_file/method4_full_sota.py
"""

# ============================================================
# CELL 3: Quick Test - Phương pháp 2 (Caption Matching)
# ============================================================
QUICK_TEST_CODE = """
import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

# Configuration
GLOVE_PATH = '/kaggle/input/glove-6b/glove.6B/glove.6B.300d.txt'
OPENKE_DIR = '/kaggle/working/OPENKE_file'
METADATA_DIR = '/kaggle/input/btkg-msvd-dataset/MSVD/metadata'
OUTPUT_DIR = '/kaggle/working/features_v2'
EMBEDDING_DIM = 300
NUM_FRAMES = 28

# Create vocabulary files
def create_vocab_files():
    os.makedirs(OPENKE_DIR, exist_ok=True)
    
    with open(os.path.join(OPENKE_DIR, 'ent.txt'), 'w') as f:
        f.write(ENT_TXT_CONTENT)
    
    with open(os.path.join(OPENKE_DIR, 'rel.txt'), 'w') as f:
        f.write(REL_TXT_CONTENT)
    
    print("Created vocabulary files!")

# Load functions
def load_glove(path):
    print(f"Loading GloVe...")
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == EMBEDDING_DIM:
                embeddings[word] = vector
    print(f"Loaded {len(embeddings)} vectors")
    return embeddings

def load_vocab(ent_path, rel_path):
    word2entity = {}
    entities = []
    
    with open(ent_path, 'r') as f:
        for idx, line in enumerate(f):
            if '|' in line:
                parts = line.strip().split('|')
                main = parts[0].strip()
                entities.append(main)
                
                if main:
                    word2entity[main.lower()] = idx
                
                if len(parts) > 1:
                    for syn in parts[1].split('#'):
                        syn = syn.strip().lower()
                        if syn:
                            word2entity[syn] = idx
    
    word2relation = {}
    relations = []
    
    with open(rel_path, 'r') as f:
        for idx, line in enumerate(f):
            if '|' in line:
                parts = line.strip().split('|')
                main = parts[0].strip()
                relations.append(main)
                
                if main:
                    word2relation[main.lower()] = idx
                
                if len(parts) > 1:
                    for syn in parts[1].split('#'):
                        syn = syn.strip().lower()
                        if syn:
                            word2relation[syn] = idx
    
    return word2entity, word2relation, entities, relations

def create_embeddings(words, glove):
    emb = np.zeros((len(words), EMBEDDING_DIM), dtype=np.float32)
    for idx, word in enumerate(words):
        w = word.lower()
        if w in glove:
            emb[idx] = glove[w]
    return emb

def process_caption(text, w2e, w2r):
    text = text.lower()
    text = re.sub(r'[^\\w\\s]', ' ', text)
    words = text.split()
    
    ent_ids = set()
    rel_ids = set()
    
    for w in words:
        if w in w2e:
            ent_ids.add(w2e[w])
        if w in w2r:
            rel_ids.add(w2r[w])
    
    return ent_ids, rel_ids

def create_feature(ent_ids, rel_ids, ent_emb, rel_emb):
    features = []
    
    for eid in ent_ids:
        if eid < len(ent_emb):
            features.append(ent_emb[eid])
    
    for rid in rel_ids:
        if rid < len(rel_emb):
            features.append(rel_emb[rid])
    
    if features:
        feat = np.mean(features, axis=0)
    else:
        feat = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    return np.tile(feat, (NUM_FRAMES, 1)).astype(np.float32)

def main():
    # Create vocab
    create_vocab_files()
    
    # Load
    glove = load_glove(GLOVE_PATH)
    w2e, w2r, ents, rels = load_vocab(
        os.path.join(OPENKE_DIR, 'ent.txt'),
        os.path.join(OPENKE_DIR, 'rel.txt')
    )
    
    ent_emb = create_embeddings(ents, glove)
    rel_emb = create_embeddings(rels, glove)
    
    print(f"Entities: {len(ents)}, Relations: {len(rels)}")
    
    # Process each split
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(METADATA_DIR, f'{split}.csv')
        df = pd.read_csv(csv_path)
        
        video_captions = defaultdict(list)
        for _, row in df.iterrows():
            vid = row['VideoID']
            cap = row['Description']
            if pd.notna(cap):
                video_captions[vid].append(str(cap))
        
        output_path = os.path.join(OUTPUT_DIR, f'MSVD_rel_{split}.hdf5')
        
        with h5py.File(output_path, 'w') as hf:
            for vid in tqdm(video_captions.keys(), desc=split):
                all_ent = set()
                all_rel = set()
                
                for cap in video_captions[vid]:
                    e, r = process_caption(cap, w2e, w2r)
                    all_ent.update(e)
                    all_rel.update(r)
                
                feat = create_feature(all_ent, all_rel, ent_emb, rel_emb)
                hf.create_dataset(str(vid), data=feat)
        
        print(f"Created: {output_path}")
    
    print("\\nDone! ✅")

main()
"""

# ============================================================
# CELL 4: Verify Results
# ============================================================
VERIFY_CODE = """
import h5py
import numpy as np

def verify_results(path):
    print(f"\\nVerifying: {path}")
    
    with h5py.File(path, 'r') as hf:
        vids = list(hf.keys())
        print(f"Videos: {len(vids)}")
        
        # Check first video
        if vids:
            data = hf[vids[0]][:]
            print(f"Shape: {data.shape}")
        
        # Uniqueness
        first_frames = []
        for vid in vids:
            first_frames.append(hf[vid][0])
        
        first_frames = np.array(first_frames)
        unique = np.unique(first_frames, axis=0)
        print(f"Unique: {len(unique)}/{len(vids)} ({len(unique)/len(vids)*100:.1f}%)")

# Verify
for split in ['train', 'val', 'test']:
    verify_results(f'/kaggle/working/features_v2/MSVD_rel_{split}.hdf5')
"""

# ============================================================
# CELL 5: Copy to data folder
# ============================================================
COPY_CODE = """
import shutil
import os

src_dir = '/kaggle/working/features_v2'
dst_dir = '/kaggle/input/btkg-msvd-dataset/MSVD/features'

# Note: On Kaggle, input is read-only. Use working directory instead.
# Or create a new dataset with the results.

# For training, update your config to point to the new path:
# FEATURE_DIR = '/kaggle/working/features_v2'

print("Features ready at:", src_dir)
print("Update your config.py to use this path for rel features.")
"""

print(__doc__)
print("\nCác script đã được tạo sẵn:")
print("1. method1_global_average.py - V1 Original (không khuyến khích)")
print("2. method2_caption_matching.py - V2 Caption Matching (khuyến khích)")
print("3. method3_sota_transe.py - TransE + Weighted (SOTA)")
print("4. method4_full_sota.py - Full SOTA với Object Detection")
print("5. compare_all_methods.py - So sánh tất cả phương pháp")
print("\nCopy các file này lên Kaggle và chạy theo thứ tự phù hợp.")
