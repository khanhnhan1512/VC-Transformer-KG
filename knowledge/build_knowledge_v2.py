"""
build_knowledge_v2.py - Improved version with lemmatization support

Key improvements over build_knowledge.py:
1. Adds lemmatization for relations (verb forms: playing->play, riding->ride)
2. Handles verb conjugations (played, plays, playing -> play)
3. Better fallback matching for relations
4. Detailed logging for debugging
"""

import os
import re
from tqdm import tqdm

# Try to import NLTK for lemmatization
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import nltk
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
    lemmatizer = WordNetLemmatizer()
    print("[info] NLTK lemmatizer loaded successfully")
except ImportError:
    NLTK_AVAILABLE = False
    lemmatizer = None
    print("[warn] NLTK not available, lemmatization disabled")


# Tạo thư mục output nếu chưa tồn tại
OUTPUT_DIR = '/kaggle/working/OPENKE_file'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Định nghĩa đường dẫn file
rel_induction = os.path.join(OUTPUT_DIR, 'rel.txt')
ent_induction = os.path.join(OUTPUT_DIR, 'ent.txt')
target_entity = '/kaggle/working/entity_total.txt'
entity2id = os.path.join(OUTPUT_DIR, 'entity2id.txt')
relation2id = os.path.join(OUTPUT_DIR, 'relation2id.txt')
total_word = os.path.join(OUTPUT_DIR, 'total_word.txt')
total_id = os.path.join(OUTPUT_DIR, 'total_id.txt')


def lemmatize_verb(word):
    """
    Lemmatize a verb to its base form.
    playing -> play, riding -> ride, cooked -> cook
    """
    if not NLTK_AVAILABLE or not word:
        return word.lower().strip()
    
    word = word.lower().strip()
    # Try lemmatizing as verb
    lemma = lemmatizer.lemmatize(word, 'v')
    return lemma


def word_boundary_match(pattern, text):
    """
    Kiểm tra xem pattern có xuất hiện như một từ hoàn chỉnh trong text không.
    """
    if not pattern or not text:
        return False
    escaped_pattern = re.escape(pattern)
    return bool(re.search(r'\b' + escaped_pattern + r'\b', text, re.IGNORECASE))


def find_best_match(text, induction_lines, match_type='entity'):
    """
    Tìm normalized form tốt nhất cho text dựa trên induction rules.
    
    With lemmatization support for relations:
    1. Exact match (highest priority)
    2. Lemmatized exact match (for relations)
    3. Word boundary match
    4. Lemmatized word boundary match (for relations)
    """
    text = text.strip().lower()
    text_lemma = lemmatize_verb(text) if match_type == 'relation' else text
    
    # Bước 1: Tìm exact match trước
    for line in induction_lines:
        if '|' not in line:
            continue
        step1 = line.split('|')
        normalized = step1[0].replace(' ', '').strip()
        variants = step1[1].replace('\n', '').split('#')
        
        for variant in variants:
            variant_clean = variant.strip().lower()
            if not variant_clean:
                continue
                
            # Exact match
            if variant_clean == text:
                return normalized
            
            # Lemmatized exact match for relations
            if match_type == 'relation':
                variant_lemma = lemmatize_verb(variant_clean)
                if variant_lemma == text_lemma and text_lemma != text:
                    return normalized
    
    # Bước 2: Word boundary match
    best_match = None
    best_match_len = 0
    
    for line in induction_lines:
        if '|' not in line:
            continue
        step1 = line.split('|')
        normalized = step1[0].replace(' ', '').strip()
        variants = step1[1].replace('\n', '').split('#')
        
        for variant in variants:
            variant = variant.strip()
            if not variant:
                continue
            
            # Word boundary match - original text
            if word_boundary_match(variant, text):
                if len(variant) > best_match_len:
                    best_match = normalized
                    best_match_len = len(variant)
            
            # Lemmatized word boundary match for relations
            if match_type == 'relation' and text_lemma != text:
                if word_boundary_match(variant, text_lemma):
                    if len(variant) > best_match_len:
                        best_match = normalized
                        best_match_len = len(variant)
    
    return best_match if best_match else 'none'


def is_valid_relation(relation):
    """
    Check if a relation is valid (not a copula/auxiliary verb).
    """
    invalid = {'be', 'is', 'are', 'was', 'were', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did', "'s", "'re", "'m"}
    return relation.lower().strip() not in invalid


# Đọc file input
print("[info] Loading induction files...")
with open(rel_induction, 'r', encoding='utf-8') as f:
    rel_induction_lines = f.readlines()

with open(ent_induction, 'r', encoding='utf-8') as f:
    ent_induction_lines = f.readlines()

with open(target_entity, 'r', encoding='utf-8') as f:
    target_lines = f.readlines()

print(f"[info] Loaded {len(rel_induction_lines)} relation rules, {len(ent_induction_lines)} entity rules")
print(f"[info] Processing {len(target_lines)} triples...")

# Build a quick lookup for relation base forms
relation_base_forms = set()
for line in rel_induction_lines:
    if '|' in line:
        base = line.split('|')[0].strip()
        relation_base_forms.add(base.lower())

print(f"[info] Available relation base forms: {len(relation_base_forms)}")

# Statistics for debugging
stats = {
    'total': 0,
    'ent1_none': 0,
    'ent2_none': 0,
    'rel_none': 0,
    'rel_invalid': 0,
    'rel_matched': {},
    'rel_unmatched': {},
}

# Phase 1: Convert to normalized words
with open(total_word, 'w', encoding='utf-8') as total_word_f:
    for target_line in tqdm(target_lines, desc="Processing triples"):
        parts = target_line.split('&')
        
        if len(parts) != 3:
            continue
        
        stats['total'] += 1
        entity1 = parts[0].strip()
        entity2 = parts[1].strip()
        relation = parts[2].strip()
        
        # Skip invalid relations (be, have, do, etc.)
        if not is_valid_relation(relation):
            stats['rel_invalid'] += 1
            total_word_f.write('none,none,none\n')
            continue
        
        # Normalize entity1
        entity1_normalized = find_best_match(entity1, ent_induction_lines, 'entity')
        if entity1_normalized == 'none':
            stats['ent1_none'] += 1
        
        # Normalize entity2
        entity2_normalized = find_best_match(entity2, ent_induction_lines, 'entity')
        if entity2_normalized == 'none':
            stats['ent2_none'] += 1
        
        # Normalize relation with lemmatization
        relation_normalized = find_best_match(relation, rel_induction_lines, 'relation')
        if relation_normalized == 'none':
            stats['rel_none'] += 1
            # Track unmatched relations for debugging
            rel_lemma = lemmatize_verb(relation)
            key = f"{relation} -> {rel_lemma}"
            stats['rel_unmatched'][key] = stats['rel_unmatched'].get(key, 0) + 1
        else:
            stats['rel_matched'][relation_normalized] = stats['rel_matched'].get(relation_normalized, 0) + 1
        
        total_word_f.write(f"{entity1_normalized},{entity2_normalized},{relation_normalized}\n")

print("\n" + "="*60)
print("PHASE 1 STATISTICS")
print("="*60)
print(f"Total triples processed: {stats['total']}")
print(f"Invalid relations (be/have/do): {stats['rel_invalid']} ({stats['rel_invalid']/stats['total']*100:.1f}%)")
print(f"Entity1 -> none: {stats['ent1_none']} ({stats['ent1_none']/stats['total']*100:.1f}%)")
print(f"Entity2 -> none: {stats['ent2_none']} ({stats['ent2_none']/stats['total']*100:.1f}%)")
print(f"Relation -> none: {stats['rel_none']} ({stats['rel_none']/stats['total']*100:.1f}%)")

print("\nTop 20 matched relations:")
for rel, count in sorted(stats['rel_matched'].items(), key=lambda x: -x[1])[:20]:
    print(f"  {rel}: {count}")

print("\nTop 20 unmatched relations (for debugging):")
for rel, count in sorted(stats['rel_unmatched'].items(), key=lambda x: -x[1])[:20]:
    print(f"  {rel}: {count}")


# Phase 2: Convert to IDs
def load_id_mapping(filepath):
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                name, id_str = parts
                mapping[name] = int(id_str)
    return mapping


reldict = load_id_mapping(relation2id)
entdict = load_id_mapping(entity2id)

print(f"\n[info] Loaded {len(entdict)} entities and {len(reldict)} relations for ID mapping")

valid_triples = 0
skipped_none = 0

with open(total_word, 'r', encoding='utf-8') as f_in, open(total_id, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in.readlines(), desc="Converting to IDs"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) != 3:
            continue
            
        entity1, entity2, relation = parts[0].strip(), parts[1].strip(), parts[2].strip()
        
        # Skip if any is none or not in mapping
        if entity1 == 'none' or entity2 == 'none' or relation == 'none':
            skipped_none += 1
            continue
        
        if entity1 not in entdict or entity2 not in entdict or relation not in reldict:
            skipped_none += 1
            continue
        
        f_out.write(f"{entdict[entity1]} {entdict[entity2]} {reldict[relation]}\n")
        valid_triples += 1

print("\n" + "="*60)
print("PHASE 2 STATISTICS")
print("="*60)
print(f"Valid triples written: {valid_triples}")
print(f"Skipped (none/missing): {skipped_none}")
print(f"Output file: {total_id}")
