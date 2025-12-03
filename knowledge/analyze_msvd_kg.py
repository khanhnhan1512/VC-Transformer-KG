"""
Phân tích entity_total.txt để tạo ent.txt và rel.txt phù hợp với MSVD
Input: entity_total.txt (format: subject&object&relation)
Output: 
    - ent_msvd.txt (entities phổ biến + synonyms)
    - rel_msvd.txt (relations phổ biến + synonyms)
    - analysis_report.txt (thống kê)
"""

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download NLTK data nếu cần
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ============== CONFIGURATION ==============
ENTITY_TOTAL_PATH = '/kaggle/working/entity_total.txt'
OUTPUT_DIR = '/kaggle/working/OPENKE_file'
ENT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'ent_msvd.txt')
REL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'rel_msvd.txt')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'analysis_report.txt')

# Thresholds
MIN_ENTITY_FREQ = 10      # Entity xuất hiện ít nhất 10 lần
MIN_RELATION_FREQ = 20    # Relation xuất hiện ít nhất 20 lần
TOP_ENTITIES = 150        # Lấy top 150 entities
TOP_RELATIONS = 50        # Lấy top 50 relations

# ============== HELPER FUNCTIONS ==============
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, remove special chars"""
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_lemma(word: str) -> str:
    """Lấy lemma của từ"""
    word = word.lower().strip()
    # Thử lemmatize như verb trước
    verb_lemma = lemmatizer.lemmatize(word, 'v')
    if verb_lemma != word:
        return verb_lemma
    # Thử lemmatize như noun
    noun_lemma = lemmatizer.lemmatize(word, 'n')
    return noun_lemma

def get_main_word(phrase: str) -> str:
    """Lấy từ chính từ phrase (thường là từ cuối hoặc từ đầu)"""
    words = phrase.split()
    if not words:
        return phrase
    
    # Loại bỏ articles và determiners
    stop_words = {'a', 'an', 'the', 'some', 'any', 'this', 'that', 'these', 'those'}
    filtered = [w for w in words if w not in stop_words]
    
    if filtered:
        # Trả về từ cuối (thường là head noun)
        return filtered[-1]
    return words[-1]

def is_valid_entity(entity: str) -> bool:
    """Kiểm tra entity có hợp lệ không"""
    if not entity or len(entity) < 2:
        return False
    # Loại bỏ pronouns, articles
    invalid = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 'him', 'her', 
               'them', 'us', 'me', 'a', 'an', 'the', 'this', 'that', 'one',
               'something', 'someone', 'anything', 'anyone', 'nothing', 'nobody'}
    if entity.lower() in invalid:
        return False
    return True

def is_valid_relation(relation: str) -> bool:
    """Kiểm tra relation có hợp lệ không"""
    if not relation or len(relation) < 2:
        return False
    # Loại bỏ be verbs đơn thuần
    invalid = {'be', 'is', 'are', 'was', 'were', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did'}
    if relation.lower() in invalid:
        return False
    return True

# ============== MAIN ANALYSIS ==============
def load_triples(filepath: str) -> List[Tuple[str, str, str]]:
    """Load triples từ entity_total.txt"""
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '&' not in line:
                continue
            parts = line.split('&')
            if len(parts) >= 3:
                subject = clean_text(parts[0])
                obj = clean_text(parts[1])
                relation = clean_text(parts[2])
                if subject and obj and relation:
                    triples.append((subject, obj, relation))
    return triples

def analyze_entities(triples: List[Tuple[str, str, str]]) -> Dict[str, Counter]:
    """Phân tích entities từ triples"""
    entity_counter = Counter()
    lemma_to_variants = defaultdict(set)
    
    for subject, obj, relation in triples:
        # Xử lý subject
        if is_valid_entity(subject):
            main_word = get_main_word(subject)
            lemma = get_lemma(main_word)
            entity_counter[lemma] += 1
            lemma_to_variants[lemma].add(subject)
            if main_word != subject:
                lemma_to_variants[lemma].add(main_word)
        
        # Xử lý object
        if is_valid_entity(obj):
            main_word = get_main_word(obj)
            lemma = get_lemma(main_word)
            entity_counter[lemma] += 1
            lemma_to_variants[lemma].add(obj)
            if main_word != obj:
                lemma_to_variants[lemma].add(main_word)
    
    return entity_counter, lemma_to_variants

def analyze_relations(triples: List[Tuple[str, str, str]]) -> Tuple[Counter, Dict]:
    """Phân tích relations từ triples"""
    relation_counter = Counter()
    lemma_to_variants = defaultdict(set)
    
    for subject, obj, relation in triples:
        if is_valid_relation(relation):
            # Lấy verb chính (từ đầu tiên thường là main verb)
            words = relation.split()
            main_verb = words[0] if words else relation
            lemma = get_lemma(main_verb)
            
            relation_counter[lemma] += 1
            lemma_to_variants[lemma].add(relation)
            if main_verb != relation:
                lemma_to_variants[lemma].add(main_verb)
    
    return relation_counter, lemma_to_variants

def create_entity_file(
    entity_counter: Counter, 
    lemma_to_variants: Dict[str, Set[str]],
    output_path: str,
    min_freq: int = 10,
    top_n: int = 150
) -> List[str]:
    """Tạo file ent_msvd.txt"""
    
    # Lọc và sắp xếp entities
    filtered = [(lemma, count) for lemma, count in entity_counter.items() 
                if count >= min_freq and is_valid_entity(lemma)]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:top_n]
    
    lines = []
    # Thêm background (bắt buộc)
    lines.append('__background|')
    
    for lemma, count in filtered:
        variants = lemma_to_variants[lemma]
        # Loại bỏ variants quá dài hoặc không hợp lệ
        clean_variants = set()
        for v in variants:
            v_clean = v.strip()
            if len(v_clean) <= 30 and is_valid_entity(v_clean):
                clean_variants.add(v_clean)
        
        # Tạo dòng format: main|variant1#variant2#...
        if clean_variants:
            variants_str = '#'.join(sorted(clean_variants))
            line = f"{lemma}|{variants_str}"
        else:
            line = f"{lemma}|{lemma}"
        lines.append(line)
    
    # Ghi file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Created {output_path} with {len(lines)} entities")
    return lines

def create_relation_file(
    relation_counter: Counter,
    lemma_to_variants: Dict[str, Set[str]],
    output_path: str,
    min_freq: int = 20,
    top_n: int = 50
) -> List[str]:
    """Tạo file rel_msvd.txt"""
    
    # Lọc và sắp xếp relations
    filtered = [(lemma, count) for lemma, count in relation_counter.items()
                if count >= min_freq and is_valid_relation(lemma)]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:top_n]
    
    lines = []
    for lemma, count in filtered:
        variants = lemma_to_variants[lemma]
        # Loại bỏ variants quá dài
        clean_variants = set()
        for v in variants:
            v_clean = v.strip()
            if len(v_clean) <= 30 and is_valid_relation(v_clean):
                clean_variants.add(v_clean)
        
        # Tạo dòng format: main|variant1#variant2#...
        if clean_variants:
            variants_str = '#'.join(sorted(clean_variants))
            line = f"{lemma}|{variants_str}"
        else:
            line = f"{lemma}|{lemma}"
        lines.append(line)
    
    # Ghi file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Created {output_path} with {len(lines)} relations")
    return lines

def create_report(
    triples: List[Tuple],
    entity_counter: Counter,
    relation_counter: Counter,
    output_path: str
):
    """Tạo báo cáo phân tích"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MSVD KNOWLEDGE GRAPH ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total triples: {len(triples)}\n")
        f.write(f"Unique entities (lemmatized): {len(entity_counter)}\n")
        f.write(f"Unique relations (lemmatized): {len(relation_counter)}\n\n")
        
        f.write("-"*40 + "\n")
        f.write("TOP 30 ENTITIES\n")
        f.write("-"*40 + "\n")
        for entity, count in entity_counter.most_common(30):
            f.write(f"  {entity}: {count}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("TOP 30 RELATIONS\n")
        f.write("-"*40 + "\n")
        for relation, count in relation_counter.most_common(30):
            f.write(f"  {relation}: {count}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("FREQUENCY DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        
        # Entity distribution
        ent_freq_ranges = [
            (1000, float('inf'), '1000+'),
            (500, 1000, '500-1000'),
            (100, 500, '100-500'),
            (50, 100, '50-100'),
            (10, 50, '10-50'),
            (1, 10, '1-10')
        ]
        f.write("\nEntity frequency distribution:\n")
        for low, high, label in ent_freq_ranges:
            count = sum(1 for _, c in entity_counter.items() if low <= c < high)
            f.write(f"  {label}: {count} entities\n")
        
        # Relation distribution
        f.write("\nRelation frequency distribution:\n")
        for low, high, label in ent_freq_ranges:
            count = sum(1 for _, c in relation_counter.items() if low <= c < high)
            f.write(f"  {label}: {count} relations\n")
    
    print(f"Created report: {output_path}")

# ============== MAIN ==============
def main():
    print("="*60)
    print("MSVD Knowledge Graph Analyzer")
    print("="*60)
    
    # Tạo output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load triples
    print(f"\nLoading triples from {ENTITY_TOTAL_PATH}...")
    triples = load_triples(ENTITY_TOTAL_PATH)
    print(f"Loaded {len(triples)} valid triples")
    
    if len(triples) == 0:
        print("ERROR: No triples found! Check entity_total.txt")
        return
    
    # Phân tích entities
    print("\nAnalyzing entities...")
    entity_counter, ent_variants = analyze_entities(triples)
    print(f"Found {len(entity_counter)} unique entities")
    
    # Phân tích relations
    print("\nAnalyzing relations...")
    relation_counter, rel_variants = analyze_relations(triples)
    print(f"Found {len(relation_counter)} unique relations")
    
    # Tạo files
    print("\n" + "-"*40)
    print("Creating output files...")
    
    create_entity_file(
        entity_counter, ent_variants, 
        ENT_OUTPUT_PATH,
        min_freq=MIN_ENTITY_FREQ,
        top_n=TOP_ENTITIES
    )
    
    create_relation_file(
        relation_counter, rel_variants,
        REL_OUTPUT_PATH,
        min_freq=MIN_RELATION_FREQ,
        top_n=TOP_RELATIONS
    )
    
    create_report(triples, entity_counter, relation_counter, REPORT_PATH)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total triples analyzed: {len(triples)}")
    print(f"Entity file: {ENT_OUTPUT_PATH}")
    print(f"Relation file: {REL_OUTPUT_PATH}")
    print(f"Report file: {REPORT_PATH}")
    
    print("\nTop 10 entities:")
    for ent, count in entity_counter.most_common(10):
        print(f"  {ent}: {count}")
    
    print("\nTop 10 relations:")
    for rel, count in relation_counter.most_common(10):
        print(f"  {rel}: {count}")

if __name__ == '__main__':
    main()