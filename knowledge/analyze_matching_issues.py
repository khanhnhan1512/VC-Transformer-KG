"""
analyze_matching_issues.py - Script to analyze why relations fail to match

This script:
1. Reads entity_total.txt (REBEL output)
2. Tests matching against rel.txt rules
3. Reports which relations fail to match and why
4. Suggests fixes for rel.txt
"""

import os
import re
from collections import Counter

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    lemmatizer = None


def lemmatize_verb(word):
    if not NLTK_AVAILABLE or not word:
        return word.lower().strip()
    return lemmatizer.lemmatize(word.lower().strip(), 'v')


def word_boundary_match(pattern, text):
    if not pattern or not text:
        return False
    escaped_pattern = re.escape(pattern)
    return bool(re.search(r'\b' + escaped_pattern + r'\b', text, re.IGNORECASE))


def find_best_match(text, induction_lines):
    """Test matching logic - returns (match_result, match_type)"""
    text = text.strip().lower()
    text_lemma = lemmatize_verb(text)
    
    # Exact match
    for line in induction_lines:
        if '|' not in line:
            continue
        normalized = line.split('|')[0].strip()
        variants = line.split('|')[1].replace('\n', '').split('#')
        
        for variant in variants:
            variant = variant.strip().lower()
            if variant == text:
                return normalized, 'exact'
            if variant == text_lemma and text_lemma != text:
                return normalized, 'lemma_exact'
    
    # Word boundary match
    best = None
    best_len = 0
    match_type = None
    
    for line in induction_lines:
        if '|' not in line:
            continue
        normalized = line.split('|')[0].strip()
        variants = line.split('|')[1].replace('\n', '').split('#')
        
        for variant in variants:
            variant = variant.strip()
            if not variant:
                continue
            
            if word_boundary_match(variant, text):
                if len(variant) > best_len:
                    best = normalized
                    best_len = len(variant)
                    match_type = 'word_boundary'
            
            if text_lemma != text and word_boundary_match(variant, text_lemma):
                if len(variant) > best_len:
                    best = normalized
                    best_len = len(variant)
                    match_type = 'lemma_word_boundary'
    
    return (best, match_type) if best else ('none', 'no_match')


def is_valid_relation(rel):
    invalid = {'be', 'is', 'are', 'was', 'were', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did'}
    return rel.lower().strip() not in invalid


def main():
    # Paths - adjust as needed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_induction_path = os.path.join(base_dir, 'OPENKE_file', 'rel.txt')
    entity_total_path = os.path.join(base_dir, 'OPENKE_file', 'entity_total.txt')
    
    # Try Kaggle path if local doesn't exist
    if not os.path.exists(entity_total_path):
        entity_total_path = '/kaggle/working/entity_total.txt'
    
    print(f"Analyzing matching issues...")
    print(f"rel.txt: {rel_induction_path}")
    print(f"entity_total: {entity_total_path}")
    print(f"NLTK available: {NLTK_AVAILABLE}")
    print()
    
    # Load rel.txt rules
    with open(rel_induction_path, 'r', encoding='utf-8') as f:
        rel_lines = f.readlines()
    
    # Extract all relation base forms from rel.txt
    rel_bases = set()
    for line in rel_lines:
        if '|' in line:
            rel_bases.add(line.split('|')[0].strip().lower())
    
    print(f"Relation base forms in rel.txt: {len(rel_bases)}")
    print(f"Sample: {list(rel_bases)[:10]}")
    print()
    
    # Load and analyze entity_total.txt
    if not os.path.exists(entity_total_path):
        print(f"[error] Cannot find entity_total.txt at {entity_total_path}")
        print("Please provide the path to entity_total.txt")
        return
    
    with open(entity_total_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Analyze relations
    relation_counts = Counter()
    match_results = Counter()
    unmatched = Counter()
    matched_by_lemma = Counter()
    
    for line in lines:
        parts = line.strip().split('&')
        if len(parts) != 3:
            continue
        
        relation = parts[2].strip()
        relation_counts[relation] += 1
        
        if not is_valid_relation(relation):
            match_results['invalid'] += 1
            continue
        
        result, match_type = find_best_match(relation, rel_lines)
        
        if result == 'none':
            match_results['no_match'] += 1
            lemma = lemmatize_verb(relation)
            unmatched[f"{relation} (lemma: {lemma})"] += 1
        else:
            match_results[match_type] += 1
            if 'lemma' in match_type:
                matched_by_lemma[f"{relation} -> {result}"] += 1
    
    # Report
    print("="*60)
    print("RELATION ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nTotal relations: {sum(relation_counts.values())}")
    print(f"Unique relations: {len(relation_counts)}")
    
    print(f"\nMatch results:")
    for mtype, count in sorted(match_results.items(), key=lambda x: -x[1]):
        pct = count / sum(match_results.values()) * 100
        print(f"  {mtype}: {count} ({pct:.1f}%)")
    
    print(f"\nTop 30 unmatched relations (need to add to rel.txt):")
    for rel, count in unmatched.most_common(30):
        print(f"  {rel}: {count}")
    
    if matched_by_lemma:
        print(f"\nMatched by lemmatization (showing lemma benefit):")
        for rel, count in matched_by_lemma.most_common(20):
            print(f"  {rel}: {count}")
    
    print(f"\nTop 30 original relations from REBEL:")
    for rel, count in relation_counts.most_common(30):
        print(f"  {rel}: {count}")


if __name__ == '__main__':
    main()
