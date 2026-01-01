"""
compare_extraction_methods.py - Compare REBEL vs Stanford OpenIE vs spaCy

This script helps you decide which extraction method is best for your use case.

Usage:
    python knowledge/compare_extraction_methods.py --sample 100
"""

import argparse
from collections import Counter


def analyze_file(filepath, name):
    """Analyze entity_total.txt file from any extraction method."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[error] File not found: {filepath}")
        return None
    
    # Parse triples
    entities = Counter()
    relations = Counter()
    triples = []
    
    for line in lines:
        parts = line.strip().split(' & ')
        if len(parts) == 3:
            subj, obj, rel = parts
            entities[subj.lower()] += 1
            entities[obj.lower()] += 1
            relations[rel.lower()] += 1
            triples.append((subj, obj, rel))
    
    # Statistics
    print(f"\nBasic Statistics:")
    print(f"  Total triples: {len(triples)}")
    print(f"  Unique entities: {len(entities)}")
    print(f"  Unique relations: {len(relations)}")
    
    # Check for action verbs
    action_verbs = {'play', 'ride', 'cook', 'cut', 'walk', 'run', 'jump', 'swim',
                    'dance', 'sing', 'eat', 'drink', 'drive', 'climb', 'fall',
                    'throw', 'catch', 'kick', 'hit', 'shoot', 'fight', 'hold'}
    
    found_actions = set()
    action_count = 0
    for rel, count in relations.items():
        rel_first_word = rel.split()[0]
        if rel_first_word in action_verbs:
            found_actions.add(rel_first_word)
            action_count += count
    
    print(f"\nAction Verb Analysis:")
    print(f"  Action verbs found: {len(found_actions)}/{len(action_verbs)}")
    print(f"  Triples with action verbs: {action_count} ({action_count/len(triples)*100:.1f}%)")
    print(f"  Found: {sorted(found_actions)}")
    print(f"  Missing: {sorted(action_verbs - found_actions)}")
    
    # Wikidata-style relations
    wikidata_rels = {'subclass', 'instance', 'part', 'facet', 'product', 'participant',
                     'spouse', 'author', 'country', 'location', 'member', 'owner'}
    
    found_wikidata = set()
    wikidata_count = 0
    for rel, count in relations.items():
        rel_first_word = rel.split()[0]
        if rel_first_word in wikidata_rels:
            found_wikidata.add(rel_first_word)
            wikidata_count += count
    
    print(f"\nWikidata-style Relations:")
    print(f"  Found: {len(found_wikidata)}")
    print(f"  Triples with Wikidata relations: {wikidata_count} ({wikidata_count/len(triples)*100:.1f}%)")
    
    # Top relations
    print(f"\nTop 20 Relations:")
    for rel, count in relations.most_common(20):
        pct = count / len(triples) * 100
        print(f"  {rel}: {count} ({pct:.1f}%)")
    
    # Sample triples
    print(f"\nSample Triples (first 10):")
    for subj, obj, rel in triples[:10]:
        print(f"  ({subj}, {rel}, {obj})")
    
    return {
        'name': name,
        'total': len(triples),
        'unique_rels': len(relations),
        'action_count': action_count,
        'action_pct': action_count/len(triples)*100 if triples else 0,
        'wikidata_count': wikidata_count,
        'relations': relations
    }


def compare_results(results):
    """Compare results from different methods."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Method':<20} {'Total':<10} {'Relations':<12} {'Action %':<10} {'Wikidata %':<10}")
    print("-"*62)
    
    for r in results:
        if r:
            wikidata_pct = r['wikidata_count'] / r['total'] * 100 if r['total'] else 0
            print(f"{r['name']:<20} {r['total']:<10} {r['unique_rels']:<12} {r['action_pct']:<10.1f} {wikidata_pct:<10.1f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    # Find best for action verbs
    best_action = max(results, key=lambda x: x['action_pct'] if x else 0)
    print(f"\nBest for action verbs: {best_action['name']} ({best_action['action_pct']:.1f}%)")
    
    # Find most diverse
    best_diverse = max(results, key=lambda x: x['unique_rels'] if x else 0)
    print(f"Most diverse relations: {best_diverse['name']} ({best_diverse['unique_rels']} unique)")


def main():
    parser = argparse.ArgumentParser(description="Compare extraction methods")
    parser.add_argument("--rebel", default="/kaggle/working/entity_total_rebel.txt",
                        help="REBEL output file")
    parser.add_argument("--stanford", default="/kaggle/working/entity_total_stanford.txt",
                        help="Stanford OpenIE output file")
    parser.add_argument("--spacy", default="/kaggle/working/entity_total_spacy.txt",
                        help="spaCy output file")
    args = parser.parse_args()
    
    results = []
    
    # Analyze each file if exists
    results.append(analyze_file(args.rebel, "REBEL"))
    results.append(analyze_file(args.stanford, "Stanford OpenIE"))
    results.append(analyze_file(args.spacy, "spaCy"))
    
    # Filter None results
    results = [r for r in results if r]
    
    if len(results) > 1:
        compare_results(results)


if __name__ == "__main__":
    main()
