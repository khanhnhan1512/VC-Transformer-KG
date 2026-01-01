"""
sen2entity(stan).py - Extract triples using Stanford CoreNLP OpenIE

This extracts ACTION-BASED relations like:
- "A man plays guitar" -> (man, play, guitar)
- "A woman is cooking" -> (woman, cook, ?)

Requires:
- Stanford CoreNLP 4.5.4+ downloaded to CORENLP_PATH
- Java 8+ installed
- stanfordcorenlp Python package: pip install stanfordcorenlp

Usage on Kaggle:
    !wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip
    !unzip stanford-corenlp-4.5.4.zip -d /kaggle/working/
    !python knowledge/sen2entity_stan.py
"""

import json
import argparse
import tqdm
from stanfordcorenlp import StanfordCoreNLP


def parse_args():
    parser = argparse.ArgumentParser(description="Extract triples using Stanford OpenIE")
    parser.add_argument("--corenlp_path", default="/kaggle/working/stanford-corenlp-4.5.4",
                        help="Path to Stanford CoreNLP directory")
    parser.add_argument("--input", default="/kaggle/working/msvd_sentence.txt",
                        help="Input sentences file")
    parser.add_argument("--output", default="/kaggle/working/entity_total.txt",
                        help="Output triples file")
    parser.add_argument("--use_lemma", action="store_true", default=True,
                        help="Use lemmatized form (play instead of playing)")
    return parser.parse_args()


def get_span_text(span_indices, tokens, use_lemma=True):
    """Extract text from token span, optionally lemmatized."""
    words = []
    for i in range(span_indices[0], span_indices[1]):
        if use_lemma:
            words.append(tokens[i]['lemma'])
        else:
            words.append(tokens[i]['originalText'])
    return " ".join(words)


def is_valid_relation(relation):
    """Filter out copula and auxiliary verbs."""
    invalid = {'be', 'is', 'are', 'was', 'were', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did', "'s", "'re"}
    # Check if any word in relation is invalid
    words = relation.lower().split()
    if len(words) == 1 and words[0] in invalid:
        return False
    return True


def main():
    args = parse_args()
    
    print(f"[config] CoreNLP path: {args.corenlp_path}")
    print(f"[config] Input: {args.input}")
    print(f"[config] Output: {args.output}")
    print(f"[config] Use lemma: {args.use_lemma}")
    
    # Initialize CoreNLP
    try:
        nlp = StanfordCoreNLP(args.corenlp_path)
        print("[info] Stanford CoreNLP initialized successfully")
    except Exception as e:
        print(f"[error] Failed to initialize CoreNLP: {e}")
        print("[hint] Make sure Java is installed and CoreNLP path is correct")
        return

    # Statistics
    stats = {
        'total_sentences': 0,
        'total_triples': 0,
        'filtered_invalid': 0,
        'errors': 0
    }
    
    # Process sentences
    with open(args.input, 'r', encoding='utf-8') as f, \
         open(args.output, 'w', encoding='utf-8') as w:
        
        lines = f.readlines()
        stats['total_sentences'] = len(lines)
        
        for line in tqdm.tqdm(lines, desc="Extracting triples"):
            sentence = line.strip()
            if not sentence:
                continue

            try:
                output = nlp.annotate(sentence, properties={
                    "annotators": "tokenize,lemma,ssplit,pos,depparse,natlog,openie",
                    "outputFormat": "json",
                    'openie.triple.strict': 'true',
                    'openie.max_entailments_per_clause': '3'  # Allow more entailments
                })
                data = json.loads(output)
            except Exception as e:
                stats['errors'] += 1
                continue

            for sent_data in data.get('sentences', []):
                openie_results = sent_data.get("openie", [])
                tokens = sent_data.get("tokens", [])
                
                for rel in openie_results:
                    try:
                        str_subject = get_span_text(rel['subjectSpan'], tokens, args.use_lemma)
                        str_object = get_span_text(rel['objectSpan'], tokens, args.use_lemma)
                        str_relation = get_span_text(rel['relationSpan'], tokens, args.use_lemma)
                        
                        # Filter invalid relations
                        if not is_valid_relation(str_relation):
                            stats['filtered_invalid'] += 1
                            continue
                        
                        # Write triple
                        w.write(f"{str_subject} & {str_object} & {str_relation}\n")
                        stats['total_triples'] += 1
                        
                    except (IndexError, KeyError):
                        continue
    
    # Close CoreNLP
    nlp.close()
    
    # Print statistics
    print("\n" + "="*50)
    print("EXTRACTION STATISTICS")
    print("="*50)
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total triples extracted: {stats['total_triples']}")
    print(f"Filtered (invalid relations): {stats['filtered_invalid']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()