"""
sen2entity_spacy.py - Extract triples using spaCy dependency parsing

This extracts ACTION-BASED relations by finding:
- Subject (nsubj) -> Verb (ROOT) -> Object (dobj/pobj)

Advantages over REBEL:
- Extracts actual action verbs (play, ride, cook) not Wikidata relations
- Faster than Stanford CoreNLP
- No Java dependency

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm  # or en_core_web_lg for better accuracy

Usage:
    python knowledge/sen2entity_spacy.py --input msvd_sentence.txt --output entity_total.txt
"""

import argparse
import spacy
from collections import Counter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract triples using spaCy")
    parser.add_argument("--input", default="/kaggle/working/msvd_sentence.txt",
                        help="Input sentences file")
    parser.add_argument("--output", default="/kaggle/working/entity_total.txt",
                        help="Output triples file")
    parser.add_argument("--model", default="en_core_web_sm",
                        help="spaCy model to use (en_core_web_sm, en_core_web_lg)")
    parser.add_argument("--use_lemma", action="store_true", default=True,
                        help="Use lemmatized form (play instead of playing)")
    parser.add_argument("--include_prep", action="store_true", default=True,
                        help="Include prepositional objects (e.g., 'walk in park')")
    return parser.parse_args()


def get_noun_chunk(token):
    """Get the full noun phrase containing the token."""
    # Try to find noun chunk containing this token
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token.text


def get_subject(verb):
    """Find subject of a verb."""
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    # Check for subject in parent clause
    if verb.head and verb.head != verb:
        for child in verb.head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
    return None


def get_objects(verb):
    """Find direct and prepositional objects of a verb."""
    objects = []
    
    for child in verb.children:
        # Direct object
        if child.dep_ == "dobj":
            objects.append((child, None))
        
        # Prepositional object: verb -> prep -> pobj
        elif child.dep_ == "prep":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    objects.append((pobj, child.text))  # (object, preposition)
        
        # Particle + object (phrasal verbs)
        elif child.dep_ == "prt":
            # Look for object after particle
            for sibling in verb.children:
                if sibling.dep_ == "dobj" and sibling not in [o[0] for o in objects]:
                    objects.append((sibling, child.text))
    
    return objects


def is_valid_verb(token):
    """Check if token is a valid action verb."""
    invalid_lemmas = {'be', 'have', 'do', 'get', 'make', 'take', 'go', 'come', 
                      'see', 'seem', 'become', 'feel', 'look', 'appear', 'sound'}
    
    # Must be a verb
    if token.pos_ != "VERB":
        return False
    
    # Filter auxiliaries
    if token.dep_ in ("aux", "auxpass"):
        return False
    
    # Filter copula (but keep action verbs)
    # Note: We keep common invalid verbs for now, can filter later
    # if token.lemma_.lower() in invalid_lemmas:
    #     return False
    
    return True


def extract_triples_from_doc(doc, use_lemma=True, include_prep=True):
    """Extract (subject, relation, object) triples from spaCy doc."""
    triples = []
    
    for token in doc:
        # Find verbs that are ROOT or have subjects
        if not is_valid_verb(token):
            continue
        
        # Get subject
        subj = get_subject(token)
        if subj is None:
            continue
        
        # Get objects
        objects = get_objects(token)
        if not objects and not include_prep:
            continue
        
        # Create triples
        subj_text = subj.lemma_ if use_lemma else subj.text
        verb_text = token.lemma_ if use_lemma else token.text
        
        # If we have direct objects
        for obj, prep in objects:
            obj_text = obj.lemma_ if use_lemma else obj.text
            
            if prep and include_prep:
                # Include preposition in relation: "walk in"
                relation = f"{verb_text} {prep}"
            else:
                relation = verb_text
            
            triples.append((subj_text, obj_text, relation))
        
        # Also try to find objects in prepositional phrases if no direct object
        if not objects and include_prep:
            for child in token.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj_text = pobj.lemma_ if use_lemma else pobj.text
                            relation = f"{verb_text} {child.text}"
                            triples.append((subj_text, obj_text, relation))
    
    return triples


def extract_triples_svo(doc, use_lemma=True):
    """
    Alternative: Simple SVO extraction.
    Finds Subject-Verb-Object patterns more aggressively.
    """
    triples = []
    
    for token in doc:
        if token.pos_ != "VERB":
            continue
        
        # Find subject
        subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
        # Find objects (direct and indirect)
        objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "pobj")]
        
        # Also check prep -> pobj
        for child in token.children:
            if child.dep_ == "prep":
                objects.extend([c for c in child.children if c.dep_ == "pobj"])
        
        verb = token.lemma_ if use_lemma else token.text
        
        for subj in subjects:
            subj_text = subj.lemma_ if use_lemma else subj.text
            
            if objects:
                for obj in objects:
                    obj_text = obj.lemma_ if use_lemma else obj.text
                    triples.append((subj_text, obj_text, verb))
            else:
                # No object, but still record subject-verb
                # Useful for intransitive verbs: "man walks", "dog runs"
                pass
    
    return triples


def is_valid_relation(relation):
    """Filter out copula and auxiliary verbs."""
    invalid = {'be', 'is', 'are', 'was', 'were', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did'}
    # Get first word (verb) of relation
    verb = relation.split()[0].lower()
    return verb not in invalid


def main():
    args = parse_args()
    
    print(f"[config] Model: {args.model}")
    print(f"[config] Input: {args.input}")
    print(f"[config] Output: {args.output}")
    print(f"[config] Use lemma: {args.use_lemma}")
    print(f"[config] Include prep: {args.include_prep}")
    
    # Load spaCy model
    print(f"[info] Loading spaCy model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"[error] Model {args.model} not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", args.model])
        nlp = spacy.load(args.model)
    
    print(f"[info] Model loaded successfully")
    
    # Read input sentences
    with open(args.input, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"[info] Processing {len(sentences)} sentences...")
    
    # Statistics
    stats = {
        'total_sentences': len(sentences),
        'total_triples': 0,
        'filtered_invalid': 0,
        'unique_relations': Counter()
    }
    
    # Process and extract
    with open(args.output, 'w', encoding='utf-8') as w:
        # Process in batches for efficiency
        batch_size = 1000
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing"):
            batch = sentences[i:i+batch_size]
            docs = list(nlp.pipe(batch, disable=["ner"]))  # Disable NER for speed
            
            for doc in docs:
                # Try dependency-based extraction
                triples = extract_triples_from_doc(doc, args.use_lemma, args.include_prep)
                
                # Fallback to simple SVO if no triples found
                if not triples:
                    triples = extract_triples_svo(doc, args.use_lemma)
                
                for subj, obj, rel in triples:
                    # Filter invalid relations
                    if not is_valid_relation(rel):
                        stats['filtered_invalid'] += 1
                        continue
                    
                    # Write triple in standard format
                    w.write(f"{subj} & {obj} & {rel}\n")
                    stats['total_triples'] += 1
                    stats['unique_relations'][rel] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTRACTION STATISTICS")
    print("="*60)
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total triples extracted: {stats['total_triples']}")
    print(f"Filtered (invalid relations): {stats['filtered_invalid']}")
    print(f"Unique relations: {len(stats['unique_relations'])}")
    
    print(f"\nTop 30 relations:")
    for rel, count in stats['unique_relations'].most_common(30):
        print(f"  {rel}: {count}")
    
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
