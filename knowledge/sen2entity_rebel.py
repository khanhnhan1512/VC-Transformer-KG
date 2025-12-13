from transformers import pipeline
import json
import tqdm

# Load the REBEL model
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

sentence_path = '/kaggle/working/msvd_sentence.txt'
entity_rel_path = '/kaggle/working/entity_total.txt'

# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets

f = open(sentence_path, 'r', encoding='utf-8')
w = open(entity_rel_path, 'w', encoding='utf-8')
lines = f.readlines()
cnt = 0

for line in tqdm.tqdm(lines):
    sentence = line.strip()
    if not sentence:
        continue
    
    # Extract triplets using REBEL
    extracted_text = triplet_extractor.tokenizer.batch_decode([
        triplet_extractor(sentence, return_tensors=True, return_text=False)[0]["generated_token_ids"]
    ])
    
    # Parse the extracted text to get triplets
    extracted_triplets = extract_triplets(extracted_text[0])
    
    for triplet in extracted_triplets:
        head = triplet['head']
        relation = triplet['type']
        tail = triplet['tail']
        
        # Write the triplet to the output file
        w.writelines(f"{head} & {tail} & {relation}\n")
    
    cnt += 1

print('total number of sentences processed: ' + str(cnt))
print('result written to ' + entity_rel_path)

f.close()
w.close()