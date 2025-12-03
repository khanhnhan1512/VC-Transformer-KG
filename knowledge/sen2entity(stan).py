import spacy
from spacy import displacy
from stanfordcorenlp import StanfordCoreNLP
import json
import tqdm

nlp = StanfordCoreNLP(r'/kaggle/working/stanford-corenlp-4.5.4')
sentence_path = '/kaggle/working/msvd_sentence.txt'
entity_rel_path = '/kaggle/working/entity_total.txt'
# doc = nlp("The 22-year-old recently won ATP Challenger tournament.")


f = open(sentence_path, 'r', encoding='utf-8')
w = open(entity_rel_path, 'w', encoding='utf-8')
lines = f.readlines()
cnt = 0
for line in tqdm.tqdm(lines):
    sentence = line
    output = nlp.annotate(sentence, properties={"annotators": "tokenize,lemma,ssplit,pos,depparse,natlog,openie",
                                                "outputFormat": "json",
                                                'openie.triple.strict': 'true',
                                                'openie.max_entailments_per_clause': '1'
                                                })
    data = json.loads(output)
    # result = data['sentences'][0]['openie']
    # print(result)
    # print(data['sentences'][0].keys())
    # print(data['sentences'][0]['openie'])
    # print(data['sentences'][0]['tokens'])

    for i in range(len(data['sentences'])):
        result = data["sentences"][i]["openie"]  # Trực tiếp lấy openie của sentence i
        lemmas = data["sentences"][i]["tokens"]  # Trực tiếp lấy tokens của sentence i
        cnt += 1
        
        for rel in result:
            l_relation, l_object, l_subject = '', '', ''
            span = str(rel['subjectSpan']), str(rel['objectSpan']), str(rel['relationSpan'])
            
            # Lấy lemma của subject
            l_subject1 = lemmas[rel['subjectSpan'][0]:rel['subjectSpan'][1]]
            for h in range(len(l_subject1)):
                l_subject = l_subject1[h]['lemma'] + ' ' + l_subject
            
            # Lấy lemma của object
            l_object1 = lemmas[rel['objectSpan'][0]:rel['objectSpan'][1]]
            for s in range(len(l_object1)):
                l_object = l_object1[s]['lemma'] + ' ' + l_object
            
            # Lấy lemma của relation
            l_relation1 = lemmas[rel['relationSpan'][0]:rel['relationSpan'][1]]
            for j in range(len(l_relation1)):
                l_relation = l_relation1[j]['lemma'] + ' ' + l_relation
                # l_relation = lemmas[i][rel['relationSpan'][0]:rel['relationSpan'][1]][0][
                #              0:rel['relationSpan'][1] - rel['relationSpan'][0]]['lemma']
                # relationSent1 = rel['subject'], rel['object'], rel['relation']
                relationSent = l_subject, '&', l_object, '&', l_relation
                # print(relationSent)
                w.writelines(relationSent)
                w.writelines('\n')

                # print(str(cnt))
                # print(relationSent1)
print('total number is ' + str(cnt) + '\n')
print('reslut write to ' + entity_rel_path)
w.close()
f.close()

# sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))  # 语法树
# print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法
# nlp.close()  # Do not forget to close! The backend server will consume a lot memery

# lines = f.readlines()
# for line in lines:
#     # print(line)
#     # print(type(line))
#     doc = nlp(line)
# displacy.render(doc, style='dep', jupyter=False)  # need to run in Jupyter

# break
# for tok in doc:
#     print(tok.text + "---------------->" + tok.pos_ + ' ' + tok.dep_ + ' ' + tok.tag_)
#     if tok.pos_ == 'NOUN':
#         w.writelines(tok.text + ' ')
#     # if tok.dep_ == 'pobj' or 'dobj':
#     #     w.writelines(tok.text + ' ')
#     if tok.pos_ == 'VERB':
#         w.writelines(tok.lemma_ + ' ')
#     if tok.pos_ == 'SPACE':
#         w.write('\n')
# print(tok.text + '------------>' + tok.dep_)

# for tok in doc:
#     print(tok.text, "...", tok.dep_)