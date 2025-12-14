import json
import tqdm
from stanfordcorenlp import StanfordCoreNLP

# Cấu hình đường dẫn (Nên đưa ra biến riêng để dễ sửa)
CORENLP_PATH = r'/kaggle/working/stanford-corenlp-4.5.4'
SENTENCE_PATH = '/kaggle/working/msvd_sentence.txt'
ENTITY_REL_PATH = '/kaggle/working/entity_total.txt'

# Khởi tạo NLP (Sử dụng with để đảm bảo đóng resource)
try:
    nlp = StanfordCoreNLP(CORENLP_PATH)
except Exception as e:
    print(f"Lỗi khởi tạo CoreNLP: {e}")
    exit()

print("Bắt đầu xử lý...")

# Dùng 'with' để quản lý file an toàn
with open(SENTENCE_PATH, 'r', encoding='utf-8') as f, \
     open(ENTITY_REL_PATH, 'w', encoding='utf-8') as w:
    
    lines = f.readlines()
    cnt = 0
    
    for line in tqdm.tqdm(lines):
        sentence = line.strip()
        if not sentence: continue

        try:
            output = nlp.annotate(sentence, properties={
                "annotators": "tokenize,lemma,ssplit,pos,depparse,natlog,openie",
                "outputFormat": "json",
                'openie.triple.strict': 'true',
                'openie.max_entailments_per_clause': '1'
            })
            data = json.loads(output)
        except Exception as e:
            print(f"Lỗi xử lý câu: {sentence[:20]}... - {e}")
            continue

        for s_idx, sent_data in enumerate(data['sentences']):
            result = sent_data["openie"]
            tokens = sent_data["tokens"]
            
            for rel in result:
                # Hàm helper để ghép chuỗi từ tokens dựa vào span
                def get_span_text(span_indices, tokens_list):
                    # span_indices là [start, end)
                    words = []
                    for i in range(span_indices[0], span_indices[1]):
                        # Chọn 'lemma' hoặc 'originalText' tùy nhu cầu
                        words.append(tokens_list[i]['lemma']) 
                    return " ".join(words)

                try:
                    # FIX LỖI 1: Dùng hàm join để nối từ đúng thứ tự
                    str_subject = get_span_text(rel['subjectSpan'], tokens)
                    str_object = get_span_text(rel['objectSpan'], tokens)
                    str_relation = get_span_text(rel['relationSpan'], tokens)

                    # Định dạng format output
                    # Lưu ý: Thêm ký tự xuống dòng rõ ràng
                    relationSent = f"{str_subject} & {str_object} & {str_relation}\n"
                    
                    # FIX LỖI 2: Ghi file NGOÀI vòng lặp xử lý từ
                    w.write(relationSent)
                    cnt += 1
                except IndexError:
                    continue

print(f'Tổng số triple trích xuất: {cnt}')
print(f'Kết quả đã ghi vào: {ENTITY_REL_PATH}')

# Đóng connection tới server Java
nlp.close()