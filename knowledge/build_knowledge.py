import os
import re
from tqdm import tqdm

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('/kaggle/working/OPENKE_file'):
    os.makedirs('/kaggle/working/OPENKE_file')

# Định nghĩa đường dẫn file
# Input: file quy tắc induction (cần upload vào /kaggle/working/OPENKE_file/ trước khi chạy)
rel_induction = '/kaggle/working/OPENKE_file/rel.txt'  # quy tắc ánh xạ relation
ent_induction = '/kaggle/working/OPENKE_file/ent.txt'  # quy tắc ánh xạ entity

# Input: file triple từ bước sen2entity
target_entity = '/kaggle/working/entity_total.txt'  # file input chứa triple (output từ sen2entity)

# Input: file ánh xạ ID (output từ build_knowledgeid.py)
entity2id = '/kaggle/working/OPENKE_file/entity2id.txt'  # file nội dung ánh xạ entity sang ID
relation2id = '/kaggle/working/OPENKE_file/relation2id.txt'  # file nội dung ánh xạ relation sang ID

# Output: file kết quả
total_word = '/kaggle/working/OPENKE_file/total_word.txt'  # file tổng hợp entity và relation dạng text
total_id = '/kaggle/working/OPENKE_file/total_id.txt'  # file tổng hợp entity và relation đã được ánh xạ sang ID
# test = '../MSR-VTT/OPENKE_file/msrvtt/test.txt'
# valid = '../MSR-VTT/OPENKE_file/msrvtt/valid.txt'

# Đọc file input với context manager
with open(rel_induction, 'r', encoding='utf-8') as f:
    rel_induction_lines = f.readlines()  # vd: drive|driving#drives#ride#drive\n

with open(ent_induction, 'r', encoding='utf-8') as f:
    ent_induction_lines = f.readlines()  # vd: bicycle|bike#cycle#pushbike\n

with open(target_entity, 'r', encoding='utf-8') as f:
    target_lines = f.readlines()


def word_boundary_match(pattern, text):
    """
    Kiểm tra xem pattern có xuất hiện như một từ hoàn chỉnh trong text không.
    Sử dụng word boundary (\\b) để tránh substring matching sai.
    Ví dụ: 'car' sẽ KHÔNG match với 'cartoon' hay 'scare'
    """
    if not pattern or not text:
        return False
    # Escape special regex characters trong pattern
    escaped_pattern = re.escape(pattern)
    # Tìm kiếm với word boundary
    return bool(re.search(r'\b' + escaped_pattern + r'\b', text, re.IGNORECASE))


def find_best_match(text, induction_lines, match_type='entity'):
    """
    Tìm normalized form tốt nhất cho text dựa trên induction rules.
    
    Chiến lược matching (theo thứ tự ưu tiên):
    1. Exact match: text khớp chính xác với một variant
    2. Word boundary match: variant xuất hiện như một từ hoàn chỉnh trong text
    
    Args:
        text: entity hoặc relation cần normalize
        induction_lines: danh sách các rule từ ent.txt hoặc rel.txt
        match_type: 'entity' hoặc 'relation' (để debug)
    
    Returns:
        normalized form nếu tìm thấy, ngược lại 'none'
    """
    text = text.strip().lower()
    
    # Bước 1: Tìm exact match trước (ưu tiên cao nhất)
    for line in induction_lines:
        if '|' not in line:
            continue
        step1 = line.split('|')
        normalized = step1[0].replace(' ', '').strip()
        variants = step1[1].replace('\n', '').split('#')
        
        for variant in variants:
            variant = variant.strip().lower()
            if variant and variant == text:
                return normalized
    
    # Bước 2: Tìm word boundary match (tránh substring matching sai)
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
            
            # Word boundary match - variant phải xuất hiện như một từ hoàn chỉnh
            if word_boundary_match(variant, text):
                # Ưu tiên match dài hơn ("young woman" > "woman")
                if len(variant) > best_match_len:
                    best_match = normalized
                    best_match_len = len(variant)
    
    return best_match if best_match else 'none'


############
# Chuyển triple từ dạng text sang dạng entity và relation
# Quy trình:
# 1. Đọc các file quy tắc và dữ liệu đầu vào
# 2. Với mỗi dòng trong entity_total.txt, tách theo ký tự & thành 3 phần: [entity1, entity2, relation]
# 3. Xử lý entity (index 0, 1):
#     - So khớp với các quy tắc trong ent.txt (exact match trước, sau đó word boundary match)
#     - Tìm entity chuẩn hóa tương ứng
#     - Ghi vào file total_word.txt hoặc none nếu không tìm thấy
# 4. Xử lý relation (index 2):
#     - So khớp với các quy tắc trong rel.txt (exact match trước, sau đó word boundary match)
#     - Tìm relation chuẩn hóa tương ứng
#     - Ghi vào file total_word.txt hoặc none nếu không tìm thấy

with open(total_word, 'w', encoding='utf-8') as total_word_f:
    for target_line in tqdm(target_lines, desc="Processing triples"):
        parts = target_line.split('&')  # triple split by &: [entity1, entity2, relation]
        
        if len(parts) != 3:
            # Skip malformed lines
            continue
        
        entity1 = parts[0].strip()
        entity2 = parts[1].strip()
        relation = parts[2].strip()
        
        # Normalize entity1
        entity1_normalized = find_best_match(entity1, ent_induction_lines, 'entity')
        total_word_f.write(entity1_normalized + ',')
        
        # Normalize entity2
        entity2_normalized = find_best_match(entity2, ent_induction_lines, 'entity')
        total_word_f.write(entity2_normalized + ',')
        
        # Normalize relation
        relation_normalized = find_best_match(relation, rel_induction_lines, 'relation')
        total_word_f.write(relation_normalized + '\n')

print("Phase 1 completed: total_word.txt generated")  # vd: entity1,entity2,relation


############
# Chuyển total.txt dạng text sang dạng id: total_word.txt → total_id.txt
# Quy trình:
# 1. Đọc file ánh xạ entity2id.txt và relation2id.txt vào dictionary
# 2. Đọc file total_word.txt (kết quả phần 1)
# 3. Với mỗi triple dạng text:
#     - Bỏ qua nếu có phần tử none
#     - Tra cứu ID từ dictionary
#     - Kiểm tra ràng buộc ID hợp lệ
#     - Ghi vào total_id.txt theo định dạng: entity1_id entity2_id relation_id


def load_id_mapping(filepath):
    """Load entity2id hoặc relation2id file vào dictionary."""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Dòng đầu tiên có thể là count (số nguyên duy nhất)
            if '\t' not in line:
                # Skip header line (count)
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                name, id_str = parts
                mapping[name] = int(id_str)
    return mapping


# Load mappings
reldict = load_id_mapping(relation2id)
entdict = load_id_mapping(entity2id)

print(f"Loaded {len(entdict)} entities and {len(reldict)} relations")
print(f"Sample entities: {list(entdict.items())[:5]}")
print(f"Sample relations: {list(reldict.items())[:5]}")

# Tính toán giới hạn ID động dựa trên số lượng entities/relations thực tế
max_entity_id = len(entdict) - 1
max_relation_id = len(reldict) - 1
print(f"Max entity ID: {max_entity_id}, Max relation ID: {max_relation_id}")

# Đọc total_word.txt và chuyển sang ID
valid_triples = 0
skipped_none = 0
skipped_missing = 0

with open(total_word, 'r', encoding='utf-8') as f_in, open(total_id, 'w', encoding='utf-8') as f_out:
    total_word_lines = f_in.readlines()
    
    for line in tqdm(total_word_lines, desc="Converting to IDs"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) != 3:
            continue
            
        entity1, entity2, relation = parts[0].strip(), parts[1].strip(), parts[2].strip()
        
        # Bỏ qua nếu có phần tử none
        if entity1 == 'none' or entity2 == 'none' or relation == 'none':
            skipped_none += 1
            continue
        
        # Kiểm tra xem entity/relation có trong mapping không
        if entity1 not in entdict:
            skipped_missing += 1
            continue
        if entity2 not in entdict:
            skipped_missing += 1
            continue
        if relation not in reldict:
            skipped_missing += 1
            continue
        
        # Lấy ID
        ent1_id = entdict[entity1]
        ent2_id = entdict[entity2]
        rel_id = reldict[relation]
        
        # Ghi vào file (format: entity1_id entity2_id relation_id)
        f_out.write(f"{ent1_id} {ent2_id} {rel_id}\n")
        valid_triples += 1

print(f"\nPhase 2 completed: total_id.txt generated")
print(f"Valid triples: {valid_triples}")
print(f"Skipped (none): {skipped_none}")
print(f"Skipped (missing in mapping): {skipped_missing}")