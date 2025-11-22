import os

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('/kaggle/working/OPENKE_file'):
    os.makedirs('/kaggle/working/OPENKE_file')

# Input: file quy tắc induction (cần upload vào /kaggle/working/OPENKE_file/ trước khi chạy)
rel_induction = '/kaggle/working/OPENKE_file/rel.txt'
ent_induction = '/kaggle/working/OPENKE_file/ent.txt'

# Output: file ánh xạ ID
entity2id = '/kaggle/working/OPENKE_file/entity2id.txt'
relation2id = '/kaggle/working/OPENKE_file/relation2id.txt'

rel_induction_file = open(rel_induction, 'r')
ent_induction_file = open(ent_induction, 'r')
relation2id_file = open(relation2id, 'w')
entity2id_file = open(entity2id, 'w')

rel_induction_lines = rel_induction_file.readlines() # vd: drive|driving#drives#ride#drive\n
ent_induction_lines = ent_induction_file.readlines() # vd: bicycle|bike#cycle#pushbike\n

cnt = 0
for line in rel_induction_lines:
    rel = line.split('|')[0] # ['drive']
    relation2id_file.writelines(rel.replace(' ', '') + '\t' + str(cnt) + '\n')
    cnt += 1
relation2id_file.close()

cnt = 0
for line in ent_induction_lines:
    ent = line.split('|')[0] # ['bicycle']
    entity2id_file.writelines(ent.replace(' ', '') + '\t' + str(cnt) + '\n')
    cnt += 1
entity2id_file.close()
rel_induction_file.close()
ent_induction_file.close()