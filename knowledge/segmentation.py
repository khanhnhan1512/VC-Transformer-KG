import os
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('/kaggle/working/OPENKE_file'):
    os.makedirs('/kaggle/working/OPENKE_file')


def ReadFileDatas(original_filename):
    with open(original_filename, 'r', encoding='utf-8') as file:
        FileNameList = file.readlines()
    random.shuffle(FileNameList)
    print(f"Number of dataset: {len(FileNameList)}")
    return FileNameList


def TrainValTestFile(FileNameList, train_ratio=0.6, val_ratio=0.2):
    """
    Split data into train/val/test sets.
    Default: 60% train, 20% val, 20% test
    """
    total = len(FileNameList)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    l_train = FileNameList[:train_end]
    l_val = FileNameList[train_end:val_end]
    l_test = FileNameList[val_end:]
    
    print(f"Total: {total}, Train: {len(l_train)}, Val: {len(l_val)}, Test: {len(l_test)}")
    return l_train, l_val, l_test


def WriteDatasToFile(listInfo, new_filename):
    """
    Write triples to file with count header (OpenKE format).
    Format: First line is count, then each line is a triple.
    """
    with open(new_filename, 'w', encoding='utf-8') as file_handle:
        # Write count header (required by OpenKE)
        file_handle.write(f"{len(listInfo)}\n")
        for str_Result in listInfo:
            # Ensure line doesn't already have newline at end
            str_Result = str_Result.strip()
            if str_Result:
                file_handle.write(str_Result + '\n')
    print(f'Write {new_filename} success! ({len(listInfo)} triples)')


if __name__ == "__main__":
    # listFileInfo = ReadFileDatas('../MSR-VTT/OPENKE_file/msrvtt/b.txt')
    listFileInfo = ReadFileDatas('/kaggle/working/OPENKE_file/total_id.txt')
    l_train, l_val, l_test = TrainValTestFile(listFileInfo)
    WriteDatasToFile(l_train, '/kaggle/working/OPENKE_file/train2id.txt')
    WriteDatasToFile(l_val, '/kaggle/working/OPENKE_file/valid2id.txt')
    WriteDatasToFile(l_test, '/kaggle/working/OPENKE_file/test2id.txt')