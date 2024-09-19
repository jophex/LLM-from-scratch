import os
import lzma
from tqdm import tqdm


def xyz_file_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xz') and os.path.join(directory, filename):
            files.append(filename)
    return files


folder_path = "openwebtext"
output_train_file = "output_train.txt"
output_val_file = "output_val.txt"
vocab_file = "vocab.txt"
# split_files = int(input("how many files would you like to split into?"))

files = xyz_file_in_directory(folder_path)
total_files = len(files)

splite_index = int(total_files * 0.9)
file_train = files[:splite_index]
file_val = files[splite_index:]
# print(splite_index)


# max_count = total_files // split_files if split_files != 0 else total_files


vocab = set()

# for i in range(split_files):
#     with open(output_file.format(i), "w", encoding="utf-8") as outfile:
#         for count, filename in enumerate(tqdm(files[:max_count], total=max_count)):
#             if count >= max_count:
#                 break
#             file_path = os.path.join(folder_path, filename)
#             with lzma.open(file_path, "rt", encoding="utf-8") as infile:
#                 texts = infile.read()
#                 outfile.write(texts)
#                 characters = set(texts)
#                 vocab.update(characters)
#         files = files[max_count:]


with open(output_train_file, 'w', encoding='utf-8') as outtrainfile:
    for filename in tqdm(file_train, total=len(file_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            texts = infile.read()
            outtrainfile.write(texts)
            characters = set(texts)
            vocab.update(characters)

with open(output_val_file, "w", encoding="utf-8") as outvalfile:
    for filename in tqdm(file_val, total=len(file_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            texts = infile.read()
            outvalfile.write(texts)
            characters = set(texts)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
