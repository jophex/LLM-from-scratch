import torch


# string_to_text = {ch: i for i, ch in enumerate(chars)}
# int_to_strint = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [string_to_text[c] for c in s]
# decode = lambda l: ''.join([int_to_strint[i] for i in l])


# data = torch.tensor(encode(cleaned_text), dtype=torch.long)


def clean_text(text):
    # Remove lines that don't contain the main text content
    cleaned_lines = []
    for line in text.split('\n'):
        if not line.startswith(("0", "Port-au-Prince", "CNN")) and len(line) > 0:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


clean_text('output_train.txt')
