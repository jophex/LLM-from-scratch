import mmap
import os
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device = 'cpu'

block_size = 64  # 8 tokens
batch_size = 128
learning_rate = 1e-4
max_iters = 30000000
eval_iter = 100
drop_out = 0.2  # so that we don't overfit
n_embd = 384
n_layer = 4  # numbers of layers in the decoder  4 blocks in the decoder layer
n_head = 4
eval_interval = 100


#
char = ""
with open('vocab.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)
print(vocab_size)

# character level tokenizer

# def clean_text(text):
#     # Remove lines that don't contain the main text content
#     cleaned_lines = []
#     for line in text.split('\n'):
#         if not line.startswith(("0", "Port-au-Prince", "CNN")) and len(line) > 0:
#             cleaned_lines.append(line)
#     return '\n'.join(cleaned_lines)
#
#
# with open('vocab.txt', 'r', encoding='utf-8') as file:
#     text = file.read()
#
# # Clean the text
# cleaned_text = clean_text(text)
#
# # Proceed with the rest of your preprocessing using cleaned_text
# chars = sorted(list(set(cleaned_text)))
#
# vocab_size = len(chars)
# print(vocab_size)

string_to_text = {ch: i for i, ch in enumerate(chars)}
int_to_strint = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_text[c] for c in s]
decode = lambda l: ''.join([int_to_strint[i] for i in l])  # Updated line

# encode_hello = encode('hello')
# decode_hello = decode(encode_hello)
# print(decode_hello)

data = torch.tensor(encode(text), dtype=torch.long)  # converts the text that's read to tensors by encoding it

# print(data)

n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "output_train.txt" if split == 'train' else "output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     # print(ix)   #getting batches of data
#
#     x = torch.stack([data[i:i + block_size] for i in ix])
#     y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
#     x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
#     return x, y


x, y = get_batch('train')
print('input', x)
print('target', y)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # self.key = nn.Linear(n_embd, head_size, bias=False)
        # self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.query = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # self.key = nn.Linear(n_embd, head_size, bias=False)
        # self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.query = nn.Linear(n_embd, head_size, bias=False)
        # self.dropout = nn.Dropout(drop_out)
        #
        # self.dropout = nn.Dropout(drop_out)

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # B, T, C = x.shape
        # k = self.key(x)
        # q = self.query(x)
        #
        # wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # wei = F.softmax(wei, dim=-1)  # Corrected dimension for softmax
        # wei = self.dropout(wei)
        #
        # v = self.value(x)
        # out = wei @ v
        # return out

        # missmatch in the dimension expected by the tril which is constant 64 but changes to 65 after mismatch in training and T is random so changed it to be constant 64

        # B, T, C = x.shape
        # k = self.key(x)
        # q = self.query(x)
        #
        # wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        #
        # # Create tril matrix dynamically based on T
        # tril = torch.tril(torch.ones(T, T, device=x.device))
        # wei = wei.masked_fill(tril == 0, float('-inf'))
        #
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        #
        # v = self.value(x)
        # out = wei @ v
        # return out

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads,
                              n_embd)  # adding another learnable  parameters to help out network learn more about the text
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # out = torch.cat([h(x) for h in self.head], dim=-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop_out)

        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(
            n_layer)])  # four layer of four decoders  values of decoders available in the model this is the decoder layer

        self.ln_f = nn.LayerNorm(
            n_embd)  # final layer normalization  taken after the decoder layer at the end converging the model
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, target=None):  # used for customization debugging optimization, understanding the process
        B, T = index.shape

        # index and targets are both B, T tensor of integer
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = target.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # def generate(self, index, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         logits, loss = self.forward(index)
    #         logits = logits[:, -1, :]
    #         probs = F.softmax(logits, dim=1)
    #
    #         print(f"logits shape: {logits.shape}")
    #         print(f"probs shape: {probs.shape}")
    #
    #         index_next = torch.multinomial(probs, num_samples=1)
    #         index = torch.cat((index, index_next), dim=1)
    #
    #     return index

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


model = BigramLanguageModel(vocab_size)
m = model.to(device)


# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated_char = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_char)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


optimization = torch.optim.AdamW(model.parameters(), learning_rate)

for iters in range(max_iters):
    if iters % eval_iter == 0:
        losses = estimate_loss()
        print(f"steps: , {iters}, train loss, {losses['train']:.3f}, val loss {losses['val']:.3f}")

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimization.zero_grad(set_to_none=True)
    loss.backward()
    optimization.step()

    print('loss', loss.item())

with open("model-01.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved")

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated_char = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# # print(generated_char)

# prompt = 'hello can you see me'
# context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
# generated_char = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
# print('generated prompt ', generated_char)
