from main import BigramLanguageModel, vocab_size, encode, decode
import pickle
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel(vocab_size)
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
