'''
Train a GPT Model
=================
'''

import torch
import sentencepiece as sp
import os.path
from model import ConfigGPT, GPT


# Hyperparams
batch_size = 256  # sequences evaluated in parallel
max_iters = 5
learning_rate = 2.5e-4
eval_losses = 200
eval_train = 500
write_text = True
text_log = []
start_txt = 'Mati je bila '
in_file = 'data/cankar-proza.txt'
out_file = 'out.txt'


def print_me(text):
    text_log.append(text)
    print(text)


# Instantiate default config
print_me('GPT Training')
config = ConfigGPT()
print(repr(config))

# Get text
with open(in_file) as f:
    text = f.read()

# Load tokenisation model
if not os.path.isfile('sp-tokens.model'):
    # Train sentencepiece model on input data
    sp.SentencePieceTrainer.train(f'--input={in_file} \
                                    --model_prefix=sp-tokens \
                                    --vocab_size={config.vocab_size}')

token_model = sp.SentencePieceProcessor()
token_model.load('sp-tokens.model')

# Train and test splits
data = torch.tensor(token_model.encode_as_ids(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


#  data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.seq_size, (batch_size,))
    x = torch.stack([data[i:i+config.seq_size] for i in ix])
    y = torch.stack([data[i+1:i+config.seq_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


#  evaluate loss on multiple runs
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_losses)
        for k in range(eval_losses):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# instantiate model
model = GPT(config)
m = model.to(config.device)

# print num of parameters
no_params = sum(p.numel() for p in m.parameters())
print_me(f'Model using {no_params/1e6:.2f}M paramaters.')
print_me(f'Training on {len(data)/1e6:.2f}M tokens.')
print_me('---------------')

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # print loss
    if iter % eval_train == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print_me(f"Iter: {iter}, Train Loss: {losses['train']:.4f}, " +
                 f"Validation Loss: {losses['val']:.4f}")

    # get new sampled data
    x, y = get_batch('train')

    # evaluate the loss
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 

# generate new text from model
print_me('---------------')
print_me('Sample output:')
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
start_ids = token_model.encode_as_ids(start_txt)
context = torch.tensor(start_ids, dtype=torch.long,
                       device=config.device).unsqueeze(0)

text = token_model.decode_ids(
    m.generate(context, max_new_tokens=10000)[0].tolist())
print(text[:500])
if write_text:
    text = '\n'.join(text_log) + '\n' + text
    open(out_file, 'w').write(text)
