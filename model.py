import torch
import torch.nn as nn
import sentencepiece as sp


torch.manual_seed(1000)

# Hyperparams
batch_size = 256  # sequences evaluated in parallel
block_size = 32  # time domain for predictions
max_iters = 50000
learning_rate = 2.5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_losses = 200
eval_train = 500
n_embed = 180
n_head = 5
n_layer = 3
dropout = 0.3
write_text = True
text_log = []
start_txt = 'Mati je bila '


def print_me(text):
    text_log.append(text)
    print(text)


print_me('GPT Training')
print_me(f'Using {device} device for processing.')
print_me('---------------')
print_me(f'Batch Size: {batch_size}')
print_me(f'Block Size: {block_size}')
print_me(f'Num Embeddings: {n_embed}')
print_me(f'Num Heads: {n_head}')
print_me(f'Num Layers: {n_layer}')
print_me(f'Learning rate: {learning_rate}')
print_me(f'Train Iter: {max_iters}')
print_me('---------------')

# Get Cankar's prose
with open('data/cankar/cankar-proza.txt') as f:
    text = f.read()

# Load tokenisation model
token_model = sp.SentencePieceProcessor()
token_model.load('cankar-tokens.model')
vocab_size = token_model.vocab_size()

# Train and test splits
data = torch.tensor(token_model.encode_as_ids(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


#  data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
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


class Head(nn.Module):
    '''Implementation of self-attention head.'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # register buffer to be automatically transferred to gpu
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # input (batch, time step, n_emb)
        # output (batch, time step, head size)
        B, T, C = input.shape
        k = self.key(input)    # (B, T, head size)
        q = self.query(input)  # (B, T, head size)
        # compute attention using dot product
        weights = q @ k.transpose(-2, -1)  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        # normalise variance before softmax
        weights = weights * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # perform weighted aggregation of values
        v = self.value(input)  # (B, T, head size)
        out = weights @ v  # (B, T, T) @ (B, T, head size) -> (B, T, head size)
        return out


class MultiHeadAttention(nn.Module):
    '''Implementation of multiple heads of self attention.'''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, input):
        return torch.cat([h(input) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    '''Implements multi layer perceptron after attention.'''

    def __init__(self, n_embed):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    '''Combines multi-head attention with MLP computation.'''

    def __init__(self, n_embed, num_head):
        super().__init__()
        head_size = n_embed // num_head
        self.self_att = MultiHeadAttention(num_head, head_size)
        self.feed_fwd = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_att(self.norm1(x))
        x = x + self.feed_fwd(self.norm2(x))
        return x


class GPT(nn.Module):
    '''Implement bigram model, predicting next token.'''

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head) for _ in range(n_layer)])
        self.norm_last = nn.LayerNorm(n_embed)
        self.linear_out = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_params)

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.norm_last(x)  # (B, T, C)
        logits = self.linear_out(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# instantiate model
model = GPT()
m = model.to(device)

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
context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

text = token_model.decode_ids(
    m.generate(context, max_new_tokens=10000)[0].tolist())
print(text[:500])
if write_text:
    text = '\n'.join(text_log) + '\n' + text
    open('more.txt', 'w').write(text)
