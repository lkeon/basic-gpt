'''
Generative Pretrained Transformer (GPT)
=======================================

References:
1) Original paper:
   https://arxiv.org/abs/1706.03762
2) Andrej Karpathy's implementation:
   https://github.com/karpathy/nanoGPT/blob/master/model.py
3) Andrej's YouTube lectures:
   https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4749s
'''

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ConfigGPT:
    seq_size:   int = 64
    vocab_size: int = 1000
    num_layer:  int = 6
    num_head:   int = 6
    num_embed:  int = 180
    dropout:    float = 0.0
    device:     str = 'cpu'

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    '''Implementation of self-attention head.'''

    def __init__(self, config):
        super().__init__()
        head_size = config.num_embed // config.num_head
        self.key = nn.Linear(config.num_embed, head_size, bias=False)
        self.query = nn.Linear(config.num_embed, head_size, bias=False)
        self.value = nn.Linear(config.num_embed, head_size, bias=False)
        # register buffer to be automatically transferred to gpu
        self.register_buffer('tril', torch.tril(
            torch.ones(config.seq_size, config.seq_size)))
        self.dropout = nn.Dropout(config.dropout)
    
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

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config) for _ in range(config.num_head)]
        )

    def forward(self, input):
        return torch.cat([h(input) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    '''Implements multi layer perceptron after attention.'''

    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.num_embed, 4 * config.num_embed),
            nn.ReLU(),
            nn.Linear(4 * config.num_embed, config.num_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    '''Combines multi-head attention with MLP computation.'''

    def __init__(self, config):
        super().__init__()
        self.self_att = MultiHeadAttention(config)
        self.feed_fwd = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.num_embed)
        self.norm2 = nn.LayerNorm(config.num_embed)

    def forward(self, x):
        x = x + self.self_att(self.norm1(x))
        x = x + self.feed_fwd(self.norm2(x))
        return x


class GPT(nn.Module):
    '''Implement bigram model, predicting next token.'''

    def __init__(self, config: ConfigGPT):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.num_embed)
        self.position_embed = nn.Embedding(config.seq_size, config.num_embed)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layer)])
        self.norm_last = nn.LayerNorm(config.num_embed)
        self.linear_out = nn.Linear(config.num_embed, config.vocab_size)
        self.apply(self._init_params)
        self.device = config.device
        self.seq_size = config.seq_size

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
        tok_emb = self.token_embed(idx)  # (B, T, C)
        pos_emb = self.position_embed(
            torch.arange(T, device=self.device))  # (T, C)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_size:]
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
