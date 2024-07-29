import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """One Head of self-attention"""

    def __init__(self, block_size, head_size, n_embd, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei) # (B, T, T)

        # Perform the weighted aggregation of the values
        v = self.value(x)   # (B, T, C)
        out = wei @ v   # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, block_size, num_heads, head_size, n_embd, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))    # Projection layer
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Projection layer
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: Communication followed by computation"""

    def __init__(self, block_size, n_embd, n_head, dropout) -> None:
        super().__init__()
        head_size = n_embd // n_head    # n_embd: embedding dimension, n_head: the number of heads we'd like to hav
        self.sa = MultiHeadAttention(block_size, n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # By adding x in the following forward pass, we get a residual path which is helpful when optimizing deep neural networks, as the gradients get a "highway" back towards the input
        x = x + self.sa(self.ln1(x))  # (B, T, C) - C is n_embd as MultiHeadAttention concatenates n_heads of dimension head_size
        x = x + self.ffwd(self.ln2(x))    # (B, T, C)
        return x

class TransformerLanguageModel(nn.Module):
    """A Language Model based on the Transformer architecture"""

    def __init__(self, vocab_size, block_size, device, n_embd, n_head, n_layer, dropout) -> None:
        super().__init__()

        self.block_size = block_size
        self.device = device

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)    # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):

        B, T = idx.shape # (B, T)

        # idx and targets are both (B, T) - (Batch, Time), where time is input string - tensor of integers  
        tok_emb = self.token_embedding_table(idx) # (B, T, C) - C is same as n_embd - Embedding the tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)    # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # .view() is similar to numpy's .reshape() function
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # Focus only on the logits from the last time step, shape: (B, C)
            logits = logits[:, -1, :]

            # Apply softmax to logits to get probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution, shape: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence, shape: (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
