import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64  # Number of independent training examples to train on in parallell
block_size = 128  # Maximum context length for predictions
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu" # As I run this on a MacBook M2 Pro chip, I can use MPS to speed up matrix multiplications
print("Using device: " + device)
eval_iters = 200
n_embd = 128 # Number of dimensions in embedding space
n_head = 4
n_layer = 6 # Number of Blocks
dropout = 0.3
# ----------------

torch.manual_seed(1337) # Same seed as used in Karpathy's tutorial video

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Open file and read it
with open("hhgttg.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get all tokens (int our case all individual characters)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from chars to ints and back
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading 
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generate batch_size number of random indices
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One Head of self-attention"""

    def __init__(self, head_size) -> None:
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

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))    # Projection layer
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd) -> None:
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

    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head    # n_embd: embedding dimension, n_head: the number of heads we'd like to hav
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # By adding x in the following forward pass, we get a residual path which is helpful when optimizing deep neural networks, as the gradients get a "highway" back towards the input
        x = x + self.sa(self.ln1(x))  # (B, T, C) - C is n_embd as MultiHeadAttention concatenates n_heads of dimension head_size
        x = x + self.ffwd(self.ln2(x))    # (B, T, C)
        return x

class TransformerLanguageModel(nn.Module):
    """A Language Model based on the Transformer architecture"""

    def __init__(self) -> None:
        super().__init__()
        
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)    # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):

        B, T = idx.shape # (B, T)

        # idx and targets are both (B, T) - (Batch, Time), where time is input string - tensor of integers  
        tok_emb = self.token_embedding_table(idx) # (B, T, C) - C is same as n_embd - Embedding the tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
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
            idx_cond = idx[:, -block_size:]

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


model = TransformerLanguageModel()
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters- 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    # Calculate gradients based on loss
    loss.backward()

    # Optimize neural network based on gradients
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))