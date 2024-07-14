import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # Number of independent training examples to train on in parallell
block_size = 8  # Maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu" # Actually faster to use cpu, I believe this is because batch size is small
print("Using device: " + device)
eval_iters = 200
# ----------------

torch.manual_seed(1337) # Same seed as used in Karpathy's tutorial video

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Open file and read it
with open("tiny_shakespear.txt", "r", encoding="utf-8") as f:
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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) - (Batch, Time), where time is input string - tensor of integers  
        logits = self.token_embedding_table(idx) # (B, T, C)

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

            # get predictions
            logits, loss = self(idx)

            # Focus only on the logits from the last time step, shape: (B, C)
            logits = logits[:, -1, :]

            # Apply softmax to logits to get probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution, shape: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence, shape: (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

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