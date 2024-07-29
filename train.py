import os
from transformer import TransformerLanguageModel
import torch

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
n_head = 4  # Should be a factor of n_embd
n_layer = 6 # Number of Blocks
dropout = 0.3
# ----------------

torch.manual_seed(1337) # Same seed as used in Karpathy's tutorial video

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Open file and read it
dataset_file = "tiny_shakespear.txt"
with open(dataset_file, "r", encoding="utf-8") as f:
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

model = TransformerLanguageModel(vocab_size, block_size, device, n_embd, n_head, n_layer, dropout)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters- 1:
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

model_dir = f"./models/model-{dataset_file.split(".")[0]}-{block_size}-ctx-{n_layer}-lyr-{n_head}-head-{max_iters}-iters-{batch_size}-batch"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

torch.save(model, f"{model_dir}/model.pth")
print("Model weights saved!")

# Save all chars in a file in model_dir
chars_file = f"{model_dir}/chars.txt"
if not os.path.exists(chars_file):
    with open(chars_file, "w", encoding="utf-8") as f:
        f.write("\n".join(chars))