import sys
import torch

CBOLD = '\33[1m'
CITALIC = '\33[3m'
CGREEN  = '\33[32m'
CBLUE = '\33[34m'
CEND = '\33[0m'

device = "mps" if torch.backends.mps.is_available() else "cpu" # As I run this on a MacBook M2 Pro chip, I can use MPS to speed up matrix multiplications

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "model-tiny_shakespear-128-ctx-6-lyr-4-head-6000-iters-64-batch"

      
model_dir = f"models/{model_name}"

model = torch.load(model_dir + "/model.pth")
m = model.to(device)
model.eval()
print(f"\nRunning model {CBOLD}{model_dir}{CEND} on device {CBOLD}{device}{CEND}\n")

# Open chars file and read it
with open(model_dir + "/chars.txt", "r", encoding="utf-8") as f:
        text = f.read()

# Get all tokens (int our case all individual characters)
chars = sorted(list(set(text)))

# Create mapping from chars to ints and back
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

user_input = input(f"{CGREEN}>>{CEND} ")
while user_input.lower() != "quit":
    
    context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    print(f"\n{CBLUE}TinyGPT:{CEND}", end="\n\n")
    # Generate from the model
    for tok in m.generate(context, max_new_tokens=500):
         print(decode(tok[0].tolist()), end="", flush=True)
    print("\n")

    user_input = input(f"{CGREEN}>>{CEND} ")

    