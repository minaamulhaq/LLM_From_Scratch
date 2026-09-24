from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
deval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

# loard the data from the file and print the first 1000 characters
with open("input.txt", "r", encoding="utf-8") as f:
    read = f.read()

print(read[:1000])

# Sorted list of unique characters in the file

char = sorted(list(set(read)))

# create maping from the charector to integer
char_to_int = {c: i for i, c in enumerate(char)}
int_to_char = {i: c for i, c in enumerate(char)}

# encoder and decoder function
encoder = lambda s: [char_to_int[c] for c in s]
decoder = lambda l: "".join([int_to_char[i] for i in l])
print(encoder("Hello"))
print(decoder([46, 43, 50, 50, 53]))

# loard the data and convert it to tensor
data = torch.tensor(encoder(read), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# split the data into train and test set
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# train_data[:block_size+1]


x = train_data[:block_size]
y = train_data[1 : block_size + 1]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x.to(device)
    y.to(device)
    return x, y

# no calculate the gradients for the model parameters
@torch.no_grad()
def estimate_loss():
    # Evaluate the model on the validation set 
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(deval_interval)
        for k in range(deval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
      # Calculate logits regardless of whether targets are provided
      logits = self.token_embedding_table(idx)

      if targets is None:
        loss = None
      else:
        B,T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

      return logits,loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(len(char))

optimiser = torch.optim.Adam(m.parameters(), lr=1e-3)
for step in range(max_iters):

  if step % deval_interval == 0 or step == max_iters - 1:
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

  xb,yb = get_batch('train')
  logits, loss = m(xb,yb)
  optimiser.zero_grad(set_to_none=True)
  loss.backward()
  optimiser.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))