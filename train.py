import torch
import torch.nn.functional as F

# read file 'train.txt'
link_train = 'train.txt'
words = open(link_train, 'r', encoding='utf-8').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
num_chars = len(itos)

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []
  for w in words:

    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)

  return X, Y

Xtr, Ytr = build_dataset(words)

# Initialize weights
C = torch.randn((num_chars, 2))

ys = C[:,1]
xs= C[:,0]

tmp=torch.arange(6).view(-1, 3)

emb = C[Xtr]

W1 = torch.randn((6, 100))
b1 = torch.randn(100)

h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
W2 = torch.randn((100, num_chars))
b2 = torch.randn(num_chars)

logits = h @ W2 + b2

counts = logits.exp()

# Calculate probabilities
prob = counts / counts.sum(1, keepdims=True)

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((num_chars, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, num_chars), generator=g)
b2 = torch.randn(num_chars, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
  p.requires_grad = True

# Model optimization
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(20000):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  stepi.append(i)
  lossi.append(loss.log10().item())

emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)

# Save the model
link_model = 'model.torch'
torch.save({'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, link_model)

print(f'Model saved to {link_model}')