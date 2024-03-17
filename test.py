import argparse
import torch
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='Perform test evaluation')
    parser.add_argument('test_file', type=str, help='Path to the test file')
    parser.add_argument('model_file', type=str, help='Path to the model file')	

    args = parser.parse_args()

    block_size = 3

    link_test = args.test_file
    words = open(link_test, 'r', encoding='utf-8').read().splitlines()

    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    num_chars = len(itos)

    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    Xte = torch.tensor(X)
    Yte = torch.tensor(Y)

    link_model = args.model_file
    model = torch.load(link_model)
    C, W1, b1, W2, b2 = model['C'], model['W1'], model['b1'], model['W2'], model['b2']

    emb = C[Xte]  # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    loss = F.cross_entropy(logits, Yte)
    print(f'test loss: {loss}')

    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            emb = C[torch.tensor([context])] # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

if __name__ == '__main__':
    main()