"""Export Walter for the browser: web/walter.bin (all weights, float32) + web/walter.json.

Run from the repository root after Rung 5 has trained Walter:
    python -m app.export_weights
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rung5_gpt'))
import walter_gpt as wg  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web')


def main():
    names = wg.Names()
    model = wg.load_or_train(names)
    model.eval()

    tensors, offset, chunks = {}, 0, []
    for key, t in model.state_dict().items():
        if key.endswith('.mask'):                 # the causal mask is rebuilt in JavaScript
            continue
        a = t.detach().numpy().astype('<f4').ravel()
        tensors[key] = {'offset': offset, 'shape': list(t.shape)}
        offset += a.size
        chunks.append(a)
    os.makedirs(WEB, exist_ok=True)
    np.concatenate(chunks).tofile(os.path.join(WEB, 'walter.bin'))

    # Rung 1's bigram table (+1 smoothing) on the training names, for the "Score" comparison.
    X, Y = names.data['train']
    keep = Y != -1
    N = torch.zeros((names.vocab_size, names.vocab_size))
    N.index_put_((X[keep], Y[keep]), torch.ones(int(keep.sum())), accumulate=True)
    P = (N + 1) / (N + 1).sum(1, keepdim=True)

    Xdev, Ydev = names.data['dev']
    meta = {
        'config': model.cfg.__dict__,
        'itos': [names.itos[i] for i in range(names.vocab_size)],
        'tensors': tensors,
        'n_params': model.num_params(),
        'dev_loss': round(wg.evaluate(model, Xdev, Ydev), 4),
        'bigram': [round(v, 7) for v in P.flatten().tolist()],
        'train_names': sorted(set(names.words['train'])),
        'other_names': sorted(set(names.words['dev'] + names.words['test']) - set(names.words['train'])),
    }
    with open(os.path.join(WEB, 'walter.json'), 'w') as f:
        json.dump(meta, f, separators=(',', ':'))
    print(f'wrote web/walter.bin ({offset * 4 / 1e6:.2f} MB) and web/walter.json')


if __name__ == '__main__':
    main()
