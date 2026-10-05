"""Everything the web app asks of Walter, in plain Python (no HTTP here).

The server in `server.py` is a thin layer over this class, so the same calls
can be tested directly.
"""
import os
import sys
import threading

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rung5_gpt'))
import walter_gpt as wg  # noqa: E402


class Walter:
    def __init__(self, checkpoint=wg.CHECKPOINT_PATH):
        self.names = wg.Names()
        self.model = wg.load_or_train(self.names, checkpoint)
        self.model.eval()
        self.cfg = self.model.cfg
        self.itos = self.names.itos
        self.max_name_len = self.cfg.block_size - 1
        self.all_names = set(sum(self.names.words.values(), []))
        self.train_names = set(self.names.words['train'])
        # Head masks are shared model state, so ablated requests take turns.
        self._lock = threading.Lock()
        self._bigram = self._build_bigram()
        Xdev, Ydev = self.names.data['dev']
        self.dev_loss = wg.evaluate(self.model, Xdev, Ydev)

    # -- helpers ------------------------------------------------------------

    def _build_bigram(self):
        """Rung 1's table (+1 smoothing), counted on the training names."""
        X, Y = self.names.data['train']
        keep = Y != -1
        N = torch.zeros((self.cfg.vocab_size, self.cfg.vocab_size))
        N.index_put_((X[keep], Y[keep]), torch.ones(int(keep.sum())), accumulate=True)
        return (N + 1) / (N + 1).sum(1, keepdim=True)

    def validate(self, text, allow_empty=False):
        text = (text or '').strip().lower()
        if not text and not allow_empty:
            raise ValueError('Type at least one letter.')
        if len(text) > self.max_name_len:
            raise ValueError(f'Walter only reads up to {self.max_name_len} letters.')
        bad = sorted({c for c in text if c not in self.names.stoi or c == '.'})
        if bad:
            raise ValueError(f"Walter only knows the letters a-z (not {' '.join(repr(c) for c in bad)}).")
        return text

    def _run(self, idx, ablate=()):
        """Forward pass with the given (layer, head) pairs switched off."""
        with self._lock:
            for l, h in ablate:
                self.model.blocks[l].attn.head_mask[h] = 0
            try:
                with torch.no_grad():
                    logits, _, residuals = self.model(idx, return_residuals=True)
                att = [b.attn.att[0].tolist() for b in self.model.blocks]
            finally:
                for l, h in ablate:
                    self.model.blocks[l].attn.head_mask[h] = 1
        return logits, residuals, att

    def _check_heads(self, ablate):
        out = []
        for l, h in ablate or []:
            if not (0 <= l < self.cfg.n_layer and 0 <= h < self.cfg.n_head):
                raise ValueError(f'There is no head L{l} H{h}.')
            out.append((l, h))
        return out

    def _dist(self, probs, k=None):
        order = probs.argsort(descending=True)
        if k is not None:
            order = order[:k]
        return [{'char': self.itos[i.item()], 'p': round(probs[i].item(), 5)} for i in order]

    # -- the things the app can do -------------------------------------------

    def info(self):
        return {
            'n_layer': self.cfg.n_layer, 'n_head': self.cfg.n_head, 'n_embd': self.cfg.n_embd,
            'params': self.model.num_params(), 'max_name_len': self.max_name_len,
            'dev_loss': round(self.dev_loss, 4), 'n_train_names': len(self.train_names),
        }

    def generate(self, n=10, temperature=1.0, prefix='', seed=None):
        prefix = self.validate(prefix, allow_empty=True)
        if len(prefix) >= self.max_name_len:
            raise ValueError(f'The prefix must be shorter than {self.max_name_len} letters.')
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else torch.seed() % (2 ** 31))
        out = []
        with torch.no_grad():
            for _ in range(n):
                idx = self.names.encode(prefix)[0].tolist()
                while len(idx) < self.cfg.block_size:
                    logits, _ = self.model(torch.tensor([idx]))
                    probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                    ix = torch.multinomial(probs, 1, generator=g).item()
                    if ix == 0:
                        break
                    idx.append(ix)
                name = self.names.decode(idx[1:])
                out.append({'name': name, 'known': name in self.all_names})
        return out

    def next_letter(self, prefix='', ablate=()):
        prefix = self.validate(prefix, allow_empty=True)
        if len(prefix) >= self.max_name_len:
            raise ValueError('That prefix is already as long as a name can be.')
        logits, _, _ = self._run(self.names.encode(prefix), self._check_heads(ablate))
        probs = F.softmax(logits[0, -1], dim=-1)
        return {'prefix': prefix, 'distribution': self._dist(probs)}

    def score(self, name):
        """Per-letter surprise (-log p) from the GPT and from the Rung 1 bigram."""
        name = self.validate(name)
        idx = self.names.encode(name)
        targets = [self.names.stoi[c] for c in name] + [0]
        logits, _, _ = self._run(idx)
        probs = F.softmax(logits[0], dim=-1)
        steps = []
        for t, y in enumerate(targets):
            p_gpt = probs[t, y].item()
            p_bi = self._bigram[idx[0, t], y].item()
            steps.append({
                'context': ('.' + name)[:t + 1], 'char': self.itos[y],
                'p_gpt': round(p_gpt, 5), 'p_bigram': round(p_bi, 5),
                'surprise_gpt': round(-torch.log(torch.tensor(p_gpt)).item(), 4),
                'surprise_bigram': round(-torch.log(torch.tensor(p_bi)).item(), 4),
                'top': self._dist(probs[t], k=3),
            })
        mean = lambda key: round(sum(s[key] for s in steps) / len(steps), 4)
        return {
            'name': name, 'steps': steps,
            'loss_gpt': mean('surprise_gpt'), 'loss_bigram': mean('surprise_bigram'),
            'in_dataset': name in self.all_names, 'in_training': name in self.train_names,
        }

    def inspect(self, name, ablate=()):
        """Attention patterns for every head, plus the logit lens at the last position."""
        name = self.validate(name, allow_empty=True)
        if len(name) >= self.max_name_len:
            raise ValueError(f'Use at most {self.max_name_len - 1} letters here.')
        ablate = self._check_heads(ablate)
        idx = self.names.encode(name)
        logits, residuals, att = self._run(idx, ablate)
        with torch.no_grad():
            lens = []
            for i, r in enumerate(residuals):
                p = F.softmax(self.model.lm_head(self.model.ln_f(r[0, -1])), dim=-1)
                lens.append({'stage': 'embeddings' if i == 0 else f'after block {i - 1}', 'top': self._dist(p, k=5)})
        return {
            'name': name, 'tokens': ['.'] + list(name),
            'attention': att,                       # [layer][head][row][col]
            'lens': lens,
            'next': self._dist(F.softmax(logits[0, -1], dim=-1), k=8),
            'ablated': [list(x) for x in ablate],
        }
