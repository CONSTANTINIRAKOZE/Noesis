"""Walter as a small GPT.

Everything Rung 5 builds lives in this one file so that Rung 6 can open up the
exact same model. Read it top to bottom: data, the four building blocks
(attention, MLP, block, GPT), then training and sampling.
"""
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES_PATH = os.path.join(HERE, '..', 'data', 'names.txt')
CHECKPOINT_PATH = os.path.join(HERE, 'walter_gpt.pt')


# ---------------------------------------------------------------------------
# Data: the same vocabulary, padding and train/dev/test split as Rungs 3 and 4
# ---------------------------------------------------------------------------

class Names:
    def __init__(self, path=NAMES_PATH):
        words = open(path).read().splitlines()
        chars = sorted(set(''.join(words)))
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.block_size = max(len(w) for w in words) + 1

        random.seed(42)
        shuffled = words[:]
        random.shuffle(shuffled)
        n1, n2 = int(0.8 * len(shuffled)), int(0.9 * len(shuffled))
        self.words = {'train': shuffled[:n1], 'dev': shuffled[n1:n2], 'test': shuffled[n2:]}
        self.data = {split: self.encode_names(ws) for split, ws in self.words.items()}

    def encode_names(self, words):
        """Input '.emma' -> target 'emma.', padded to block_size; padded targets are -1 (ignored)."""
        X = torch.zeros((len(words), self.block_size), dtype=torch.long)
        Y = torch.full((len(words), self.block_size), -1, dtype=torch.long)
        for i, w in enumerate(words):
            ix = [self.stoi[c] for c in w]
            X[i, 1:len(ix) + 1] = torch.tensor(ix)
            Y[i, :len(ix) + 1] = torch.tensor(ix + [0])
        return X, Y

    def encode(self, name):
        """'emma' -> tensor([[0, 5, 13, 13, 1]]), ready to feed Walter."""
        return torch.tensor([[0] + [self.stoi[c] for c in name]])

    def decode(self, ix):
        return ''.join(self.itos[int(i)] for i in ix)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 27
    block_size: int = 16
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Rung 4's multi-head attention, with all heads computed in one batched matmul."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_size = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)   # query, key, value for every head at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)                  # mixes the heads back together
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(cfg.block_size, cfg.block_size)) == 0)
        # Interpretability handles (Rung 6): the last attention pattern, and a per-head
        # multiplier on each head's output (1 = normal, 0 = head switched off).
        self.att = None
        self.head_mask = torch.ones(cfg.n_head)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v))  # (B, heads, T, hs)
        scores = q @ k.transpose(-2, -1) / self.head_size ** 0.5
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(scores, dim=-1)
        self.att = att.detach()
        out = self.drop(att) @ v                                   # (B, heads, T, hs)
        out = out * self.head_mask.view(1, -1, 1, 1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # heads side by side again
        return self.drop(self.proj(out))


class MLP(nn.Module):
    """The 'thinking' part: the same idea as Rung 3's hidden layer, applied at every position."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: gather (attention), then think (MLP).

    Each sub-layer reads a normalised copy of the residual stream and *adds* its
    result back to it, so information flows straight through the stack.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, return_residuals=False):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        residuals = [x]
        for block in self.blocks:
            x = block(x)
            residuals.append(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if return_residuals:
            return logits, loss, residuals
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, n, names, seed=2026, temperature=1.0):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(n):
            idx = [0]
            while len(idx) < self.cfg.block_size:
                logits, _ = self(torch.tensor([idx]))
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                ix = torch.multinomial(probs, 1, generator=g).item()
                if ix == 0:
                    break
                idx.append(ix)
            out.append(names.decode(idx[1:]))
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, X, Y, batch_size=1024):
    """Average loss per predicted character over a whole split."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(X), batch_size):
        logits, _ = model(X[i:i + batch_size])
        y = Y[i:i + batch_size].reshape(-1)
        keep = y != -1
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1))[keep], y[keep], reduction='sum').item()
        count += keep.sum().item()
    model.train(was_training)
    return total / count


def train(names, cfg=None, steps=20_000, batch_size=64, lr=1e-3, weight_decay=0.1,
          seed=2026, log_every=2_000, verbose=True):
    cfg = cfg or Config(vocab_size=names.vocab_size, block_size=names.block_size)
    torch.manual_seed(seed)
    model = GPT(cfg)
    Xtr, Ytr = names.data['train']
    Xdev, Ydev = names.data['dev']

    # Weight decay on the big matrices only; biases, LayerNorms and embeddings are left alone.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'tok' not in n and 'pos' not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and 'tok' not in n and 'pos' not in n)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.99))
    # Learning rate: short warm-up, then a cosine glide down to 10% of the peak.
    warmup = steps // 50
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(s / steps * 3.14159265)).item())))

    g = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(Xtr), (batch_size,), generator=g)
        _, loss = model(Xtr[ix], Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            tr, dv = evaluate(model, Xtr[:5000], Ytr[:5000]), evaluate(model, Xdev, Ydev)
            history.append((step, tr, dv))
            if verbose:
                print(f'step {step:6d}/{steps}  train {tr:.4f}  dev {dv:.4f}')
    model.eval()
    return model, history


def save(model, path=CHECKPOINT_PATH):
    torch.save({'cfg': model.cfg.__dict__, 'state': model.state_dict()}, path)


def load(path=CHECKPOINT_PATH):
    ckpt = torch.load(path, weights_only=True)
    model = GPT(Config(**ckpt['cfg']))
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model


def load_or_train(names, path=CHECKPOINT_PATH):
    """Load the Walter trained in Rung 5, or train him from scratch (same seed, same result)."""
    if os.path.exists(path):
        return load(path)
    model, _ = train(names)
    save(model, path)
    return model
