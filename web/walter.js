// Walter (the Rung 5 GPT) running in plain JavaScript, so the site needs no server.
// It mirrors app/walter_service.py: same functions, same results, same error messages.
// tests/test_web_parity.py checks it against PyTorch number for number.

export class Walter {
  constructor(meta, buffer) {
    this.meta = meta;
    this.cfg = meta.config;
    this.itos = meta.itos;
    this.stoi = Object.fromEntries(meta.itos.map((c, i) => [c, i]));
    this.maxNameLen = this.cfg.block_size - 1;
    this.train = new Set(meta.train_names);
    this.all = new Set([...meta.train_names, ...meta.other_names]);
    const data = new Float32Array(buffer);
    this.w = {};
    for (const [key, { offset, shape }] of Object.entries(meta.tensors)) {
      this.w[key] = data.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
    const V = this.cfg.vocab_size;
    this.bigram = Array.from({ length: V }, (_, i) => meta.bigram.slice(i * V, (i + 1) * V));
  }

  static async load(base = '.') {
    const [meta, buffer] = await Promise.all([
      fetch(`${base}/walter.json`).then((r) => r.json()),
      fetch(`${base}/walter.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new Walter(meta, buffer);
  }

  // ---- building blocks ------------------------------------------------------

  // y = x W^T + b for one vector; W is (out, in) as in torch.nn.Linear
  linear(x, key, bias = true) {
    const W = this.w[`${key}.weight`], b = bias ? this.w[`${key}.bias`] : null;
    const nIn = x.length, nOut = W.length / nIn, y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = b ? b[o] : 0;
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  layerNorm(x, key) {
    const g = this.w[`${key}.weight`], b = this.w[`${key}.bias`], n = x.length;
    let mean = 0;
    for (const v of x) mean += v;
    mean /= n;
    let vr = 0;
    for (const v of x) vr += (v - mean) ** 2;
    const inv = 1 / Math.sqrt(vr / n + 1e-5);
    return Float64Array.from(x, (v, i) => (v - mean) * inv * g[i] + b[i]);
  }

  static gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }

  static softmax(xs) {
    const m = Math.max(...xs);
    const e = xs.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // Full forward pass over a sequence of token ids. `ablate` is a list of [layer, head] to switch off.
  forward(idx, ablate = []) {
    const { n_layer: L, n_head: H, n_embd: C } = this.cfg, T = idx.length, hs = C / H;
    const off = new Set(ablate.map(([l, h]) => `${l},${h}`));
    let x = idx.map((tok, t) => Float64Array.from({ length: C },
      (_, c) => this.w['tok.weight'][tok * C + c] + this.w['pos.weight'][t * C + c]));
    const residuals = [x], attention = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}`;
      const qkv = x.map((v) => this.linear(this.layerNorm(v, `${p}.ln1`), `${p}.attn.qkv`, false));
      const heads = [], out = x.map(() => new Float64Array(C));
      for (let h = 0; h < H; h++) {
        const pattern = [];
        for (let t = 0; t < T; t++) {
          const scores = [];
          for (let s = 0; s <= t; s++) {
            let d = 0;
            for (let i = 0; i < hs; i++) d += qkv[t][h * hs + i] * qkv[s][C + h * hs + i];
            scores.push(d / Math.sqrt(hs));
          }
          const a = Walter.softmax(scores);
          pattern.push([...a, ...new Array(T - t - 1).fill(0)]);
          if (off.has(`${l},${h}`)) continue;
          for (let s = 0; s <= t; s++) {
            for (let i = 0; i < hs; i++) out[t][h * hs + i] += a[s] * qkv[s][2 * C + h * hs + i];
          }
        }
        heads.push(pattern);
      }
      attention.push(heads);
      x = x.map((v, t) => {
        const a = this.linear(out[t], `${p}.attn.proj`);
        return Float64Array.from(v, (vi, i) => vi + a[i]);
      });
      x = x.map((v) => {
        const hid = this.linear(this.layerNorm(v, `${p}.ln2`), `${p}.mlp.net.0`).map(Walter.gelu);
        const m = this.linear(hid, `${p}.mlp.net.2`);
        return Float64Array.from(v, (vi, i) => vi + m[i]);
      });
      residuals.push(x);
    }
    const logits = x.map((v) => Array.from(this.readout(v)));
    return { logits, attention, residuals };
  }

  readout(v) { return this.linear(this.layerNorm(v, 'ln_f'), 'lm_head'); }

  // ---- helpers mirroring the Python service ----------------------------------

  validate(text, allowEmpty = false) {
    text = (text || '').trim().toLowerCase();
    if (!text && !allowEmpty) throw new Error('Type at least one letter.');
    if (text.length > this.maxNameLen) throw new Error(`Walter only reads up to ${this.maxNameLen} letters.`);
    const bad = [...new Set([...text].filter((c) => !(c in this.stoi) || c === '.'))].sort();
    if (bad.length) throw new Error(`Walter only knows the letters a-z (not ${bad.map((c) => `'${c}'`).join(' ')}).`);
    return text;
  }

  checkHeads(ablate) {
    for (const [l, h] of ablate || []) {
      if (!(l >= 0 && l < this.cfg.n_layer && h >= 0 && h < this.cfg.n_head)) throw new Error(`There is no head L${l} H${h}.`);
    }
    return ablate || [];
  }

  encode(name) { return [0, ...[...name].map((c) => this.stoi[c])]; }

  dist(probs, k) {
    const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
    return order.slice(0, k ?? order.length).map((i) => ({ char: this.itos[i], p: round(probs[i], 5) }));
  }

  // ---- what the app can do --------------------------------------------------

  info() {
    const { n_layer, n_head, n_embd } = this.cfg;
    return { n_layer, n_head, n_embd, params: this.meta.n_params, max_name_len: this.maxNameLen,
      dev_loss: this.meta.dev_loss, n_train_names: this.train.size };
  }

  generate(n = 10, temperature = 1.0, prefix = '', seed = null) {
    prefix = this.validate(prefix, true);
    if (prefix.length >= this.maxNameLen) throw new Error(`The prefix must be shorter than ${this.maxNameLen} letters.`);
    const rand = mulberry32(seed ?? Math.floor(Math.random() * 2 ** 31));
    const out = [];
    for (let k = 0; k < n; k++) {
      const idx = this.encode(prefix);
      while (idx.length < this.cfg.block_size) {
        const { logits } = this.forward(idx);
        const probs = Walter.softmax(logits[idx.length - 1].map((v) => v / temperature));
        let r = rand(), ix = 0;
        while (ix < probs.length - 1 && (r -= probs[ix]) > 0) ix++;
        if (ix === 0) break;
        idx.push(ix);
      }
      const name = idx.slice(1).map((i) => this.itos[i]).join('');
      out.push({ name, known: this.all.has(name) });
    }
    return out;
  }

  nextLetter(prefix = '', ablate = []) {
    prefix = this.validate(prefix, true);
    if (prefix.length >= this.maxNameLen) throw new Error('That prefix is already as long as a name can be.');
    const { logits } = this.forward(this.encode(prefix), this.checkHeads(ablate));
    return { prefix, distribution: this.dist(Walter.softmax(logits[logits.length - 1])) };
  }

  score(name) {
    name = this.validate(name);
    const idx = this.encode(name), targets = [...idx.slice(1), 0];
    const { logits } = this.forward(idx);
    const steps = targets.map((y, t) => {
      const probs = Walter.softmax(logits[t]);
      const pG = probs[y], pB = this.bigram[idx[t]][y];
      return {
        context: ('.' + name).slice(0, t + 1), char: this.itos[y],
        p_gpt: round(pG, 5), p_bigram: round(pB, 5),
        surprise_gpt: round(-Math.log(pG), 4), surprise_bigram: round(-Math.log(pB), 4),
        top: this.dist(probs, 3),
      };
    });
    const mean = (k) => round(steps.reduce((a, s) => a + s[k], 0) / steps.length, 4);
    return { name, steps, loss_gpt: mean('surprise_gpt'), loss_bigram: mean('surprise_bigram'),
      in_dataset: this.all.has(name), in_training: this.train.has(name) };
  }

  inspect(name = '', ablate = []) {
    name = this.validate(name, true);
    if (name.length >= this.maxNameLen) throw new Error(`Use at most ${this.maxNameLen - 1} letters here.`);
    ablate = this.checkHeads(ablate);
    const { logits, attention, residuals } = this.forward(this.encode(name), ablate);
    const last = residuals.map((r) => r[r.length - 1]);
    return {
      name, tokens: ['.', ...name],
      attention,
      lens: last.map((v, i) => ({ stage: i === 0 ? 'embeddings' : `after block ${i - 1}`,
        top: this.dist(Walter.softmax(Array.from(this.readout(v))), 5) })),
      next: this.dist(Walter.softmax(logits[logits.length - 1]), 8),
      ablated: ablate,
    };
  }
}

function round(v, d) { const f = 10 ** d; return Math.round(v * f) / f; }

// Small seeded random number generator, so a seed gives repeatable names.
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// erf, for the exact GELU PyTorch uses (Abramowitz & Stegun 7.1.26, error < 1.5e-7).
function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
