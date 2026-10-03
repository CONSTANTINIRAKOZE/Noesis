# Noesis

*νόησις (Greek): the act of understanding.*

Noesis is a public record of one person learning how language models think, built from the ground up and then opened up to see what's inside.

The model being built here is called **Walter**, after [Walter Pitts](https://en.wikipedia.org/wiki/Walter_Pitts). In 1943, Pitts and Warren McCulloch wrote the first mathematical model of a neuron. Every neural network since, including Walter, descends from that idea.

## Why this exists

I want to answer one question honestly: **how does something as human as text become mathematics, and how does that mathematics end up predicting what comes next?**

I'm not taking a shortcut through a library. I'm climbing one rung at a time, and every rung is a working model I built, measured and understood before moving on. Once the climb is done, the real work starts: mechanistic interpretability, which means looking inside Walter to find the circuits he uses.

## The ladder

| Rung | What Walter becomes | Status |
|---|---|---|
| 1 | [**A counting table.** Bigram model: text → numbers → probabilities → a loss](rung1_bigram/walter_counts.ipynb) | ✅ |
| 2 | **A tiny neural net** that *learns* the same table by lowering the loss | ⏳ |
| 3 | **A network that remembers** several characters back (MLP) | |
| 4 | **Attention.** Walter learns which earlier parts of the text matter | |
| 5 | **A small GPT** | |
| 6 | **Opening Walter up.** What do his attention heads actually do? | |

## Rung 1 in one picture

Walter's first brain is a 27 × 27 table counting how often each letter follows another across 32,000 names. Nobody wrote rules into it, but the patterns show up anyway (`q` → `u`, names ending in `a` and `n`).

| | Loss (lower is better) |
|---|---|
| Random guessing | 3.30 |
| Walter, Rung 1 | **2.45** |

That gap is pattern that Walter found on his own.

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook rung1_bigram/walter_counts.ipynb
```

## Standing on shoulders

The path follows Andrej Karpathy's *Neural Networks: Zero to Hero*. The destination is the interpretability work of Chris Olah, Neel Nanda and Anthropic's interpretability team. The dataset (`data/names.txt`) comes from Karpathy's [makemore](https://github.com/karpathy/makemore).

---

Built by **GASASIRA Irakoze Constantin** in Kigali, Rwanda.
