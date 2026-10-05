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
| 2 | [**A tiny neural net** that *learns* the same table by lowering the loss, with gradients derived by hand](rung2_neural_bigram/walter_learns.ipynb) | ✅ |
| 3 | [**A network that remembers** several characters back (embeddings + MLP)](rung3_mlp/walter_remembers.ipynb) | ✅ |
| 4 | [**Attention.** Walter learns which earlier parts of the text matter](rung4_attention/walter_attends.ipynb) | ✅ |
| 5 | [**A small GPT**](rung5_gpt/walter_becomes_gpt.ipynb) (code in [`walter_gpt.py`](rung5_gpt/walter_gpt.py)) | ✅ |
| 6 | [**Opening Walter up.** What do his attention heads actually do?](rung6_interpretability/opening_walter_up.ipynb) | ✅ |

## The climb in numbers

Every rung is scored on the same 3,203 names it never trained on (loss: average surprise per letter, lower is better).

| | Dev loss |
|---|---|
| Random guessing | 3.30 |
| Rung 1: counting letter pairs | 2.45 |
| Rung 2: *learning* letter pairs by gradient descent | 2.45 |
| Rung 3: embeddings + MLP, 3 letters back | 2.11 |
| Rung 4: attention alone, 4 heads | 2.15 |
| Rung 5: 4-layer GPT, ~200k parameters | **1.95** |

*Rungs 1–2 train and score on all names. The 2.45 here is the same bigram rebuilt on the training names only and scored on the dev names (Rung 3, Step 2), and it comes out the same.*

A few things the climb showed along the way:

- **Rung 2:** learning and counting land on the same table, and regularisation turns out to be smoothing in disguise.
- **Rung 3:** the letter embeddings pick up structure nobody wrote in (`o` ≈ `u`, vowels a little alike, `.` unlike anything), but not tidy human categories.
- **Rung 4:** attention alone *gathers* well but can't beat the MLP, which can *think*. Rung 5 needs both.
- **Rung 6:** Walter has two **previous-letter heads** in layer 0 that back each other up. Knock both out and he forgets that names never triple a letter: P(third repeat after a double) goes from 0.3% to 4.9%. That's a small circuit, found and causally tested.

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open the rungs in order. Rungs 1–2 use only numpy, and Rungs 3–6 use PyTorch on a CPU. Rung 5 trains Walter in about 10–15 minutes and saves him to `rung5_gpt/walter_gpt.pt`. Rung 6 loads that file, or retrains Walter with the same seed if it's missing.

## Standing on shoulders

The path follows Andrej Karpathy's *Neural Networks: Zero to Hero*. The destination is the interpretability work of Chris Olah, Neel Nanda and Anthropic's interpretability team. The dataset (`data/names.txt`) comes from Karpathy's [makemore](https://github.com/karpathy/makemore).

---

Built by **GASASIRA Irakoze Constantin** in Kigali, Rwanda.
