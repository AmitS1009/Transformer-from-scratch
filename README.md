# Translation Model (Transformer From Scratch)

> **Elegant. Reproducible. From-first-principles.**
> A handcrafted Transformer (encoder-decoder) implemented in PyTorch with a training & inference notebook — built to teach, reproduce, and deploy production-grade sequence-to-sequence translation models.

---

## Table of contents

1. [Project Overview](#project-overview)
2. [Highlights & Features](#highlights--features)
3. [Architecture & Implementation notes](#architecture--implementation-notes)
4. [Files in this repo](#files-in-this-repo)
5. [Requirements & Installation](#requirements--installation)
6. [Quickstart — training & inference](#quickstart--training--inference)
7. [How the notebook works (step-by-step)](#how-the-notebook-works-step-by-step)
8. [Tips for experiments & scaling](#tips-for-experiments--scaling)
9. [Results examples](#results-examples)
10. [Known limitations & TODOs](#known-limitations--todos)
11. [Contribution, License](#contribution-license--contact)

---

## Project overview

This repository implements a **Transformer** (encoder-decoder) built entirely from scratch in PyTorch — no high-level Transformer library calls for the core modules. The code demonstrates how the model is constructed (positional encodings, multi-head self-attention, cross-attention, custom layer normalization, position-wise feed-forward networks), how to build masks for training, and how to do inference for sequence generation.

The model file (`transformer.py`) contains the full implementation of the encoder, decoder, attention blocks, and the high-level `Transformer` wrapper. 

---

## Highlights & Features

* Custom implementation of:

  * Scaled dot-product attention and **multi-head self-attention**
  * **Cross-attention** (encoder–decoder attention)
  * **Positional encoding** (sinusoidal)
  * **Layer normalization** implemented as a learnable module
  * Position-wise feed-forward networks and residual connections
* End-to-end notebook demonstrating:

  * Tokenization and vocab construction (source & target)
  * Mask creation (padding masks + look-ahead)
  * DataLoader / Dataset wrapper
  * Training loop with teacher forcing and inference generation
* Device agnostic: CPU / CUDA support built-in
* Clean, well-separated modules for easy experimentation and extension

---

## Architecture & Implementation notes

Key modules that were hand-implemented:

* `PositionalEncoding` — sinusoidal positional encodings computed for a fixed `max_sequence_length`.
* `SentenceEmbedding` — token embedding layer + positional addition + dropout.
* `MultiHeadAttention` & `MultiHeadCrossAttention` — Q/K/V projection and concatenation of heads; scaled dot-product implemented in `scaled_dot_product`.
* `LayerNormalization` — custom, learnable layer norm (gamma & beta).
* `PositionwiseFeedForward` — two-layer MLP with ReLU and dropout.
* `EncoderLayer`, `DecoderLayer` — standard pre/post residual + normalization patterns.
* `Transformer` wrapper exposes `forward(x, y, masks...)` and outputs logits over target vocabulary.

See full code in: `transformer.py`. 

---

## Files in this repo

* `transformer.py` — Full PyTorch implementation of Transformer modules (encoder, decoder, attention, positional encodings). 
* `TranslationModel_Transformers_from_scratch.ipynb` — Guided notebook that prepares data, creates vocabularies, builds masks, trains the model, and demonstrates translation/inference.
* `English.txt`, `Hindi.txt` (or other language text files used by the notebook) — parallel sentence corpora used in the demo.
* `README.md` — (this file)

---

## Requirements & installation

Recommended: create a virtualenv.

```bash
python -m venv venv
source venv/bin/activate           # linux / macOS
# .\venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install torch numpy jupyterlab matplotlib
```

(If you have CUDA, install a CUDA-enabled `torch` wheel matching your CUDA version.)

---

## Quickstart — training & inference

1. Start Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

2. Open `TranslationModel_Transformers_from_scratch.ipynb` and run the cells in order. The notebook contains sections to:

* load `English.txt` / `Hindi.txt` and build `english_to_index` and `hindi_to_index` vocabularies,
* instantiate `Transformer(...)` with desired hyperparameters,
* create `DataLoader` and masks,
* run the training loop (with `num_epochs` variable),
* run the `translate()` helper to generate sentences.

3. Example inference (as shown in the notebook):

```py
translation = translate("what should we do when the day starts?")
print(translation)
```

---

## How the notebook works (step-by-step)

1. **Vocabulary & token definitions**

   * Defines special tokens: `START_TOKEN`, `PADDING_TOKEN`, `END_TOKEN`.
   * Builds character-level vocabularies (can be adapted to subword units like BPE).

2. **Mappings**

   * Builds `english_to_index`, `hindi_to_index`, plus reverse mappings.

3. **Model instantiation**

   * Example hyperparameters the notebook uses:

     * `d_model` (embedding dimension), `ffn_hidden` (FFN inner dim)
     * `num_heads`, `num_layers`, `drop_prob`, `max_sequence_length`
   * Instantiates `Transformer` from `transformer.py`.

4. **Dataset & DataLoader**

   * A `TextDataset` class creates pairs; the notebook builds padded batches and data masks.

5. **Mask generation**

   * `create_masks()` produces:

     * encoder padding mask (converted to `NEG_INFTY` where needed),
     * decoder self-attention mask (look-ahead + padding),
     * decoder cross-attention mask.

6. **Training loop**

   * Typical training loop with `optimizer = torch.optim.Adam(...)`, `nn.CrossEntropyLoss()` and teacher forcing.
   * Embedding resizing logic included (if vocab indices exceed initial embedding size).

7. **Inference (translate)**

   * Greedy decoding loop using model outputs to pick the next token (`argmax`), break on `END_TOKEN`.
   * Demonstrated with example sentences.

---

## Tips for experiments & scaling

* **Subword tokenization**: For real datasets switch from char-level to BPE / SentencePiece for better generalization.
* **Beam search**: Replace greedy `argmax` decoding with beam search for improved translation quality.
* **Mixed precision & gradient accumulation**: To train larger models on limited GPUs.
* **Optimizer tweaks**: Experiment with AdamW and learning-rate schedules (warmup + decay used in Transformer papers).
* **Checkpointing**: Add `torch.save(model.state_dict(), path)` every N epochs and load with `model.load_state_dict(...)`.

---

## Results examples

The notebook includes quick inference examples such as:

```py
translation = translate("what should we do when the day starts?")
print(translation)

translation = translate("how is this the truth?")
print(translation)
```

(These serve as demonstration of model behavior after training; for production quality results use larger corpora + subword tokenization.)

---

## Known limitations & TODOs

* Current notebook uses character-level vocab — switch to subword for better real-world performance.
* No advanced learning-rate scheduler or regularization beyond dropout.
* Decoding uses greedy search — beam search would improve translation quality.
* Add evaluation metrics (BLEU / chrF) to quantify progress.

---

## Contribution

Contributions welcome! Suggestions:

* Add BPE tokenization with SentencePiece and re-run experiments.
* Implement beam search decoder and compare BLEU scores.
* Add `train.py` and `evaluate.py` scripts to make training CLI-driven (scriptable CI-friendly experiments).

---

## License

MIT License — feel free to use, adapt, and build upon this work for research and education.

---

