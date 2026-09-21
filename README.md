# TLM -- Tiny Language Model

A small language model built entirely from scratch. The neural network,
every backward-pass gradient, the training loop, checkpoint saving/loading,
and the chat loop are all hand-written -- no PyTorch, no TensorFlow, no ML
framework of any kind. NumPy (or CuPy/cugpu on a GPU) is used purely as
fast array math; nothing about the model itself comes from a library.

## What it is

A **decoder-only Transformer** (GPT-style: masked self-attention + a
feed-forward network, no encoder, no cross-attention), trained with
next-character prediction:

```
Output Embedding -> (+) Positional Encoding
  -> [ Masked Multi-Head Attention -> Add & Norm
       -> Feed Forward -> Add & Norm ] x Nx
  -> Linear -> Softmax -> Output Probabilities
```

Paired with a **pure character-level tokenizer** -- one token per
character, no words, no merges. The model has no shortcuts: letters
combining into real words, spacing, and punctuation is exactly what
millions of repetitions of "given the last N characters, predict the
next one" teaches it, nothing more.

Every forward computation in `architecture.py` has a matching
hand-derived backward computation, and it's been verified correct with
numerical gradient checking (not just written and assumed to work).

What it is **not**: a ChatGPT-style assistant. It has no world
knowledge, no real reasoning, and a modest context window. Trained on a
few KB-MB of text, it produces output that mimics the *style and
patterns* of that text -- treat it as a transparent way to see how a
language model actually works, from tokenizer to weights to backprop.

## Compute backends

- **CPU (NumPy)** -- works everywhere, no setup. Fine for the small
  models this project is built around.
- **cugpu (AMD / Intel / NVIDIA)** -- a custom GPU backend built on
  ILGPU/OpenCL, for GPUs CuPy can't reach. Build it via
  `GPU_factory/build_package.bat` -- see `GPU_factory/README.md`.
- **CuPy (NVIDIA CUDA)** -- `pip install cupy-cuda12x` (see
  `requirements-gpu.txt`).

Pick one when `factory_TLM.py` asks; if nothing GPU-related is
installed, everything just runs on CPU automatically.

## The factory workflow

TLM works as a **factory**: you name and create models, pick an
architecture size, tokenize them, train them, optionally fine-tune
them, and chat with them -- each one living in its own folder under
`models/`, side by side.

```
1. python factory/factory_TLM.py     Create a model: name -> architecture
                                      size (Tiny/Small/Medium/Custom) ->
                                      compute backend. Scaffolds
                                      models/<name>/ with architecture.py,
                                      chat.py, and a filled-in config.py.
                                      No weights are created yet --
                                      vocab_size isn't known until step 2.

2. python factory/tokenizer.py       Scans data/*.txt, builds a
                                      character-level vocab (every
                                      character actually found in the
                                      data, plus a full-ASCII safety
                                      floor), saves
                                      models/<name>/tokenize_vocab.json.

3. python training/train.py          Builds the model's weights for the
                                      first time (or resumes an existing
                                      checkpoint.bin) and trains it on
                                      data/*.txt via next-character
                                      prediction. Ctrl+C-safe: interrupting
                                      saves checkpoint.bin + train_meta.json
                                      just like finishing normally does.

4. python training/finetune.py       Optional. Continues an already-
                                      trained checkpoint on the small,
                                      curated conversations in
                                      finetune_data/*.json, to teach it
                                      actual Q&A turn-taking instead of
                                      free-running text continuation.
                                      Loss is masked to only the
                                      assistant's reply characters.
                                      Backs up checkpoint.bin to
                                      checkpoint.pre_finetune.bin first.

5. python models/<name>/chat.py      Terminal chat. Imports
                                      architecture.py directly and drives
                                      its forward pass token by token --
                                      no GUI, no dependency on the rest
                                      of the project.
```

## Folder structure

```
TLM/
├── factory/
│   ├── factory_TLM.py          # creates new models (name/size/backend)
│   ├── tokenizer.py             # builds each model's character vocab
│   └── template/                # copied into every new models/<name>/
│       ├── architecture.py      #   the model itself (single source of truth)
│       ├── chat.py              #   terminal inference, imports architecture.py
│       └── config.py            #   per-model hyperparameters (template)
├── GPU_factory/                 # custom AMD/Intel/NVIDIA GPU backend (cugpu)
├── data/                        # shared training text (.txt)
├── finetune_data/                # shared Q&A fine-tuning data (.json)
├── models/<name>/                # one self-contained folder per model you create
│   ├── architecture.py, chat.py, config.py
│   ├── tokenize_vocab.json       # from tokenizer.py
│   ├── checkpoint.bin             # from train.py / finetune.py
│   └── train_meta.json            # step count, last loss, fine-tune status
├── training/
│   ├── train.py                  # base training
│   └── finetune.py                # Q&A fine-tuning (masked loss)
└── README.md
```

## Install

```
pip install -r requirements.txt          # CPU only, all you need to get started
pip install -r requirements-gpu.txt      # + NVIDIA CUDA via CuPy
```

For AMD/Intel GPU support instead, see `GPU_factory/README.md`.

## Quickstart

```
python factory/factory_TLM.py     # create a model
python factory/tokenizer.py       # build its vocab from data/
python training/train.py          # train it
python models/<name>/chat.py      # talk to it
```
