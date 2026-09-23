"""
train.py

Trains a TLM model. Builds a fresh TinyTransformer (reading vocab_size
from tokenize_vocab.json and hyperparameters from config.py) or resumes
an existing checkpoint.bin, then sweeps the training corpus in
SEQ_LENGTH-token chunks calling loss_and_grads() + update() (Adam) from
architecture.py every step.

The task is always the same: given characters 0..t, predict character
t+1. There's no word-level supervision anywhere in this file -- letters
combining into real words is exactly what repeating that task teaches
the model, nothing more.

Ctrl+C-safe: interrupting mid-epoch saves checkpoint.bin + train_meta.json
exactly like reaching the end normally does, so you never lose progress.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
FACTORY_DIR = os.path.join(PROJECT_ROOT, "factory")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

sys.path.insert(0, FACTORY_DIR)
from tokenizer import Tokenizer, find_data_files, load_corpus, list_models  # noqa: E402


def import_from_path(module_name, path):
    """Loads architecture.py/config.py straight from a specific model's
    folder (not via sys.path), so each model's own copy is always the
    one actually used, even if the templates change later."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Menu helpers
# ============================================================================

def ask_model():
    models = list_models()
    if not models:
        sys.exit("No models found. Run factory_TLM.py first to create one.")
    if len(models) == 1:
        print(f"Using model: {models[0]}")
        return models[0]
    print("Which model do you want to train?")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input("> ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print("  Enter a number from the list.")


def ask_epochs(default):
    raw = input(f"Epochs to train [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("  Not a number, using default.")
        return default


# ============================================================================
# Data prep
# ============================================================================

def build_chunks(ids, seq_length):
    """Non-overlapping (inputs, targets) windows sweeping the whole corpus."""
    chunks = []
    i = 0
    while i + seq_length + 1 <= len(ids):
        chunks.append((ids[i:i + seq_length], ids[i + 1:i + seq_length + 1]))
        i += seq_length
    return chunks


def batch_iter(chunks, order, batch_size):
    """Groups shuffled chunk indices batch_size at a time and stacks each
    group into (B, seq_length) arrays -- this is what actually lets the
    GPU (or CPU) work on many sequences per step instead of one. Every
    chunk from build_chunks is the same seq_length, so stacking needs no
    padding. The last batch in an epoch may come out smaller if the chunk
    count doesn't divide evenly -- that's fine, loss_and_grads handles any
    batch size the same way."""
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        inputs = np.stack([chunks[i][0] for i in idxs])
        targets = np.stack([chunks[i][1] for i in idxs])
        yield inputs, targets


def save_checkpoint(model, checkpoint_path, meta_path, meta):
    model.save(checkpoint_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def make_preview_fn(architecture_mod):
    """Small live sample of what the model currently writes, using that
    model's own architecture.py forward pass directly -- not a canned
    string, an actual generation."""
    softmax = architecture_mod.softmax

    def preview(model, tokenizer, rng, max_new_tokens=60, seed_text="\n"):
        seed_ids = tokenizer.encode(seed_text) or [0]
        ids = list(seed_ids)
        out_ids = []
        for _ in range(max_new_tokens):
            window = ids[-model.max_seq_len:]
            logits, _, _ = model.forward(np.array(window))
            probs = softmax(logits[-1] / 0.8)
            next_id = int(rng.choice(model.vocab_size, p=probs))
            ids.append(next_id)
            out_ids.append(next_id)
        return tokenizer.decode(out_ids).replace("\n", " / ")

    return preview


# ============================================================================
# Training
# ============================================================================

def train(model_name):
    model_dir = os.path.join(MODELS_DIR, model_name)
    vocab_path = os.path.join(model_dir, "tokenize_vocab.json")
    config_path = os.path.join(model_dir, "config.py")
    arch_path = os.path.join(model_dir, "architecture.py")
    checkpoint_path = os.path.join(model_dir, "checkpoint.bin")
    meta_path = os.path.join(model_dir, "train_meta.json")

    if not os.path.exists(vocab_path):
        sys.exit(f"No tokenize_vocab.json for '{model_name}'. Run tokenizer.py first.")
    if not os.path.exists(config_path) or not os.path.exists(arch_path):
        sys.exit(f"'{model_name}' is missing config.py/architecture.py. Recreate it with factory_TLM.py.")

    config = import_from_path(f"config_{model_name}", config_path)
    architecture = import_from_path(f"architecture_{model_name}", arch_path)

    tokenizer = Tokenizer.load(vocab_path)
    print(f"Loaded vocab: {tokenizer.vocab_size} characters")

    data_files = find_data_files()
    if not data_files:
        sys.exit("No .txt files found in data/. Add training text first.")
    corpus = load_corpus(data_files)
    # encode_to_array (not encode()) -- a NumPy uint16/uint32 array instead
    # of a Python list of ints, which matters once the corpus is large: a
    # Python int in a list costs 28+ bytes, an array element costs 2-4.
    ids = tokenizer.encode_to_array(corpus)
    print(f"Corpus: {len(corpus):,} characters -> {len(ids):,} tokens")

    resume = os.path.exists(checkpoint_path)
    if resume:
        print(f"Existing checkpoint found for '{model_name}'.")
        print("  1. Resume training")
        print("  2. Start fresh (discards current progress)")
        if input("> ").strip() == "2":
            resume = False

    model = None
    if resume:
        model = architecture.TinyTransformer.load(checkpoint_path, device=config.DEVICE)
        if model.vocab_size != tokenizer.vocab_size:
            print("Vocab changed since the last checkpoint -- starting fresh.")
            model, resume = None, False
        else:
            print(f"Resumed model ({model.total_params:,} params).")
    if model is None:
        model = architecture.TinyTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.D_MODEL,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            d_ff=config.D_FF,
            max_seq_len=config.MAX_SEQ_LEN,
            device=config.DEVICE,
        )
        print(f"Built new model ({model.total_params:,} params).")

    prev_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev_meta = json.load(f)
    total_steps = prev_meta.get("total_steps", 0)
    last_loss = prev_meta.get("last_loss")

    epochs = ask_epochs(config.EPOCHS)
    seq_length = min(config.SEQ_LENGTH, model.max_seq_len - 1)
    batch_size = max(1, getattr(config, "BATCH_SIZE", 16))
    chunks = build_chunks(ids, seq_length)
    if not chunks:
        sys.exit("Corpus is too short for the configured SEQ_LENGTH.")
    batches_per_epoch = -(-len(chunks) // batch_size)  # ceil div
    print(f"{len(chunks)} training chunks/epoch, seq_length={seq_length}, "
          f"batch_size={batch_size} -> {batches_per_epoch} steps/epoch")

    rng = np.random.default_rng()
    preview = make_preview_fn(architecture)
    print_every = max(1, batches_per_epoch // 5)

    print("\nTraining -- Ctrl+C any time to stop and save.\n")

    interrupted = False
    try:
        for epoch in range(1, epochs + 1):
            order = list(range(len(chunks)))
            rng.shuffle(order)
            epoch_loss = 0.0
            for step_i, (inputs, targets) in enumerate(batch_iter(chunks, order, batch_size), 1):
                loss, grads = model.loss_and_grads(inputs, targets)
                model.update(grads, learning_rate=config.LEARNING_RATE)
                epoch_loss += loss
                total_steps += 1
                last_loss = loss

                if step_i % print_every == 0 or step_i == batches_per_epoch:
                    avg = epoch_loss / step_i
                    print(f"  epoch {epoch}/{epochs}  step {step_i}/{batches_per_epoch}  "
                          f"loss={loss:.3f}  avg={avg:.3f}  total_steps={total_steps}")

            sample = preview(model, tokenizer, rng)
            print(f"  [epoch {epoch} sample] {sample!r}\n")

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- saving before exit...")

    meta = {
        "model_name": model_name,
        "total_steps": total_steps,
        "last_loss": float(last_loss) if last_loss is not None else None,
        "vocab_size": tokenizer.vocab_size,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interrupted": interrupted,
    }
    save_checkpoint(model, checkpoint_path, meta_path, meta)
    print(f"Saved {checkpoint_path}")
    print(f"Saved {meta_path}")
    print("\nTraining stopped early but progress is saved. Run train.py again to resume."
          if interrupted else "\nTraining complete.")


def main():
    print("=" * 52)
    print(" TLM Trainer")
    print("=" * 52)
    model_name = ask_model()
    train(model_name)


if __name__ == "__main__":
    main()
