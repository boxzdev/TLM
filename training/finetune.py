"""
finetune.py

Continues training an EXISTING checkpoint.bin on a small set of curated
conversations (finetune_data/*.json), instead of the general corpus, to
teach the model actual Q&A turn-taking on top of whatever grammar and
spelling train.py already gave it.

Requires a model that has already been trained with train.py -- fine-
tuning an undertrained/random model has nothing to "align" and just
memorizes a handful of exact sentences.

Data format expected in finetune_data/*.json (already present:
sample_finetune.json):
    [
      {"messages": [{"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}, ...]},
      ...
    ]

Each conversation is flattened into the same "User: ...\\nBot: ...\\n"
text chat.py already builds prompts with. Loss is MASKED: the model is
only trained to predict the assistant's reply characters, not the
user's question or the "Bot: " tag itself -- because chat.py already
supplies "User: ...\\nBot: " as the prompt and only ever asks the model
to generate what comes after it. Training on exactly that matches
inference exactly.

Before changing anything, checkpoint.bin is backed up to
checkpoint.pre_finetune.bin, so this is never a one-way door.
"""

import glob
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
FACTORY_DIR = os.path.join(PROJECT_ROOT, "factory")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FINETUNE_DATA_DIR = os.path.join(PROJECT_ROOT, "finetune_data")

sys.path.insert(0, FACTORY_DIR)
from tokenizer import Tokenizer, list_models  # noqa: E402

DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.0005  # lower than base training -- small data, easy to overfit


def import_from_path(module_name, path):
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
    print("Which model do you want to fine-tune?")
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
    raw = input(f"Epochs to fine-tune [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("  Not a number, using default.")
        return default


# ============================================================================
# Data prep: conversation JSON -> (char ids, loss mask) examples
# ============================================================================

def find_finetune_files():
    if not os.path.isdir(FINETUNE_DATA_DIR):
        return []
    return sorted(glob.glob(os.path.join(FINETUNE_DATA_DIR, "*.json")))


def load_conversations(paths):
    conversations = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            msgs = entry.get("messages", [])
            if msgs:
                conversations.append(msgs)
    return conversations


def build_examples(conversations, tokenizer, max_seq_len):
    """Turns each conversation into one (input_ids, target_ids, loss_mask)
    training example, character by character. mask=1 only on assistant
    reply characters (+ trailing newline); mask=0 everywhere else,
    including the 'Bot: ' tag itself (chat.py always supplies that tag
    as part of the prompt, so the model is never asked to produce it)."""
    examples = []
    skipped = 0
    eos = "<eos>" if getattr(tokenizer, "eos_id", None) is not None else ""

    for msgs in conversations:
        text_parts = []   # list of (substring, mask_value)
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                question = msgs[i]["content"]
                answer = msgs[i + 1]["content"]
                text_parts.append((f"User: {question}\nBot: ", 0))
                text_parts.append((f"{answer}\n{eos}", 1))
                i += 2
            else:
                i += 1

        if not text_parts:
            skipped += 1
            continue

        ids, mask = [], []
        for chunk, m in text_parts:
            chunk_ids = tokenizer.encode(chunk)
            ids.extend(chunk_ids)
            mask.extend([m] * len(chunk_ids))

        if len(ids) < 2 or sum(mask) == 0:
            skipped += 1
            continue

        # Keep the sequence inside the model's context window -- if a
        # conversation is longer than max_seq_len, keep the TAIL (the
        # most recent turn matters most, and it guarantees some
        # assistant content survives truncation more often than keeping
        # the head would for long questions).
        if len(ids) > max_seq_len + 1:
            ids = ids[-(max_seq_len + 1):]
            mask = mask[-(max_seq_len + 1):]
            if sum(mask[1:]) == 0:  # truncation ate all the assistant content
                skipped += 1
                continue

        input_ids = np.array(ids[:-1])
        target_ids = np.array(ids[1:])
        loss_mask = np.array(mask[1:], dtype=np.float32)  # mask aligned to targets
        examples.append((input_ids, target_ids, loss_mask))

    return examples, skipped


# ============================================================================
# Fine-tuning
# ============================================================================

def save_checkpoint(model, checkpoint_path, meta_path, meta):
    model.save(checkpoint_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def finetune(model_name):
    model_dir = os.path.join(MODELS_DIR, model_name)
    vocab_path = os.path.join(model_dir, "tokenize_vocab.json")
    config_path = os.path.join(model_dir, "config.py")
    arch_path = os.path.join(model_dir, "architecture.py")
    checkpoint_path = os.path.join(model_dir, "checkpoint.bin")
    backup_path = os.path.join(model_dir, "checkpoint.pre_finetune.bin")
    meta_path = os.path.join(model_dir, "train_meta.json")

    if not os.path.exists(checkpoint_path):
        sys.exit(f"No checkpoint.bin for '{model_name}' yet. Run train.py first -- "
                  f"fine-tuning needs a model that already knows the language.")

    finetune_files = find_finetune_files()
    if not finetune_files:
        sys.exit(f"No .json files found in {FINETUNE_DATA_DIR}/.")

    config = import_from_path(f"config_{model_name}", config_path)
    architecture = import_from_path(f"architecture_{model_name}", arch_path)
    tokenizer = Tokenizer.load(vocab_path)

    model = architecture.TinyTransformer.load(checkpoint_path, device=config.DEVICE)
    print(f"Loaded '{model_name}' ({model.total_params:,} params) for fine-tuning.")

    shutil.copy(checkpoint_path, backup_path)
    print(f"Backed up pre-finetune weights to {backup_path}")

    conversations = load_conversations(finetune_files)
    examples, skipped = build_examples(conversations, tokenizer, model.max_seq_len)
    if not examples:
        sys.exit("No usable examples after processing finetune_data/. Check the format.")
    print(f"Loaded {len(examples)} conversation example(s) from {len(finetune_files)} file(s)"
          f"{f' ({skipped} skipped)' if skipped else ''}.")

    epochs = ask_epochs(DEFAULT_EPOCHS)
    rng = np.random.default_rng()

    print("\nFine-tuning -- Ctrl+C any time to stop and save.\n")

    interrupted = False
    total_steps = 0
    last_loss = None
    try:
        for epoch in range(1, epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            epoch_loss = 0.0
            active_tokens = 0
            for idx in order:
                input_ids, target_ids, loss_mask = examples[idx]
                loss, grads = model.loss_and_grads(input_ids, target_ids, loss_mask=loss_mask)
                model.update(grads, learning_rate=DEFAULT_LEARNING_RATE)
                epoch_loss += loss
                active_tokens += max(int(loss_mask.sum()), 1)
                total_steps += 1
                last_loss = loss

            avg = epoch_loss / active_tokens
            print(f"  epoch {epoch}/{epochs}  avg loss/answer-char={avg:.3f}  total_steps={total_steps}")

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
        "fine_tuned": True,
        "fine_tune_examples": len(examples),
    }
    save_checkpoint(model, checkpoint_path, meta_path, meta)
    print(f"Saved {checkpoint_path}")
    print(f"Saved {meta_path}")
    print(f"\nOriginal pre-finetune weights are still at {backup_path} if you want to revert.")
    print("Fine-tuning stopped early but progress is saved." if interrupted else "Fine-tuning complete.")


def main():
    print("=" * 52)
    print(" TLM Fine-tuner")
    print("=" * 52)
    model_name = ask_model()
    finetune(model_name)


if __name__ == "__main__":
    main()
