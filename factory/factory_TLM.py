"""
factory_TLM.py

Scaffolds new TLM models. No neural-net math lives here -- that's all
in architecture.py. This script only:

    1. asks for a model name
    2. asks for an architecture size (preset or custom)
    3. asks for a compute backend
    4. creates models/<name>/ and drops in architecture.py + chat.py
       (copied verbatim from factory/template/) plus a filled-in config.py

vocab_size is deliberately NOT decided here -- it isn't known until
tokenizer.py has run on the training data. No checkpoint.bin is created
at this stage; train.py builds and saves the initial weights the first
time it runs for this model.
"""

import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

sys.path.insert(0, TEMPLATE_DIR)
from architecture import calculate_model_parameters, prompt_device  # noqa: E402

PRESETS = {
    "1": dict(label="Tiny   -- fast, small data",
              d_model=64, num_layers=2, num_heads=2, d_ff=256, max_seq_len=64),
    "2": dict(label="Small  -- default, good general starting point",
              d_model=128, num_layers=4, num_heads=4, d_ff=512, max_seq_len=128),
    "3": dict(label="Medium -- bigger data, slower to train",
              d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_seq_len=256),
}


# ============================================================================
# Prompts
# ============================================================================

def ask_model_name():
    while True:
        name = input("Model name: ").strip()
        if not name:
            print("  Name can't be empty.")
            continue
        if not all(c.isalnum() or c in "-_" for c in name):
            print("  Use only letters, numbers, '-' and '_'.")
            continue
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest):
            confirm = input(f"  '{name}' already exists -- overwrite it? [y/N] ").strip().lower()
            if confirm != "y":
                continue
        return name


def ask_architecture():
    print("\nArchitecture size:")
    for key, p in PRESETS.items():
        print(f"  {key}. {p['label']}")
        print(f"       d_model={p['d_model']}, layers={p['num_layers']}, "
              f"heads={p['num_heads']}, d_ff={p['d_ff']}, max_seq_len={p['max_seq_len']}")
    print("  4. Custom")
    choice = input("> ").strip() or "2"

    if choice in PRESETS:
        return dict(PRESETS[choice])

    def ask_int(prompt, default):
        raw = input(f"  {prompt} [{default}]: ").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            print("  Not a number, using default.")
            return default

    d_model = ask_int("d_model", 128)
    num_heads = ask_int("num_heads", 4)
    while d_model % num_heads != 0:
        print(f"  d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        num_heads = ask_int("num_heads", 4)
    num_layers = ask_int("num_layers", 4)
    d_ff = ask_int("d_ff", d_model * 4)
    max_seq_len = ask_int("max_seq_len", 128)
    return dict(d_model=d_model, num_heads=num_heads, num_layers=num_layers,
                d_ff=d_ff, max_seq_len=max_seq_len)


def preview_params(arch, vocab_estimate=2000):
    total = calculate_model_parameters(
        arch['d_model'], vocab_estimate, arch['num_layers'],
        arch['num_heads'], arch['d_ff'])
    print(f"\nEstimated parameters at a ~{vocab_estimate}-token vocab: ~{total:,}")
    print("  (Final count depends on the real vocab size from tokenizer.py.)")


# ============================================================================
# Scaffolding
# ============================================================================

def create_model(name, arch, device):
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)

    shutil.copy(os.path.join(TEMPLATE_DIR, "architecture.py"), dest)
    shutil.copy(os.path.join(TEMPLATE_DIR, "chat.py"), dest)

    with open(os.path.join(TEMPLATE_DIR, "config.py"), "r") as f:
        config_text = f.read()

    replacements = {
        "__MODEL_NAME__": name,
        "__D_MODEL__": str(arch["d_model"]),
        "__NUM_LAYERS__": str(arch["num_layers"]),
        "__NUM_HEADS__": str(arch["num_heads"]),
        "__D_FF__": str(arch["d_ff"]),
        "__MAX_SEQ_LEN__": str(arch["max_seq_len"]),
        "__DEVICE__": device,
        "__SEQ_LENGTH__": str(min(arch["max_seq_len"], 64)),
        "__BATCH_SIZE__": "16",
        "__LEARNING_RATE__": "0.001",
        "__EPOCHS__": "10",
    }
    for placeholder, value in replacements.items():
        config_text = config_text.replace(placeholder, value)

    with open(os.path.join(dest, "config.py"), "w") as f:
        f.write(config_text)

    print(f"\nCreated models/{name}/")
    print("  architecture.py  (the model, imported not duplicated)")
    print("  chat.py          (terminal inference, connects to architecture.py)")
    print("  config.py        (this model's hyperparameters)")
    print("\nNext steps:")
    print(f"  1. Run tokenizer.py to build models/{name}/tokenize_vocab.json")
    print(f"  2. Run train.py to build and train models/{name}/checkpoint.bin")
    print(f"  3. Run models/{name}/chat.py to talk to it")


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 52)
    print(" TLM Factory -- create a new model")
    print("=" * 52)

    os.makedirs(MODELS_DIR, exist_ok=True)
    existing = sorted(d for d in os.listdir(MODELS_DIR)
                       if os.path.isdir(os.path.join(MODELS_DIR, d)))
    if existing:
        print(f"Existing models: {', '.join(existing)}")

    name = ask_model_name()
    arch = ask_architecture()
    preview_params(arch)
    device = prompt_device()
    create_model(name, arch, device)


if __name__ == "__main__":
    main()
