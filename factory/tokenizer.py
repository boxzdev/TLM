"""
tokenizer.py

Pure character-level tokenizer for TLM. One token = one character.
No words, no merges, no frequency ranking -- the model has to learn
letters combining into words purely from next-character prediction
during training.

Produces models/<name>/tokenize_vocab.json in the same
{"vocab": {token: id, ...}} shape chat.py's SimpleTokenizer already
reads, so a model trained with this vocab file works with chat.py
unmodified.

Run directly for the menu:
    python tokenizer.py
"""

import json
import os
import string
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Guarantees full coverage even for characters the training data happens
# not to contain, so chat.py never silently drops a character someone
# types later.
# Special tokens always occupy ids 0..3, ahead of every character.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

SAFETY_CHARS = string.ascii_letters + string.digits + string.punctuation + " \n\t"


# ============================================================================
# Tokenizer
# ============================================================================

class Tokenizer:
    """One token per character. train.py imports this class directly;
    chat.py carries its own tiny self-contained copy of the same idea
    on purpose, so a trained model stays a shareable, dependency-free
    5-file package."""

    def __init__(self, vocab=None):
        self.token_to_id = vocab or {}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        # Old vocab files without special tokens still load fine.
        self.specials = [t for t in SPECIAL_TOKENS if t in self.token_to_id]

    pad_id = property(lambda self: self.token_to_id.get("<pad>"))
    unk_id = property(lambda self: self.token_to_id.get("<unk>"))
    bos_id = property(lambda self: self.token_to_id.get("<bos>"))
    eos_id = property(lambda self: self.token_to_id.get("<eos>"))

    @classmethod
    def build(cls, text, extra_chars=SAFETY_CHARS):
        chars = set(text) | set(extra_chars)
        # Sorted for a deterministic, reproducible vocab (same text always
        # produces the same token ids across runs/machines).
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in sorted(chars):
            vocab[ch] = len(vocab)
        return cls(vocab)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["vocab"] if "vocab" in data else data
        return cls(vocab)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "vocab": self.token_to_id},
                       f, indent=2, ensure_ascii=False)

    def encode(self, text, parse_special=True):
        # Special-token strings like "<eos>" become a single id. Unknown
        # characters map to <unk> (or are skipped if the vocab has none).
        ids, i, n = [], 0, len(text)
        specials = self.specials if parse_special else []
        while i < n:
            if text[i] == "<":
                tok = next((t for t in specials if text.startswith(t, i)), None)
                if tok:
                    ids.append(self.token_to_id[tok])
                    i += len(tok)
                    continue
            ch = text[i]
            i += 1
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            elif self.unk_id is not None:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids, skip_special=False):
        out = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if skip_special and t in self.specials:
                continue
            out.append(t)
        return "".join(out)

    @property
    def vocab_size(self):
        return len(self.token_to_id)


# ============================================================================
# Data loading
# ============================================================================

def find_data_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt"))


def load_corpus(filenames):
    chunks = []
    for name in filenames:
        path = os.path.join(DATA_DIR, name)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())
    return "\n".join(chunks)


# ============================================================================
# Menu
# ============================================================================

def list_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(d for d in os.listdir(MODELS_DIR)
                  if os.path.isdir(os.path.join(MODELS_DIR, d)))


def ask_model():
    models = list_models()
    if not models:
        sys.exit("No models found. Run factory_TLM.py first to create one.")
    print("Which model is this vocab for?")
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


def ask_data_files():
    files = find_data_files()
    if not files:
        sys.exit(f"No .txt files found in {DATA_DIR}/. Add training text there first.")
    print(f"\nFound {len(files)} text file(s) in data/: {', '.join(files)}")
    choice = input("Use all of them? [Y/n] ").strip().lower()
    if choice in ("", "y"):
        return files
    print("Enter filenames to use, comma-separated:")
    chosen = [f.strip() for f in input("> ").split(",") if f.strip()]
    invalid = [f for f in chosen if f not in files]
    if invalid:
        sys.exit(f"Not found in data/: {', '.join(invalid)}")
    return chosen or files


def report(tokenizer, corpus):
    print(f"\nVocab size: {tokenizer.vocab_size} characters")
    preview = "".join(sorted(tokenizer.token_to_id, key=lambda c: tokenizer.token_to_id[c]))
    preview = preview.replace("\n", "\\n").replace("\t", "\\t")
    print(f"Characters: {preview[:120]}{'...' if len(preview) > 120 else ''}")

    # Live round-trip proof, not just a claim -- encode/decode the real
    # corpus and confirm it comes back out exactly.
    sample = corpus[:5000]
    ids = tokenizer.encode(sample)
    back = tokenizer.decode(ids)
    ok = back == sample
    print(f"Round-trip check on {len(sample):,} chars of the real corpus: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        for i, (a, b) in enumerate(zip(sample, back)):
            if a != b:
                print(f"  First mismatch at position {i}: {a!r} != {b!r}")
                break


def main():
    print("=" * 52)
    print(" TLM Tokenizer -- build a character-level vocab")
    print("=" * 52)

    model_name = ask_model()
    filenames = ask_data_files()
    corpus = load_corpus(filenames)
    print(f"\nLoaded {len(corpus):,} characters from {len(filenames)} file(s).")

    tokenizer = Tokenizer.build(corpus)
    report(tokenizer, corpus)

    dest = os.path.join(MODELS_DIR, model_name, "tokenize_vocab.json")
    tokenizer.save(dest)
    print(f"\nSaved {dest}")
    print(f"\nNext step: run train.py for '{model_name}'.")


if __name__ == "__main__":
    main()
