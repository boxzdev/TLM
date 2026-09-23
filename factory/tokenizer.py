"""
tokenizer.py

Word-aware BPE tokenizer for TLM.

Plain character-level BPE on a small corpus tends to merge meaningless
pairs ("ui", "ny", "thi") before it ever reaches real words, because
those pairs repeat by coincidence often enough on a few dozen KB of
text. This version avoids that three ways:
  1. Merges never cross a whitespace boundary (learned per-word).
  2. A merge needs to occur BPE_MIN_FREQ+ times to be kept.
  3. A short list of common English/TLM words is seeded into the vocab
     directly, so real words are guaranteed even if the corpus is too
     small for BPE to discover them on its own.

Falls back to individual characters for anything not covered, so
encode/decode is still lossless for any text made of known characters.

Produces models/<name>/tokenize_vocab.json in the same
{"vocab": {token: id, ...}} shape chat.py's SimpleTokenizer already
reads (it does greedy longest-match, which works for a mix of
characters, words, and subword pieces without any change on its end).

Run directly for the menu:
    python tokenizer.py
"""

import json
import os
import re
import string
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# Guarantees full coverage even for characters the training data happens
# not to contain, so chat.py never silently drops a character someone
# types later.
def normalize_prompt(text):
    """Lowercase, drop punctuation, collapse spaces, so "Hello!!" and
    "hello" become the same prompt."""
    return " ".join(re.sub(r"[^\w' ]+", " ", text.lower()).split())


# Special tokens always occupy ids 0..3, ahead of every character.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

SAFETY_CHARS = string.ascii_letters + string.digits + string.punctuation + " \n\t"

# --- BPE settings ---------------------------------------------------------
# 16k is a real subword vocab size (GPT-2 uses ~50k on a vastly bigger
# corpus) -- big enough that most common words get their own token
# instead of being split into pieces, once there's enough data to earn
# them. BPE_MIN_FREQ scales with it: a bigger vocab means digging deeper
# into rarer pairs, so raise the bar a little to still avoid junk merges.
BPE_VOCAB_SIZE = 16000   # total tokens: specials + characters + words + merges
BPE_MIN_FREQ = 10        # a merge must occur at least this many times to count
BPE_MAX_SEED_WORDS = 350

# Guaranteed whole-word tokens, so common words exist even if the corpus
# is too small for BPE to find them by frequency alone.
COMMON_WORDS = """
the a an is are was were be been being am i you he she it we they
this that these those my your his her its our their me him us them
and or but if so because when while as of in on at to from for with
without into onto over under about above below through during before
after between around not no yes do does did done can could will would
should shall may might must have has had do not don't isn't aren't
what who whom which why where how here there now then today tomorrow
yesterday hello hi hey bye goodbye please thanks thank you sorry okay
ok good bad great nice fine well well done yes no maybe sure alright
name is my what your who are you how old where live what can do help
know think feel like love hate want need go come see look say tell
ask answer make take give get put find use work play read write learn
teach show hear listen speak talk understand remember forget try start
stop continue open close begin end wake sleep eat drink walk run sit
stand happy sad angry tired bored excited scared surprised worried
calm proud afraid brave kind funny smart silly friendly warm cold hot
big small tall short long short new old young fast slow easy hard
right wrong true false real fake same different better best worse
worst more most less least many few some any all none every each
other another one two three four five six seven eight nine ten
hundred thousand million zero first second third last next monday
tuesday wednesday thursday friday saturday sunday january february
march april may june july august september october november december
morning afternoon evening night today week month year time day hour
minute second computer program code data model train learn network
neural transformer attention token embedding vocabulary vocab special
checkpoint parameter layer weight bias epoch gradient loss function
language artificial intelligence machine robot chat bot tlm
""".split()

COMMON_WORDS = list(dict.fromkeys(COMMON_WORDS))[:BPE_MAX_SEED_WORDS]


# ============================================================================
# BPE training (word-frequency based, never crosses a whitespace boundary)
# ============================================================================

def _word_freqs(text):
    freqs = {}
    for m in re.finditer(r"\S+", text):
        w = m.group(0)
        freqs[w] = freqs.get(w, 0) + 1
    return freqs


def _word_pairs(symbols):
    """All adjacent symbol pairs in one word's current split."""
    return zip(symbols, symbols[1:])


def _apply_merge(symbols, pair, merged):
    """Returns a new symbol list with every non-overlapping occurrence
    of `pair` collapsed into the single symbol `merged`."""
    new_symbols, i, n = [], 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            new_symbols.append(merged)
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols


def _learn_merges(word_freqs, budget, min_freq):
    """Returns a list of merged substrings, most-frequent-pair-first,
    stopping once `budget` merges are made or no pair repeats often
    enough (>= min_freq) to be worth keeping.

    The naive way to do this -- rescan every word in the corpus to
    recount every pair, on every single merge -- is fine for a few
    hundred merges over a few dozen KB, but it doesn't scale: at 16k
    merges over a real (multi-MB+) corpus it would mean re-touching the
    entire vocabulary tens of thousands of times. Real BPE implementations
    avoid that by tracking, for each pair, exactly which words contain it
    (`pair_to_words`), so a merge only has to re-examine the handful of
    words it actually affects, not the whole corpus every time."""
    splits = {w: list(w) for w in word_freqs}
    pair_counts = {}
    pair_to_words = {}

    def touch(word, delta):
        """Add (or remove, if delta is -1) this word's current pairs
        to/from the running counts, keyed by the word's own frequency."""
        freq = word_freqs[word]
        for pair in _word_pairs(splits[word]):
            pair_counts[pair] = pair_counts.get(pair, 0) + delta * freq
            if delta > 0:
                pair_to_words.setdefault(pair, set()).add(word)

    for w in word_freqs:
        touch(w, +1)

    merges = []
    while len(merges) < budget and pair_counts:
        best_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_pair] < min_freq:
            break
        merged = best_pair[0] + best_pair[1]
        merges.append(merged)

        affected = pair_to_words.pop(best_pair, ())
        for w in affected:
            touch(w, -1)                          # remove this word's old pairs
            splits[w] = _apply_merge(splits[w], best_pair, merged)
            touch(w, +1)                           # re-add its new pairs

        pair_counts.pop(best_pair, None)           # fully consumed, never re-pick it
    return merges


# ============================================================================
# Tokenizer
# ============================================================================

class Tokenizer:
    """Greedy longest-match over a flat {token: id} vocab of specials,
    characters, seeded words, and learned subword merges. train.py
    imports this class directly; chat.py carries its own tiny
    self-contained copy of the same matching logic on purpose, so a
    trained model stays a shareable, dependency-free 5-file package."""

    def __init__(self, vocab=None):
        self.token_to_id = vocab or {}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        # Old vocab files without special tokens still load fine.
        self.specials = [t for t in SPECIAL_TOKENS if t in self.token_to_id]
        self.max_token_len = max((len(t) for t in self.token_to_id), default=1)
        # For each starting character, which token lengths actually exist.
        # At 16k tokens, trying every length from max_token_len down to 1
        # at every text position (most of which have no real candidate)
        # gets slow over a real corpus -- this lets encode() only try
        # lengths that could possibly match.
        self._lens_by_first_char = {}
        for t in self.token_to_id:
            if t:
                self._lens_by_first_char.setdefault(t[0], set()).add(len(t))
        self._lens_by_first_char = {
            c: sorted(lens, reverse=True)
            for c, lens in self._lens_by_first_char.items()
        }

    pad_id = property(lambda self: self.token_to_id.get("<pad>"))
    unk_id = property(lambda self: self.token_to_id.get("<unk>"))
    bos_id = property(lambda self: self.token_to_id.get("<bos>"))
    eos_id = property(lambda self: self.token_to_id.get("<eos>"))

    @classmethod
    def build(cls, text, extra_chars=SAFETY_CHARS, vocab_size=BPE_VOCAB_SIZE,
              min_freq=BPE_MIN_FREQ, seed_words=COMMON_WORDS):
        # 1. specials + full character coverage (guarantees lossless fallback)
        chars = set(text) | set(extra_chars)
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in sorted(chars):
            vocab[ch] = len(vocab)

        # 2. seed common words directly, so real words exist even if the
        #    corpus is too small for BPE to find them by frequency alone
        for w in seed_words:
            if len(vocab) >= vocab_size:
                break
            if w not in vocab and all(c in vocab for c in w):
                vocab[w] = len(vocab)

        # 3. spend whatever budget is left on data-driven merges, learned
        #    per-word so a merge never crosses a whitespace boundary
        budget = max(0, vocab_size - len(vocab))
        if budget:
            for m in _learn_merges(_word_freqs(text), budget, min_freq):
                if m not in vocab and len(vocab) < vocab_size:
                    vocab[m] = len(vocab)

        return cls(vocab)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["vocab"] if "vocab" in data else data
        return cls(vocab)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "vocab": self.token_to_id},
                       f, indent=2, ensure_ascii=False)

    def encode(self, text, parse_special=True):
        vocab = self.token_to_id
        skip = self.specials if not parse_special else ()
        ids, i, n = [], 0, len(text)
        while i < n:
            token_id = None
            for match_len in self._lens_by_first_char.get(text[i], ()):
                if match_len > n - i:
                    continue
                piece = text[i:i + match_len]
                if piece in vocab and piece not in skip:
                    token_id = vocab[piece]
                    i += match_len
                    break
            if token_id is None:
                if self.unk_id is not None:
                    ids.append(self.unk_id)
                i += 1
                continue
            ids.append(token_id)
        return ids

    def encode_to_array(self, text, parse_special=True):
        """Same tokenization as encode(), but packed into a NumPy array
        instead of a Python list -- a Python int in a list costs ~28+
        bytes; a uint16/uint32 array element costs 2-4. That's the
        difference between a large corpus fitting comfortably in memory
        or not. Used by train.py when loading the full training corpus;
        encode() (a plain list) stays the default for short chat prompts
        and finetune examples, where a NumPy array would just be overhead."""
        ids = self.encode(text, parse_special=parse_special)
        dtype = np.uint16 if self.vocab_size <= 65536 else np.uint32
        return np.array(ids, dtype=dtype)

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
    if len(models) == 1:
        print(f"Using model: {models[0]}")
        return models[0]
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
    chars = [t for t in tokenizer.token_to_id if len(t) == 1 and t not in tokenizer.specials]
    words = [t for t in tokenizer.token_to_id if len(t) > 1]
    print(f"\nVocab size: {tokenizer.vocab_size} tokens "
          f"({len(tokenizer.specials)} special, {len(chars)} characters, "
          f"{len(words)} words/subwords)")
    sample_words = sorted(words, key=len, reverse=True)[:20]
    print(f"Longest tokens learned: {', '.join(sample_words)}")

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
    compression = len(sample) / max(len(ids), 1)
    print(f"Average {compression:.2f} characters per token on that sample.")


def main():
    print("=" * 52)
    print(" TLM Tokenizer -- build a word-aware BPE vocab")
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
