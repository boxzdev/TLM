"""
chat.py - Terminal chat interface for a trained TLM model.

Connects DIRECTLY to architecture.py: it imports TinyTransformer (the
decoder-only Transformer defined there) and drives it token-by-token
through its own hand-written forward pass. There is no GUI layer
anymore (gui.py was removed) -- this is the whole interface.

Expected files, sitting next to this script in the same model folder:
    architecture.py      <- the model definition (imported directly)
    checkpoint.bin        <- weights saved by TinyTransformer.save()
    tokenize_vocab.json   <- {"vocab": {token_string: token_id, ...}}
                             produced by the (still-to-be-rewritten)
                             tokenizer.py

Run it with:
    python chat.py
    python chat.py --temperature 0.6 --max-tokens 150
    python chat.py --raw          (skip the User:/Bot: chat template)
"""

import argparse
import json
import os
import sys

import numpy as np

# architecture.py lives in this same folder -- direct, explicit connection
# to the model definition, no duplicated/forked copy of the network here.
from architecture import TinyTransformer, softmax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoint.bin")
DEFAULT_VOCAB = os.path.join(SCRIPT_DIR, "tokenize_vocab.json")

STOP_STRINGS = ["\nUser:", "User:", "\nHuman:", "<eos>"]


# ============================================================================
# Tokenizer (minimal, self-contained -- just reads the vocab json that
# tokenizer.py is responsible for producing)
# ============================================================================

class SimpleTokenizer:
    """Greedy longest-match tokenizer over a flat {token: id} vocab.
    Works whether the vocab is pure characters, whole words, or a hybrid
    of both -- it always prefers the longest matching token at each
    position, and falls back one character at a time if nothing in the
    vocab matches (so it never hard-crashes on unseen input, as long as
    the vocab has full character coverage)."""

    def __init__(self, vocab):
        self.token_to_id = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.max_token_len = max((len(t) for t in vocab), default=1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["vocab"] if "vocab" in data else data
        return cls(vocab)

    def encode(self, text):
        ids = []
        i = 0
        n = len(text)
        while i < n:
            match_len = min(self.max_token_len, n - i)
            token_id = None
            while match_len > 0:
                piece = text[i:i + match_len]
                if piece in self.token_to_id:
                    token_id = self.token_to_id[piece]
                    i += match_len
                    break
                match_len -= 1
            if token_id is None:
                # Nothing matched, not even a single character -- skip it
                # rather than crash (e.g. a codepoint the vocab never saw).
                i += 1
                continue
            ids.append(token_id)
        return ids

    def decode(self, ids):
        return "".join(self.id_to_token.get(i, "") for i in ids)


# ============================================================================
# Loading
# ============================================================================

def load_tokenizer(vocab_path=DEFAULT_VOCAB):
    if not os.path.exists(vocab_path):
        sys.exit(
            f"[!] No vocab file found at {vocab_path}\n"
            f"    Run tokenizer.py on this model first to generate it."
        )
    return SimpleTokenizer.load(vocab_path)


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
    if not os.path.exists(checkpoint_path):
        sys.exit(
            f"[!] No checkpoint found at {checkpoint_path}\n"
            f"    Run train.py on this model first to create one."
        )
    return TinyTransformer.load(checkpoint_path, device=device)


# ============================================================================
# Generation (drives architecture.py's forward() directly, one token at a
# time, so we can stream output and detect stop sequences -- this is the
# "connection" to the architecture: no black-box .generate() call)
# ============================================================================

def generate_reply(model, tokenizer, prompt_ids, max_new_tokens=200,
                    temperature=0.8, stop_strings=STOP_STRINGS, stream=False,
                    rng=None):
    rng = rng or np.random.default_rng()
    ids = list(prompt_ids)
    generated_ids = []

    for _ in range(max_new_tokens):
        window = ids[-model.max_seq_len:]
        logits, _, _ = model.forward(np.array(window))
        last_logits = logits[-1] / max(temperature, 1e-6)
        probs = softmax(last_logits)
        next_id = int(rng.choice(model.vocab_size, p=probs))

        ids.append(next_id)
        generated_ids.append(next_id)

        partial_text = tokenizer.decode(generated_ids)
        if stream:
            sys.stdout.write(tokenizer.decode([next_id]))
            sys.stdout.flush()

        for stop in stop_strings:
            if stop in partial_text:
                return partial_text[:partial_text.index(stop)].rstrip()

    return tokenizer.decode(generated_ids).rstrip()


def build_prompt(user_text, raw=False):
    if raw:
        return user_text
    return f"User: {user_text}\nBot: "


# ============================================================================
# REPL
# ============================================================================

def print_banner(model, checkpoint_path):
    print("=" * 56)
    print(" TLM Chat -- connected to architecture.TinyTransformer")
    print("=" * 56)
    print(f" checkpoint     : {checkpoint_path}")
    print(f" parameters     : {model.total_params:,}")
    print(f" d_model        : {model.d_model}")
    print(f" num_layers     : {model.num_layers}")
    print(f" num_heads      : {model.num_heads}")
    print(f" vocab_size     : {model.vocab_size}")
    print(f" max_seq_len    : {model.max_seq_len}")
    print("=" * 56)
    print(" Type your message and press Enter. Ctrl+C or /exit to quit.")
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained TLM model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--raw", action="store_true",
                         help="Skip the User:/Bot: chat template (raw base-model completion).")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.vocab)
    model = load_model(args.checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed)

    print_banner(model, args.checkpoint)

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            print("Goodbye.")
            break

        prompt = build_prompt(user_text, raw=args.raw)
        prompt_ids = tokenizer.encode(prompt)

        print("Bot: ", end="", flush=True)
        reply = generate_reply(
            model, tokenizer, prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
            rng=rng,
        )
        print()  # newline after streamed output


if __name__ == "__main__":
    main()
