import json
import os
import random
import re
import time
import urllib.error
import urllib.request

API_BASE = "https://datasets-server.huggingface.co/rows"
OUTPUT_DIR = "finetune_data"   # Files will be saved directly here
MAX_ROWS = 1500               # Rows to fetch per dataset
MAX_TURN_WORDS = 25           # Word limit per dialogue turn
PAGE_SIZE = 100               # API page size
VAL_RATIO = 0.10              # 10% reserved for validation / testing
HEADERS = {"User-Agent": "TLM-multi-dataset-fetcher/1.0"}


# ---------------------------------------------------------------------------
# Text cleaning helper
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Fix common spacing artifacts around punctuation and contractions."""
    if not text:
        return ""
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"(\b\w+)\s+['’]\s+(\w+\b)", r"\1'\2", text)
    text = re.sub(r"\s+(['’][a-zA-Z]+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Format Parsers for each dataset
# ---------------------------------------------------------------------------

def parse_daily_dialog(row, max_words):
    """DailyDialog parser."""
    raw_dialog = row.get("dialog", [])
    turns = [clean_text(t) for t in raw_dialog if clean_text(t)]
    if len(turns) < 2:
        return None

    u, a = turns[0], turns[1]
    if len(u.split()) > max_words or len(a.split()) > max_words:
        return None

    return {
        "messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]
    }


def parse_smoltalk(row, max_words):
    """SmolTalk parser."""
    messages = row.get("messages", [])
    if not messages or len(messages) < 2:
        return None

    cleaned = []
    for msg in messages:
        content = clean_text(msg.get("content", ""))
        if not content or len(content.split()) > max_words:
            return None
        cleaned.append({"role": msg.get("role", "user"), "content": content})

    if cleaned[-1]["role"] != "assistant":
        return None
    return {"messages": cleaned}


def parse_alpaca(row, max_words):
    """Alpaca cleaned parser."""
    inst = clean_text(row.get("instruction", ""))
    inp = clean_text(row.get("input", ""))
    out = clean_text(row.get("output", ""))

    prompt = f"{inst}\n{inp}".strip() if inp else inst
    if not prompt or not out:
        return None
    if len(prompt.split()) > max_words or len(out.split()) > max_words:
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out},
        ]
    }


def parse_dolly(row, max_words):
    """Databricks Dolly 15k parser."""
    inst = clean_text(row.get("instruction", ""))
    ctx = clean_text(row.get("context", ""))
    resp = clean_text(row.get("response", ""))

    prompt = f"{inst}\n{ctx}".strip() if ctx else inst
    if not prompt or not resp:
        return None
    if len(prompt.split()) > max_words or len(resp.split()) > max_words:
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp},
        ]
    }


# ---------------------------------------------------------------------------
# Dataset Registry
# Add or remove datasets here; each gets its own output files.
# ---------------------------------------------------------------------------

DATASETS_TO_FETCH = [
    {
        "name": "daily_dialog",
        "dataset": "OpenRL/daily_dialog",
        "config": "default",
        "split": "train",
        "parser": parse_daily_dialog,
    },
    {
        "name": "smoltalk_everyday",
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "everyday-conversations",
        "split": "train",
        "parser": parse_smoltalk,
    },
    {
        "name": "alpaca_cleaned",
        "dataset": "yahma/alpaca-cleaned",
        "config": "default",
        "split": "train",
        "parser": parse_alpaca,
    },
    {
        "name": "dolly_15k",
        "dataset": "databricks/databricks-dolly-15k",
        "config": "default",
        "split": "train",
        "parser": parse_dolly,
    },
]


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_page(dataset: str, config: str, split: str, offset: int, length: int, retries: int = 3):
    url = f"{API_BASE}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise


def save_dataset(name: str, examples: list):
    """Saves complete dataset as well as train/val splits into OUTPUT_DIR."""
    # Deterministic shuffle for reproducible splits
    random.seed(42)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * VAL_RATIO)
    val_set = shuffled[:val_count]
    train_set = shuffled[val_count:]

    # 1. Full file
    full_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # 2. Train split
    train_path = os.path.join(OUTPUT_DIR, f"{name}_train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)

    # 3. Validation split
    val_path = os.path.join(OUTPUT_DIR, f"{name}_val.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)

    print(f"  -> Saved full file  : {full_path} ({len(examples)} examples)")
    print(f"  -> Saved train split: {train_path} ({len(train_set)} examples)")
    print(f"  -> Saved val split  : {val_path} ({len(val_set)} examples)\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving output files into directory: '{OUTPUT_DIR}/'\n")

    for target in DATASETS_TO_FETCH:
        name = target["name"]
        print(f"=== Fetching {name} ({target['dataset']}) ===")
        examples = []
        offset = 0

        while offset < MAX_ROWS:
            try:
                payload = fetch_page(
                    dataset=target["dataset"],
                    config=target["config"],
                    split=target["split"],
                    offset=offset,
                    length=PAGE_SIZE,
                )
            except Exception as e:
                print(f"  Stopped early at offset {offset}: {e}")
                break

            rows = payload.get("rows", [])
            if not rows:
                break

            for item in rows:
                parsed = target["parser"](item.get("row", {}), max_words=MAX_TURN_WORDS)
                if parsed:
                    examples.append(parsed)

            offset += PAGE_SIZE
            print(f"  Fetched {offset:4d} / {MAX_ROWS} rows -> {len(examples):4d} valid examples", end="\r")
            time.sleep(0.25)

        print()
        if examples:
            save_dataset(name, examples)
        else:
            print(f"  No valid examples found for {name}.\n")

    print(f"All datasets processed! Check your '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()