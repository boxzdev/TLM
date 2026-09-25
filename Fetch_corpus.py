"""
fetch_corpus.py

Downloads free training text into your data/ folder from two sources:
  - Project Gutenberg (public-domain novels and foundational classics),
    found via the Gutendex API (https://gutendex.com).
  - Wikipedia (CC BY-SA articles), via Wikipedia's extract API.

Both sources are free to reuse without licensing friction.

Run from your project root:
    python fetch_corpus.py
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

DATA_DIR = os.path.join(os.getcwd(), "data")
HEADERS = {"User-Agent": "TLM-corpus-fetcher/1.1 (personal ML training research; contact@example.com)"}
PAUSE_SECONDS = 1.0  # Polite interval for free public APIs
SKIP_EXISTING = True # Skip files that are already downloaded

# =====================================================================
# 1. Project Gutenberg: Public-domain books (novels, philosophy, science)
# =====================================================================
GUTENBERG_SEARCHES = [
    # Philosophy & Political Theory
    "Plato Republic",
    "Meditations Marcus Aurelius",
    "Nietzsche Thus Spoke Zarathustra",
    "Nietzsche Beyond Good and Evil",
    "Aristotle Nicomachean Ethics",
    "The Prince Niccolo Machiavelli",
    "Leviathan Thomas Hobbes",
    "Walden Henry David Thoreau",
    "The Wealth of Nations Adam Smith",
    "Critique of Pure Reason Immanuel Kant",
    "Spinoza Ethics",
    "A Vindication of the Rights of Woman Mary Wollstonecraft",
    "The Art of War Sunzi",

    # Foundational Science & Essays
    "On the Origin of Species Charles Darwin",
    "Relativity Special and General Theory Albert Einstein",
    "Dialogues Concerning Two New Sciences Galileo",

    # Classic Literature & Fiction
    "Pride and Prejudice Jane Austen",
    "Frankenstein Mary Shelley",
    "Alice's Adventures in Wonderland Lewis Carroll",
    "The Adventures of Sherlock Holmes Arthur Conan Doyle",
    "Moby Dick Herman Melville",
    "Great Expectations Charles Dickens",
    "A Tale of Two Cities Charles Dickens",
    "The Count of Monte Cristo Alexandre Dumas",
    "Crime and Punishment Fyodor Dostoyevsky",
    "The Brothers Karamazov Fyodor Dostoyevsky",
    "The Picture of Dorian Gray Oscar Wilde",
    "Dracula Bram Stoker",
    "The Time Machine H. G. Wells",
    "The War of the Worlds H. G. Wells",
    "The Iliad Homer",
    "The Odyssey Homer",
    "The Metamorphosis Franz Kafka",
    "The Adventures of Tom Sawyer Mark Twain",
    "Heart of Darkness Joseph Conrad",
    "Treasure Island Robert Louis Stevenson",
]

# =====================================================================
# 2. Wikipedia: Concise, factual, high-density expository articles
# =====================================================================
WIKI_TOPICS = [
    # Philosophy & Mind
    "Philosophy", "Stoicism", "Existentialism", "Epistemology", "Ethics",
    "Metaphysics", "Socrates", "Plato", "Aristotle", "Immanuel Kant",
    "Friedrich Nietzsche", "Rationalism", "Empiricism", "Logic",
    "Consciousness", "Free will", "Utilitarianism", "Phenomenology",
    "Cognitive science", "Philosophy of mind",

    # Computer Science, AI & Mathematics
    "Computer science", "Algorithm", "Artificial intelligence", "Machine learning",
    "Deep learning", "Artificial neural network", "Natural language processing",
    "Information theory", "Turing machine", "Mathematics", "Calculus",
    "Probability theory", "Linear algebra", "Graph theory", "Cryptography",

    # Natural Sciences & Astronomy
    "Physics", "General relativity", "Quantum mechanics", "Thermodynamics",
    "Speed of light", "Black hole", "Astronomy", "Solar System", "Milky Way",
    "Evolution", "DNA", "Cell biology", "Ecology", "Plate tectonics",

    # World History & Civilizations
    "History of the world", "Ancient Greece", "Roman Empire", "Silk Road",
    "Renaissance", "Age of Enlightenment", "Industrial Revolution",
    "Printing press", "Scientific Revolution", "World War I", "World War II",
]


def slugify(text, max_len=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "untitled"


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def strip_gutenberg_boilerplate(text):
    """Gutenberg wraps every file in a license header/footer. Keep only
    the actual work between the START and END markers."""
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if start and end and end.start() > start.end():
        text = text[start.end():end.start()]
    return text.strip()


def fetch_gutenberg(search_term):
    slug = slugify(search_term)
    target_pattern = f"gutenberg_{slug}"

    if SKIP_EXISTING:
        for existing in os.listdir(DATA_DIR):
            if existing.startswith(target_pattern) and os.path.getsize(os.path.join(DATA_DIR, existing)) > 0:
                print(f"  [skipped] '{search_term}' already downloaded.")
                return 0

    query = urllib.parse.quote(search_term)
    url = f"https://gutendex.com/books/?search={query}"
    data = json.loads(http_get(url))
    results = data.get("results", [])
    if not results:
        print(f"  [missing] no match found for '{search_term}'")
        return None

    book = results[0]
    formats = book.get("formats", {})
    text_url = next((u for k, u in formats.items() if k.startswith("text/plain")), None)

    if not text_url:
        print(f"  [unsupported] no plain-text format for '{book.get('title')}'")
        return None

    raw = http_get(text_url)
    text = strip_gutenberg_boilerplate(raw)
    title = book.get("title", search_term)
    path = os.path.join(DATA_DIR, f"gutenberg_{slug}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  [saved] '{title}' -> {os.path.basename(path)} ({len(text):,} chars)")
    return len(text)


def fetch_wikipedia(title):
    path = os.path.join(DATA_DIR, f"wiki_{slugify(title)}.txt")
    if SKIP_EXISTING and os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  [skipped] Wikipedia '{title}' already exists.")
        return 0

    query = urllib.parse.quote(title)
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&prop=extracts&explaintext=1&redirects=1&titles={query}"
    )
    data = json.loads(http_get(url))
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    text = page.get("extract", "").strip()

    if not text:
        print(f"  [missing] article extract empty or not found for '{title}'")
        return None

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  [saved] '{title}' -> {os.path.basename(path)} ({len(text):,} chars)")
    return len(text)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Destination: {DATA_DIR}\n")
    total_chars, ok, failed = 0, 0, 0

    print(f"=== Project Gutenberg ({len(GUTENBERG_SEARCHES)} targets) ===")
    for term in GUTENBERG_SEARCHES:
        try:
            n = fetch_gutenberg(term)
            if n is not None:
                total_chars += n
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [error] '{term}': {e}")
            failed += 1
        time.sleep(PAUSE_SECONDS)

    print(f"\n=== Wikipedia ({len(WIKI_TOPICS)} targets) ===")
    for title in WIKI_TOPICS:
        try:
            n = fetch_wikipedia(title)
            if n is not None:
                total_chars += n
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [error] '{title}': {e}")
            failed += 1
        time.sleep(PAUSE_SECONDS)

    print(f"\nSummary: {ok} processed, {failed} failed.")
    print(f"New data downloaded: {total_chars:,} characters.")
    print("Next step: Retrain your tokenizer (`tokenizer.py`), then launch training (`train.py`).")


if __name__ == "__main__":
    main()