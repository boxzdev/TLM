"""
fetch_corpus.py

Downloads free training text into your data/ folder from two sources:
  - Project Gutenberg (public-domain novels and philosophy classics),
      found via the Gutendex API (https://gutendex.com) so no book IDs
          need to be hardcoded.
            - Wikipedia (CC BY-SA articles), via Wikipedia's own API.

            Both sources are free to reuse -- Gutenberg books here are out of
            copyright, and Wikipedia text is openly licensed.

            Run from your project's root folder (the one containing data/,
            factory/, training/):
                python fetch_corpus.py

                Edit GUTENBERG_SEARCHES and WIKI_TOPICS below to change what it grabs.
                After it finishes, re-run tokenizer.py and train.py from scratch,
                since the vocab and corpus both changed.
                """

                import json
                import os
                import re
                import time
                import urllib.parse
                import urllib.request

                DATA_DIR = os.path.join(os.getcwd(), "data")
                HEADERS = {"User-Agent": "TLM-corpus-fetcher/1.0 (personal ML training data)"}
                PAUSE_SECONDS = 1.0  # be polite to free APIs

                # Gutenberg: public-domain novels and philosophy, found by search term.
                GUTENBERG_SEARCHES = [
                    "Pride and Prejudice Jane Austen",
                        "Frankenstein Mary Shelley",
                            "Alice's Adventures in Wonderland",
                                "Adventures of Sherlock Holmes",
                                    "Plato Republic",
                                        "Meditations Marcus Aurelius",
                                            "Nietzsche Thus Spoke Zarathustra",
                                                "Aristotle Nicomachean Ethics",
                                                ]

                                                # Wikipedia: open-license articles, by exact page title.
                                                WIKI_TOPICS = [
                                                    "Philosophy", "Stoicism", "Existentialism", "Epistemology", "Ethics",
                                                        "Metaphysics", "Socrates", "Plato", "Aristotle", "Immanuel Kant",
                                                            "Friedrich Nietzsche", "Rationalism", "Empiricism", "Logic",
                                                                "Consciousness", "Free will", "Utilitarianism",
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
                                                                                                the actual work, between the START and END markers."""
                                                                                                    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
                                                                                                        end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
                                                                                                            if start and end and end.start() > start.end():
                                                                                                                    text = text[start.end():end.start()]
                                                                                                                        return text.strip()


                                                                                                                        def fetch_gutenberg(search_term):
                                                                                                                            query = urllib.parse.quote(search_term)
                                                                                                                                data = json.loads(http_get(f"https://gutendex.com/books/?search={query}"))
                                                                                                                                    results = data.get("results", [])
                                                                                                                                        if not results:
                                                                                                                                                print(f"  no match for '{search_term}'")
                                                                                                                                                        return None
                                                                                                                                                            book = results[0]
                                                                                                                                                                formats = book.get("formats", {})
                                                                                                                                                                    text_url = next((u for k, u in formats.items()
                                                                                                                                                                                          if k.startswith("text/plain")), None)
                                                                                                                                                                                              if not text_url:
                                                                                                                                                                                                      print(f"  no plain-text format for '{book.get('title')}'")
                                                                                                                                                                                                              return None
                                                                                                                                                                                                                  raw = http_get(text_url)
                                                                                                                                                                                                                      text = strip_gutenberg_boilerplate(raw)
                                                                                                                                                                                                                          title = book.get("title", search_term)
                                                                                                                                                                                                                              path = os.path.join(DATA_DIR, f"gutenberg_{slugify(title)}.txt")
                                                                                                                                                                                                                                  with open(path, "w", encoding="utf-8") as f:
                                                                                                                                                                                                                                          f.write(text)
                                                                                                                                                                                                                                              print(f"  saved '{title}' -> {os.path.basename(path)} ({len(text):,} chars)")
                                                                                                                                                                                                                                                  return len(text)


                                                                                                                                                                                                                                                  def fetch_wikipedia(title):
                                                                                                                                                                                                                                                      query = urllib.parse.quote(title)
                                                                                                                                                                                                                                                          url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
                                                                                                                                                                                                                                                                     f"&prop=extracts&explaintext=1&redirects=1&titles={query}")
                                                                                                                                                                                                                                                                         data = json.loads(http_get(url))
                                                                                                                                                                                                                                                                             pages = data.get("query", {}).get("pages", {})
                                                                                                                                                                                                                                                                                 page = next(iter(pages.values()), {})
                                                                                                                                                                                                                                                                                     text = page.get("extract", "").strip()
                                                                                                                                                                                                                                                                                         if not text:
                                                                                                                                                                                                                                                                                                 print(f"  no article found for '{title}'")
                                                                                                                                                                                                                                                                                                         return None
                                                                                                                                                                                                                                                                                                             path = os.path.join(DATA_DIR, f"wiki_{slugify(title)}.txt")
                                                                                                                                                                                                                                                                                                                 with open(path, "w", encoding="utf-8") as f:
                                                                                                                                                                                                                                                                                                                         f.write(text)
                                                                                                                                                                                                                                                                                                                             print(f"  saved '{title}' -> {os.path.basename(path)} ({len(text):,} chars)")
                                                                                                                                                                                                                                                                                                                                 return len(text)


                                                                                                                                                                                                                                                                                                                                 def main():
                                                                                                                                                                                                                                                                                                                                     os.makedirs(DATA_DIR, exist_ok=True)
                                                                                                                                                                                                                                                                                                                                         print(f"Saving into: {DATA_DIR}\n")
                                                                                                                                                                                                                                                                                                                                             total_chars, ok, failed = 0, 0, 0

                                                                                                                                                                                                                                                                                                                                                 print(f"Project Gutenberg ({len(GUTENBERG_SEARCHES)} books):")
                                                                                                                                                                                                                                                                                                                                                     for term in GUTENBERG_SEARCHES:
                                                                                                                                                                                                                                                                                                                                                             try:
                                                                                                                                                                                                                                                                                                                                                                         n = fetch_gutenberg(term)
                                                                                                                                                                                                                                                                                                                                                                                     total_chars += n or 0
                                                                                                                                                                                                                                                                                                                                                                                                 ok += 1 if n else 0
                                                                                                                                                                                                                                                                                                                                                                                                             failed += 0 if n else 1
                                                                                                                                                                                                                                                                                                                                                                                                                     except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                 print(f"  failed '{term}': {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                             failed += 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                     time.sleep(PAUSE_SECONDS)

                                                                                                                                                                                                                                                                                                                                                                                                                                                         print(f"\nWikipedia ({len(WIKI_TOPICS)} articles):")
                                                                                                                                                                                                                                                                                                                                                                                                                                                             for title in WIKI_TOPICS:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 n = fetch_wikipedia(title)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             total_chars += n or 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ok += 1 if n else 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     failed += 0 if n else 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         print(f"  failed '{title}': {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     failed += 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             time.sleep(PAUSE_SECONDS)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 print(f"\nDone: {ok} files saved, {failed} failed, {total_chars:,} characters total.")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     print("Next: run tokenizer.py, then train.py (start fresh).")


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         main()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         