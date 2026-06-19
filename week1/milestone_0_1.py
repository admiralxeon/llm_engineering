"""
Milestone 0 + 1 — verify the whole chain works before building the real project.

This is COMPLETE and runnable (no TODOs). It checks three things in order:
  0a. Ollama server is reachable
  0b. The model responds to a simple call
  1.  Scraping returns links + contents

Run:
    python milestone_0_1.py                      # uses a default test URL
    python milestone_0_1.py https://anthropic.com
"""

import sys
import requests
from openai import OpenAI
from scraper import fetch_website_links, fetch_website_contents

MODEL = "llama3.2"   # change to "llama3.2:1b" if your machine is small
OLLAMA = "http://localhost:11434"

client = OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama")


def check_server():
    print("0a. Checking Ollama server...")
    try:
        r = requests.get(OLLAMA, timeout=5)
        if "Ollama is running" in r.text:
            print("    OK — Ollama is running.\n")
            return True
    except requests.exceptions.RequestException:
        pass
    print("    NOT reachable. Open a terminal and run:  ollama serve\n")
    return False


def check_model():
    print(f"0b. Calling the model ({MODEL})...")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Tell me a fun fact in one sentence."}],
        )
        print("    Model says:", resp.choices[0].message.content.strip(), "\n")
        return True
    except Exception as e:
        print(f"    Call failed: {e}")
        print(f"    Did you pull the model?  ollama pull {MODEL}\n")
        return False


def check_scrape(url):
    print(f"1.  Scraping {url} ...")
    contents = fetch_website_contents(url)
    links = fetch_website_links(url)
    print(f"    Got {len(contents)} chars of text and {len(links)} links.")
    print("    First few links:")
    for link in links[:8]:
        print("      -", link)
    print("\n    Text preview:")
    print("   ", contents[:300].replace("\n", " "), "...\n")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://anthropic.com"

    if not check_server():
        sys.exit(1)
    if not check_model():
        sys.exit(1)
    check_scrape(url)

    print("All three checks passed. The chain works — now build out the milestones in README.md.")


if __name__ == "__main__":
    main()
