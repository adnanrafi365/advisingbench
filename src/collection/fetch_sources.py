"""
Fetch public official UIC source pages for AdvisingBench.

This script:
1. Reads source URLs from data/raw/uic_sources.csv
2. Checks robots.txt before fetching
3. Downloads public page text slowly and respectfully
4. Saves cleaned text locally in data/raw/source_texts/
5. Saves a fetch log in data/raw/source_fetch_log.csv

Important:
- Do not use this for login-protected pages.
- Do not collect private student data.
- Do not publish large copied UIC page text publicly.
"""

from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import time
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


SOURCES_CSV = Path("data/raw/uic_sources.csv")
OUTPUT_DIR = Path("data/raw/source_texts")
FETCH_LOG_CSV = Path("data/raw/source_fetch_log.csv")

USER_AGENT = "AdvisingBenchResearchBot/1.0 (student research project; respectful crawling)"
REQUEST_DELAY_SECONDS = 2
TIMEOUT_SECONDS = 20


def get_robots_url(url: str) -> str:
    """Return robots.txt URL for a given page URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def is_allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check whether the URL is allowed by robots.txt."""
    robots_url = get_robots_url(url)

    try:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as error:
        print(f"Could not read robots.txt for {url}: {error}")
        print("Skipping this URL to stay cautious.")
        return False


def clean_text_from_html(html: str) -> str:
    """Extract readable text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned


def fetch_page_text(url: str) -> tuple[bool, str, str]:
    """
    Fetch a page and return:
    success, text, error_message
    """
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return False, "", f"Unsupported content type: {content_type}"

        text = clean_text_from_html(response.text)

        if len(text) < 200:
            return False, text, "Extracted text is too short"

        return True, text, ""

    except Exception as error:
        return False, "", str(error)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCES_CSV.exists():
        raise FileNotFoundError(f"Could not find {SOURCES_CSV}")

    sources = pd.read_csv(SOURCES_CSV)
    logs = []

    print(f"Loaded {len(sources)} sources from {SOURCES_CSV}")

    for _, row in sources.iterrows():
        source_id = row["source_id"]
        title = row["title"]
        url = row["url"]

        print(f"\nProcessing {source_id}: {title}")
        print(url)

        allowed = is_allowed_by_robots(url)

        if not allowed:
            print("Skipped: robots.txt does not allow this fetch.")
            logs.append({
                "source_id": source_id,
                "url": url,
                "status": "skipped_robots",
                "text_file": "",
                "num_characters": 0,
                "error": "Disallowed by robots.txt or robots.txt unavailable"
            })
            continue

        success, text, error = fetch_page_text(url)

        if success:
            output_file = OUTPUT_DIR / f"{source_id}.txt"

            with output_file.open("w", encoding="utf-8") as f:
                f.write(f"Source ID: {source_id}\n")
                f.write(f"Title: {title}\n")
                f.write(f"URL: {url}\n")
                f.write("\n--- PAGE TEXT ---\n\n")
                f.write(text)

            print(f"Saved: {output_file}")
            print(f"Characters: {len(text)}")

            logs.append({
                "source_id": source_id,
                "url": url,
                "status": "success",
                "text_file": str(output_file),
                "num_characters": len(text),
                "error": ""
            })

        else:
            print(f"Failed: {error}")
            logs.append({
                "source_id": source_id,
                "url": url,
                "status": "failed",
                "text_file": "",
                "num_characters": len(text),
                "error": error
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(FETCH_LOG_CSV, index=False)

    print("\nFetch complete.")
    print(f"Log saved to: {FETCH_LOG_CSV}")
    print(log_df["status"].value_counts())


if __name__ == "__main__":
    main()
