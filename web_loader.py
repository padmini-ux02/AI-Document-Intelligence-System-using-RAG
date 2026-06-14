"""
web_loader.py — URL / Wikipedia ingestion for Nexus AI.
Fetches and cleans web page text without heavy dependencies.
"""
import re
from typing import Tuple
from config import WEB_REQUEST_TIMEOUT, MAX_WEB_CHARS


def _clean_html(html_text: str) -> str:
    """Strip tags, scripts, styles; collapse whitespace."""
    # Remove script / style blocks
    html_text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html_text, flags=re.S | re.I)
    # Remove all remaining HTML tags
    html_text = re.sub(r"<[^>]+>", " ", html_text)
    # Decode common HTML entities
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
                 "&quot;": '"', "&#39;": "'", "&ndash;": "–", "&mdash;": "—"}
    for ent, char in entities.items():
        html_text = html_text.replace(ent, char)
    # Collapse whitespace
    html_text = re.sub(r"\s+", " ", html_text)
    return html_text.strip()


def load_url(url: str) -> Tuple[str, str]:
    """
    Fetch a URL and return (clean_text, title).
    Raises RuntimeError on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("Please install 'requests' and 'beautifulsoup4' to use URL ingestion.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=WEB_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove nav, footer, header, aside, ads
    for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style",
                               "form", "button", "iframe", "noscript"]):
        tag.decompose()

    # Prefer article/main/body content
    content_tags = ["article", "main", "section", "div[role='main']"]
    body = None
    for selector in ["article", "main"]:
        found = soup.find(selector)
        if found:
            body = found
            break
    if not body:
        body = soup.find("body") or soup

    # Extract text with paragraph breaks
    paragraphs = []
    for elem in body.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "th"]):
        text = elem.get_text(separator=" ", strip=True)
        if len(text) > 30:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)

    if not full_text.strip():
        # Fallback: just clean the raw HTML
        full_text = _clean_html(resp.text)

    # Truncate to max chars
    full_text = full_text[:MAX_WEB_CHARS]

    return full_text, title


def is_wikipedia_url(url: str) -> bool:
    return "wikipedia.org/wiki/" in url


def load_wikipedia(url: str) -> Tuple[str, str]:
    """
    Use Wikipedia's REST API for cleaner extraction when possible.
    Falls back to load_url on failure.
    """
    try:
        import requests
        # Extract article title from URL
        match = re.search(r"wikipedia\.org/wiki/(.+)$", url)
        if not match:
            return load_url(url)
        article = match.group(1)
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}"
        resp = requests.get(api_url, timeout=WEB_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            title = data.get("title", article)
            # Also fetch the full content via the parse API
            full_api = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={article}&format=json"
            full_resp = requests.get(full_api, timeout=WEB_REQUEST_TIMEOUT)
            if full_resp.status_code == 200:
                pages = full_resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    full_text = page.get("extract", extract)
                    return full_text[:MAX_WEB_CHARS], title
            return extract[:MAX_WEB_CHARS], title
    except Exception:
        pass
    return load_url(url)


def ingest_url(url: str) -> Tuple[str, str]:
    """Main entry point. Routes to Wikipedia or generic loader."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if is_wikipedia_url(url):
        return load_wikipedia(url)
    return load_url(url)
