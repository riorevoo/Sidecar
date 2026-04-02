"""Text normalization: HTML stripping, unicode fixing, whitespace collapsing."""
from __future__ import annotations

import re
import unicodedata

import ftfy
from bs4 import BeautifulSoup

# Patterns for noise removal
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TRACKING_PIXELS_RE = re.compile(r"\[img[^\]]*\]|\[/img\]", re.IGNORECASE)

# Common boilerplate phrases found in news feed summaries
_BOILERPLATE_PHRASES = [
    "click here to read more",
    "read more at",
    "continue reading",
    "subscribe to our newsletter",
    "sign up for our",
    "follow us on",
    "share this article",
    "all rights reserved",
    "terms of service",
    "privacy policy",
    "advertisement",
    "sponsored content",
    "this article originally appeared",
    "the post appeared first on",
]


def normalize(text: str) -> str:
    """Apply the full normalization pipeline to a single piece of text.

    Steps:
    1. Fix unicode mojibake / encoding issues (ftfy)
    2. Strip HTML tags
    3. Normalize unicode to NFKC form
    4. Remove URLs and tracking tokens
    5. Remove boilerplate phrases
    6. Collapse whitespace
    7. Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Step 1: Fix encoding issues
    text = ftfy.fix_text(text)

    # Step 2: Strip HTML using BeautifulSoup (handles malformed HTML gracefully)
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "lxml")
        text = soup.get_text(separator=" ")

    # Step 3: Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Step 4: Remove URLs
    text = _URL_RE.sub("", text)
    text = _TRACKING_PIXELS_RE.sub("", text)

    # Step 5: Remove boilerplate (case-insensitive line-level removal)
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        lower = line.lower().strip()
        if not any(phrase in lower for phrase in _BOILERPLATE_PHRASES):
            clean_lines.append(line)
    text = " ".join(clean_lines)

    # Step 6: Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text)

    # Step 7: Strip
    return text.strip()
