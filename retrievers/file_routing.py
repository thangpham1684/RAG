"""Heuristic file routing utilities for matching query keywords to filenames.

Originally from the Streamlit app (app.py), extracted for reuse after migrating
to the Web UI.
"""

import re
import unicodedata

FILE_HINT_STOPWORDS = {
    "file", "tai", "lieu", "document", "van", "ban", "pdf", "docx",
    "moi", "nhat", "cu", "nay", "kia", "do", "trong",
    "ve", "cho", "cua", "voi", "the", "la", "co", "duoc",
}


def normalize_for_match(text: str) -> str:
    """Normalize Vietnamese text for fuzzy file matching.

    - Lowercases
    - Removes accents (diacritics)
    - Strips punctuation and extra spaces
    """
    text = text.lower().strip()
    # Remove combining diacritical marks (accents)
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Keep only letters, digits, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_file_aliases(files: list[str]) -> dict[str, set[str]]:
    """Build a mapping of filename → set of searchable alias strings.

    Aliases include:
    - The original normalized filename
    - Each word in the filename (minus stopwords)
    - Bigrams of consecutive words
    """
    aliases = {}
    for fname in files:
        norm = normalize_for_match(fname)
        # Remove extension
        base = norm.rsplit(".", 1)[0] if "." in norm else norm
        words = base.split()
        fname_aliases = {norm, base}
        for w in words:
            if w not in FILE_HINT_STOPWORDS and len(w) > 1:
                fname_aliases.add(w)
        # Bigrams
        for i in range(len(words) - 1):
            bg = words[i] + words[i + 1]
            if len(bg) > 2:
                fname_aliases.add(bg)
        aliases[fname] = fname_aliases
    return aliases


def detect_files_from_query(query: str, file_aliases: dict[str, set[str]]) -> list[str]:
    """Given a user query and filename→aliases mapping, return matching filenames.

    Returns filenames sorted by hit count descending.
    """
    norm_query = normalize_for_match(query)
    query_words = set(norm_query.split())
    scores: dict[str, int] = {}
    for fname, aliases in file_aliases.items():
        score = 0
        # Direct alias match (e.g. "baocaotaichinh" matches alias "baocaotaichinh")
        for alias in aliases:
            if alias in norm_query:
                score += len(alias)
        # Word-level overlap
        alias_words = set()
        for a in aliases:
            alias_words.update(a.split())
        overlap = query_words & alias_words
        score += len(overlap) * 3
        if score > 0:
            scores[fname] = score
    return sorted(scores, key=lambda f: scores[f], reverse=True)
