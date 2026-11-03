"""Text analysis: tokenization, folding, stopwords and a light stemmer.

The analyzer is deterministic and dependency-free so that the lexical index is
fully reproducible and the property-based tests can rely on exact behaviour.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, explicit stopword list. Kept short on purpose: an aggressive list
# hurts recall on short queries, which is where lexical search matters most.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with
    """.split()
)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _stem(token: str) -> str:
    """A conservative suffix stripper (a tiny Porter-style subset).

    It is intentionally mild: over-stemming collapses distinct terms and makes
    ranking worse, so only high-confidence suffixes are removed.
    """
    for suffix in ("ational", "ization", "iveness", "fulness"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "ied", "ess", "ers", "er", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            return base
    return token


@dataclass(frozen=True)
class Analyzer:
    """Turns raw text into a list of normalized terms.

    Parameters are frozen so an index and its queries always share one config.
    """

    lowercase: bool = True
    fold_accents: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    min_token_len: int = 1

    def analyze(self, text: str) -> list[str]:
        if text is None:
            return []
        s = text
        if self.fold_accents:
            s = _strip_accents(s)
        if self.lowercase:
            s = s.lower()
        tokens = _TOKEN_RE.findall(s)
        out: list[str] = []
        for tok in tokens:
            if len(tok) < self.min_token_len:
                continue
            if self.remove_stopwords and tok in STOPWORDS:
                continue
            if self.stem:
                tok = _stem(tok)
            if tok:
                out.append(tok)
        return out
