"""
Phase 5 - Shared Bangla text normalisation and tokenisation.

Every script on the lexical route (build_lexicon.py, asr_diagnostics.py,
lexical_did.py) must tokenise identically, otherwise a word in an ASR
transcript will not match the same word in a lexicon. Import from here.
"""

import re
import unicodedata
from functools import lru_cache

# Bengali letters and dependent signs only: drops Bangla digits (U+09E6-09EF),
# currency/fraction signs, danda, Latin text and all punctuation. ZWNJ/ZWJ are
# kept until the word-level normaliser has seen them, then stripped.
_NOT_WORD = re.compile(r"[^ঀ-ৣৰৱ‌‍]+")
_ZW = re.compile(r"[‌‍]")
_bnorm = None


@lru_cache(maxsize=None)
def normalize_word(word):
    """Canonical Unicode form of one Bangla word ('' if nothing valid is left)."""
    global _bnorm
    if _bnorm is None:
        from bnunicodenormalizer import Normalizer
        _bnorm = Normalizer()
    out = _bnorm(word)["normalized"]
    return _ZW.sub("", out) if out else ""


def tokenize(text):
    """Text -> list of normalised Bangla word tokens."""
    if not isinstance(text, str):          # NaN from an empty transcript cell
        return []
    words = _NOT_WORD.sub(" ", unicodedata.normalize("NFC", text)).split()
    return [w for w in map(normalize_word, words) if w]


def load_vocab(path, min_count=1):
    """Read a 'word<TAB>count' file written by build_lexicon.py into a set."""
    vocab = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            if int(count) >= min_count:
                vocab.add(word)
    return vocab
