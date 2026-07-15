"""TF-IDF/cosine relevance ranking over metric descriptions.

Plain edit-distance fuzzy matching (rapidfuzz) was tried first and
empirically fails here: with ~4,000 verbose, similarly-worded ACS
descriptions per year sharing lots of boilerplate ("Total", "In The Past
12 Months", "Households"), character-level string similarity produces
high-confidence-looking matches for completely unrelated queries (e.g.
"average salary of software engineers" scored 85.5 against a geographic
mobility field). TF-IDF down-weights exactly that boilerplate and scores
zero for queries that share no distinctive vocabulary with the corpus at
all, which is what we actually want for reasonable-but-unanswerable
detection. Rapidfuzz remains the right tool for state/county name typo
correction (a small, clean list where edit distance is the relevant
signal) -- see resolve_geography in resolver.py.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small and generic (not census-specific): these words are
# excluded only from the *coverage* signal (see query()), not from the
# TF-IDF vectors themselves -- idf weighting already discounts them there.
_STOPWORDS = {
    "of", "the", "in", "for", "and", "by", "a", "an", "is", "are", "to",
    "with", "on", "or", "as", "at", "from",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _meaningful_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOPWORDS}


PHRASE_MATCH_BONUS = 1.0


class MetricRanker:
    def __init__(self, documents: list[str]):
        self._documents_lower = [d.lower() for d in documents]
        doc_tokens = [_tokenize(d) for d in documents]
        doc_frequency: Counter = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                doc_frequency[term] += 1

        n_docs = max(len(documents), 1)
        self._idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in doc_frequency.items()
        }
        self._doc_vectors = [self._vectorize(tokens) for tokens in doc_tokens]
        self._doc_meaningful_tokens = [_meaningful_tokens(d) for d in documents]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        term_frequency = Counter(tokens)
        vector = {
            term: count * self._idf.get(term, 0.0)
            for term, count in term_frequency.items()
        }
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def query(self, text: str, top_n: int) -> list[tuple[int, float, float]]:
        """Returns [(document_index, coverage, cosine_similarity), ...]
        sorted by coverage first, cosine second, limited to top_n.

        Coverage (fraction of the query's meaningful/non-stopword terms
        present in the document) is the primary signal, cosine only
        breaks ties within it. Plain cosine alone was tried first and
        empirically ranks a short, mostly-irrelevant document above a
        long document matching every query term, because idf weighting
        lets one rare shared word dominate a short vector's norm (e.g.
        "median household income" ranked "Median Age By Sex" above the
        actual median household income entry). Coverage-first fixes
        that: a document containing all of the query's distinctive
        words always outranks one containing only one of them.
        """
        query_tokens = _tokenize(text)
        query_meaningful = _meaningful_tokens(text)
        query_vector = self._vectorize(query_tokens)
        query_lower = text.lower().strip()

        results = []
        for i, doc_vector in enumerate(self._doc_vectors):
            cosine = sum(query_vector.get(term, 0.0) * weight for term, weight in doc_vector.items())
            if cosine <= 0:
                continue
            # A near-exact phrase match (query appears contiguously in
            # the description) is strong evidence bag-of-words alone
            # can't see -- e.g. "median household income" is literally
            # in "Median Household Income In The Past 12 Months..." but
            # not in "Median Gross Rent As A Percentage Of Household
            # Income", even though both share the same 3 tokens.
            if query_lower and len(query_lower) > 3 and query_lower in self._documents_lower[i]:
                cosine += PHRASE_MATCH_BONUS
            if query_meaningful:
                overlap = query_meaningful & self._doc_meaningful_tokens[i]
                coverage = len(overlap) / len(query_meaningful)
            else:
                coverage = 0.0
            results.append((i, coverage, cosine))

        results.sort(key=lambda triple: (triple[1], triple[2]), reverse=True)
        return results[:top_n]
