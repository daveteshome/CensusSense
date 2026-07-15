from metadata.metric_ranker import MetricRanker, _meaningful_tokens, _tokenize


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert _tokenize("Median Age By Sex: Median age: Total") == [
        "median", "age", "by", "sex", "median", "age", "total",
    ]


def test_meaningful_tokens_drops_stopwords():
    tokens = _meaningful_tokens("Median Household Income In The Past 12 Months")
    assert "in" not in tokens
    assert "the" not in tokens
    assert "median" in tokens
    assert "household" in tokens


def test_query_returns_nothing_for_zero_vocabulary_overlap():
    ranker = MetricRanker(["Median Age By Sex: Total", "Median Gross Rent"])
    results = ranker.query("completely unrelated gibberish term", top_n=5)
    assert results == []


def test_query_ranks_full_coverage_above_partial_coverage():
    docs = [
        "Median Household Income In The Past 12 Months",
        "Median Age By Sex: Total",
    ]
    ranker = MetricRanker(docs)
    results = ranker.query("median household income", top_n=5)
    assert results[0][0] == 0  # the household income doc, not median age
    assert results[0][1] == 1.0  # full coverage of the query's meaningful terms


def test_exact_phrase_match_gets_a_bonus_over_partial_overlap():
    docs = [
        "Median Household Income In The Past 12 Months",
        "Median Gross Rent As A Percentage Of Household Income",
    ]
    ranker = MetricRanker(docs)
    results = ranker.query("median household income", top_n=5)
    # both share median/household/income tokens, but only doc 0 contains
    # the literal phrase "median household income"
    assert results[0][0] == 0
    assert results[0][2] > results[1][2]
