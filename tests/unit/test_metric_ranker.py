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


def test_typo_in_query_word_is_corrected_against_the_corpus_vocabulary():
    ranker = MetricRanker(["Total Population: Total", "Median Age By Sex: Total"])
    typo_results = ranker.query("populaton", top_n=5)
    clean_results = ranker.query("population", top_n=5)
    assert typo_results and clean_results
    assert typo_results[0][0] == clean_results[0][0] == 0
    assert typo_results[0][1] == clean_results[0][1] == 1.0  # same full coverage


def test_short_words_are_not_typo_corrected():
    # "of" and "by" are stopwords/short words that shouldn't get corrected
    # against arbitrary short vocabulary entries -- guards against noisy
    # corrections on function words.
    ranker = MetricRanker(["Median Age By Sex: Total"])
    corrected = ranker._correct_tokens(["by", "of", "a"])
    assert corrected == ["by", "of", "a"]


def test_correction_does_not_fire_on_unrelated_words():
    # A word with no close real typo relationship to the vocabulary
    # should pass through unchanged rather than being force-matched to
    # the closest (irrelevant) vocabulary word.
    ranker = MetricRanker(["Median Household Income In The Past 12 Months"])
    corrected = ranker._correct_tokens(["spaceship"])
    assert corrected == ["spaceship"]
