import json
from unittest.mock import MagicMock

import pytest

import agent.gatekeeper as gatekeeper_module
from agent.gatekeeper import classify
from agent.llm_client import LLMError


def _fake_client(payload: dict) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_classify_returns_parsed_result(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client({
            "status": "IN_SCOPE",
            "concept": "population",
            "state": "Texas",
            "county": "",
            "year": "",
            "clarification_reason": "",
        }),
    )
    result = classify("population of Texas", [], cfg=object())
    assert result.status == "IN_SCOPE"
    assert result.concept == "population"
    assert result.state == "Texas"


def test_classify_fails_closed_on_unknown_status(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module, "get_client", lambda cfg: _fake_client({"status": "SOMETHING_NEW"})
    )
    result = classify("...", [], cfg=object())
    assert result.status == "OUT_OF_SCOPE"


def test_classify_parses_glossary_term(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client({"status": "OUT_OF_SCOPE", "glossary_term": "country"}),
    )
    result = classify("what is a country", [], cfg=object())
    assert result.glossary_term == "country"


def test_classify_defaults_missing_optional_fields_to_empty_string(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module, "get_client", lambda cfg: _fake_client({"status": "IN_SCOPE"})
    )
    result = classify("population", [], cfg=object())
    assert result.concept == ""
    assert result.state == ""
    assert result.county == ""
    assert result.year == ""
    assert result.glossary_term == ""
    assert result.clarification_reason == ""
    assert result.operation == "LOOKUP"
    assert result.sort == ""
    assert result.limit == 0
    assert result.limit_was_all is False
    assert result.ranking_scope == ""


def test_classify_parses_ranking_fields(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client({
            "status": "IN_SCOPE",
            "concept": "population",
            "operation": "RANKING",
            "sort": "DESC",
            "limit": "1",
            "ranking_scope": "STATE",
        }),
    )
    result = classify("which state has the most population", [], cfg=object())
    assert result.operation == "RANKING"
    assert result.sort == "DESC"
    assert result.limit == 1
    assert result.limit_was_all is False
    assert result.ranking_scope == "STATE"


def test_classify_fails_closed_on_invalid_operation_and_scope(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client({"status": "IN_SCOPE", "operation": "DESTROY", "ranking_scope": "PLANET"}),
    )
    result = classify("population", [], cfg=object())
    assert result.operation == "LOOKUP"
    assert result.sort == ""
    assert result.ranking_scope == ""


def test_classify_ranking_with_empty_sort_falls_back_to_lookup(monkeypatch):
    # A ranking request with no coherent direction isn't safely
    # actionable as a ranking at all -- fails all the way back to LOOKUP.
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client({"status": "IN_SCOPE", "operation": "RANKING", "sort": "", "ranking_scope": "STATE"}),
    )
    result = classify("population", [], cfg=object())
    assert result.operation == "LOOKUP"
    assert result.sort == ""
    assert result.ranking_scope == ""


def test_classify_limit_all_resolves_to_max_and_flags_it(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {"status": "IN_SCOPE", "operation": "RANKING", "sort": "DESC", "limit": "ALL", "ranking_scope": "STATE"}
        ),
    )
    result = classify("all states ordered by population", [], cfg=object())
    assert result.limit == gatekeeper_module.MAX_RANKING_LIMIT
    assert result.limit_was_all is True


def test_classify_limit_numeric_string_parses(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {"status": "IN_SCOPE", "operation": "RANKING", "sort": "DESC", "limit": "10", "ranking_scope": "STATE"}
        ),
    )
    result = classify("top 10 states by population", [], cfg=object())
    assert result.limit == 10
    assert result.limit_was_all is False


def test_classify_limit_over_cap_clamps_down(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {"status": "IN_SCOPE", "operation": "RANKING", "sort": "DESC", "limit": "9999", "ranking_scope": "STATE"}
        ),
    )
    result = classify("top 9999 states by population", [], cfg=object())
    assert result.limit == gatekeeper_module.MAX_RANKING_LIMIT


def test_classify_missing_limit_under_ranking_defaults_to_one(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {"status": "IN_SCOPE", "operation": "RANKING", "sort": "DESC", "ranking_scope": "STATE"}
        ),
    )
    result = classify("most populated state", [], cfg=object())
    assert result.limit == 1
    assert result.limit_was_all is False


def test_classify_limit_as_raw_json_int_still_parses(monkeypatch):
    # LLMs sometimes emit a bare number instead of the requested string.
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {"status": "IN_SCOPE", "operation": "RANKING", "sort": "DESC", "limit": 10, "ranking_scope": "STATE"}
        ),
    )
    result = classify("top 10 states by population", [], cfg=object())
    assert result.limit == 10


def test_classify_nationwide_county_ranking_has_no_state(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module,
        "get_client",
        lambda cfg: _fake_client(
            {
                "status": "IN_SCOPE",
                "operation": "RANKING",
                "sort": "DESC",
                "limit": "1",
                "ranking_scope": "COUNTY",
                "state": "",
            }
        ),
    )
    result = classify("which county in the US has the most people", [], cfg=object())
    assert result.ranking_scope == "COUNTY"
    assert result.state == ""


def test_classify_raises_llmerror_on_client_exception(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    monkeypatch.setattr(gatekeeper_module, "get_client", lambda cfg: client)
    with pytest.raises(LLMError):
        classify("population", [], cfg=object())


def test_classify_raises_llmerror_on_invalid_json(monkeypatch):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(gatekeeper_module, "get_client", lambda cfg: client)
    with pytest.raises(LLMError):
        classify("population", [], cfg=object())
