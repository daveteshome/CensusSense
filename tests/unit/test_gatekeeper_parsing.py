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
            "metric_phrase": "population",
            "state": "Texas",
            "county": "",
            "year": "",
            "clarification_reason": "",
        }),
    )
    result = classify("population of Texas", [], cfg=object())
    assert result.status == "IN_SCOPE"
    assert result.metric_phrase == "population"
    assert result.state == "Texas"


def test_classify_fails_closed_on_unknown_status(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module, "get_client", lambda cfg: _fake_client({"status": "SOMETHING_NEW"})
    )
    result = classify("...", [], cfg=object())
    assert result.status == "OUT_OF_SCOPE"


def test_classify_defaults_missing_optional_fields_to_empty_string(monkeypatch):
    monkeypatch.setattr(
        gatekeeper_module, "get_client", lambda cfg: _fake_client({"status": "IN_SCOPE"})
    )
    result = classify("population", [], cfg=object())
    assert result.metric_phrase == ""
    assert result.state == ""
    assert result.county == ""
    assert result.year == ""
    assert result.clarification_reason == ""


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
