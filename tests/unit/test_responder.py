from unittest.mock import MagicMock

import pytest

import agent.responder as responder_module
from agent.llm_client import LLMError
from agent.responder import ResponderInput, generate_answer


def _fake_client(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _payload(**overrides) -> ResponderInput:
    defaults = dict(
        question="population of Texas",
        metric_description="Total Population",
        value=28260856,
        block_group_count=15811,
        year="2019",
        state_name="Texas",
        county_name=None,
        is_approximate=False,
        caveat=None,
        year_was_defaulted=False,
    )
    defaults.update(overrides)
    return ResponderInput(**defaults)


def test_generate_answer_returns_stripped_text(monkeypatch):
    monkeypatch.setattr(responder_module, "get_client", lambda cfg: _fake_client("  Texas has about 28.3 million people.  "))
    answer = generate_answer(_payload(), [], cfg=object())
    assert answer == "Texas has about 28.3 million people."


def test_generate_answer_includes_caveat_in_prompt_facts(monkeypatch):
    captured = {}

    def fake_get_client(cfg):
        client = MagicMock()

        def _create(model, messages, temperature):
            captured["messages"] = messages
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="answer"))]
            return response

        client.chat.completions.create.side_effect = _create
        return client

    monkeypatch.setattr(responder_module, "get_client", fake_get_client)
    generate_answer(_payload(is_approximate=True, caveat="treat as approximate"), [], cfg=object())
    user_message = captured["messages"][1]["content"]
    assert "treat as approximate" in user_message


def test_generate_answer_raises_llmerror_on_failure(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    monkeypatch.setattr(responder_module, "get_client", lambda cfg: client)
    with pytest.raises(LLMError):
        generate_answer(_payload(), [], cfg=object())
