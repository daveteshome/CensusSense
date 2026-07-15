import pytest

from config import _parse_demo_accounts


def test_parse_demo_accounts_parses_multiple_pairs():
    accounts = _parse_demo_accounts("alice:pass1,bob:pass2,carol:pass3")
    assert accounts == {"alice": "pass1", "bob": "pass2", "carol": "pass3"}


def test_parse_demo_accounts_tolerates_whitespace_around_pairs():
    accounts = _parse_demo_accounts(" alice:pass1 , bob:pass2 ")
    assert accounts == {"alice": "pass1", "bob": "pass2"}


def test_parse_demo_accounts_rejects_entry_missing_colon():
    with pytest.raises(RuntimeError, match="Malformed"):
        _parse_demo_accounts("alice-pass1")


def test_parse_demo_accounts_rejects_entry_missing_password():
    with pytest.raises(RuntimeError, match="Malformed"):
        _parse_demo_accounts("alice:")


def test_parse_demo_accounts_rejects_empty_string():
    with pytest.raises(RuntimeError, match="at least one"):
        _parse_demo_accounts("")
