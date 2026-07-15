import pathlib

import pytest

from agent.sql_validator import SqlValidationError, validate
from metadata.metadata_store import load_store

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def store():
    return load_store(
        metadata_path=FIXTURES / "sample_metadata.json",
        geography_path=FIXTURES / "sample_geography.json",
    )


def test_valid_select_passes(store):
    sql = (
        'SELECT SUM("B01001e1") AS VALUE, COUNT(*) AS BLOCK_GROUP_COUNT '
        'FROM "2019_CBG_B01" '
        "WHERE SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '48' LIMIT 1"
    )
    validate(sql, store)  # should not raise


def test_ranking_query_with_value_alias_in_order_by_passes(store):
    # Regression test: VALUE is sql_builder's own generated alias, not a
    # real Census column -- the validator must not flag ORDER BY VALUE
    # as an "unknown column" reference.
    sql = (
        'SELECT SUBSTR("CENSUS_BLOCK_GROUP", 1, 2) AS GROUP_FIPS, SUM("B01001e1") AS VALUE '
        'FROM "2019_CBG_B01" '
        'GROUP BY SUBSTR("CENSUS_BLOCK_GROUP", 1, 2) '
        "ORDER BY VALUE DESC LIMIT 1"
    )
    validate(sql, store)  # should not raise


def test_valid_select_with_county_filter_passes(store):
    sql = (
        'SELECT AVG("B19013e1") AS VALUE, COUNT(*) AS BLOCK_GROUP_COUNT '
        'FROM "2019_CBG_B19" '
        "WHERE SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '48' "
        "AND SUBSTR(\"CENSUS_BLOCK_GROUP\", 3, 3) = '453' LIMIT 1"
    )
    validate(sql, store)


def test_rejects_unknown_table(store):
    sql = 'SELECT SUM("B01001e1") FROM "SOME_OTHER_TABLE" LIMIT 1'
    with pytest.raises(SqlValidationError, match="unknown table"):
        validate(sql, store)


def test_rejects_unknown_column(store):
    sql = 'SELECT SUM("NOT_A_REAL_COLUMN") FROM "2019_CBG_B01" LIMIT 1'
    with pytest.raises(SqlValidationError, match="unknown column"):
        validate(sql, store)


def test_rejects_missing_limit(store):
    sql = 'SELECT SUM("B01001e1") FROM "2019_CBG_B01"'
    with pytest.raises(SqlValidationError, match="LIMIT"):
        validate(sql, store)


def test_rejects_non_select_statement(store):
    sql = 'DROP TABLE "2019_CBG_B01"'
    with pytest.raises(SqlValidationError):
        validate(sql, store)


def test_rejects_stacked_statements_injection_attempt(store):
    sql = 'SELECT SUM("B01001e1") FROM "2019_CBG_B01" LIMIT 1; DROP TABLE "2019_CBG_B01";'
    with pytest.raises(SqlValidationError):
        validate(sql, store)


def test_rejects_update_statement(store):
    sql = 'UPDATE "2019_CBG_B01" SET "B01001e1" = 0'
    with pytest.raises(SqlValidationError):
        validate(sql, store)


def test_rejects_cross_table_column_not_in_referenced_table(store):
    # B19013e1 exists on 2019_CBG_B19, not on 2019_CBG_B01 -- referencing
    # it against the wrong table should still fail even though the
    # column exists somewhere in the allow-list.
    sql = 'SELECT SUM("B19013e1") FROM "2019_CBG_B01" LIMIT 1'
    with pytest.raises(SqlValidationError, match="unknown column"):
        validate(sql, store)
