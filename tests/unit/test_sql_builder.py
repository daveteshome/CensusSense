import pytest

from agent.sql_builder import SqlBuildError, build_query
from metadata.metadata_store import MetricEntry

SUM_ENTRY = MetricEntry(
    table="2019_CBG_B01",
    column="B01001e1",
    year="2019",
    description="Sex By Age: Total",
    table_topics="Age and Sex",
    universe="Total population",
    aggregation_type="SUM",
    moe_column="B01001m1",
)

MEDIAN_ENTRY = MetricEntry(
    table="2019_CBG_B19",
    column="B19013e1",
    year="2019",
    description="Median Household Income",
    table_topics="Income",
    universe="Households",
    aggregation_type="MEDIAN",
    moe_column="B19013m1",
)


def test_sum_entry_state_filter_uses_sum_no_caveat():
    built = build_query(SUM_ENTRY, state_fips="48")
    assert 'SUM("B01001e1")' in built.sql
    assert 'FROM "2019_CBG_B01"' in built.sql
    assert "SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '48'" in built.sql
    assert built.is_approximate is False
    assert built.caveat is None


def test_sum_entry_state_and_county_filter():
    built = build_query(SUM_ENTRY, state_fips="48", county_fips="453")
    assert "SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '48'" in built.sql
    assert "SUBSTR(\"CENSUS_BLOCK_GROUP\", 3, 3) = '453'" in built.sql
    assert " AND " in built.sql


def test_no_geography_filter_queries_nationwide():
    built = build_query(SUM_ENTRY)
    assert "WHERE" not in built.sql


def test_median_entry_uses_avg_and_sets_caveat():
    built = build_query(MEDIAN_ENTRY, state_fips="48")
    assert 'AVG("B19013e1")' in built.sql
    assert built.is_approximate is True
    assert built.caveat is not None
    assert "average" in built.caveat.lower()


def test_median_entry_never_sums():
    built = build_query(MEDIAN_ENTRY, state_fips="48")
    assert "SUM(" not in built.sql


def test_county_without_state_is_rejected():
    with pytest.raises(SqlBuildError):
        build_query(SUM_ENTRY, county_fips="453")


@pytest.mark.parametrize("bad_fips", ["48;DROP TABLE x--", "TX", "48 OR 1=1", ""])
def test_non_numeric_fips_is_rejected(bad_fips):
    with pytest.raises(SqlBuildError):
        build_query(SUM_ENTRY, state_fips=bad_fips)


def test_query_always_includes_limit():
    built = build_query(SUM_ENTRY, state_fips="48")
    assert "LIMIT 1" in built.sql
