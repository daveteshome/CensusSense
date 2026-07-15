"""Unit tests for the concept-catalog cluster-selection algorithm, run
against small synthetic row sets -- fast and independent of data changes,
unlike the real-data checks in test_metadata_store.py/test_resolver.py."""

from metadata.build_concept_catalog import (
    _base_group,
    _build_curated_concepts,
    _cluster_acs_entries,
    _derive_label,
    _is_race_iteration_group,
    _validate,
)


def _row(table, column, description, year="2019"):
    return {"table": table, "column": column, "year": year, "description": description}


def test_singleton_cluster_is_accepted_as_is():
    rows = [_row("t", "B01003e1", "Total Population: Total")]
    canonical, log = _cluster_acs_entries(rows)
    assert len(canonical) == 1
    assert canonical[0]["column"] == "B01003e1"
    assert log == []


def test_race_iteration_variants_fold_to_the_unqualified_row():
    rows = [
        _row("t", "B01002e1", "Median Age By Sex: Median age: Total"),
        _row("t", "B01002Ae1", "Median Age By Sex (White Alone): Median age: Total"),
        _row("t", "B01002Be1", "Median Age By Sex (Black Or African American Alone): Median age: Total"),
    ]
    canonical, log = _cluster_acs_entries(rows)
    assert len(canonical) == 1
    assert canonical[0]["column"] == "B01002e1"
    assert log == []


def test_multiple_unqualified_totals_are_excluded_not_guessed():
    # Mirrors a real case (B08's several bundled commuting concepts) --
    # more than one plausible "Total" row with no further signal to break
    # the tie should be an honest exclusion, not an arbitrary pick.
    rows = [
        _row("t", "B08301e1", "Means Of Transportation To Work: Total"),
        _row("t", "B08301e50", "Time Of Departure To Go To Work: Total"),
    ]
    canonical, log = _cluster_acs_entries(rows)
    assert len(canonical) == 0
    assert len(log) == 1
    assert "no clean default row" in log[0]


def test_cluster_with_no_total_row_at_all_is_excluded():
    rows = [
        _row("t", "B25022e2", "Aggregate Gross Rent: Under 20 percent"),
        _row("t", "B25022e3", "Aggregate Gross Rent: 20 to 24.9 percent"),
    ]
    canonical, log = _cluster_acs_entries(rows)
    assert len(canonical) == 0
    assert "0 unqualified totals" in log[0]


def test_base_group_folds_race_letter_but_not_a_genuine_base_group():
    race_row = _row("t", "B01002Ae1", "Median Age By Sex (White Alone): Median age: Total")
    assert _base_group(race_row) == "B01002"
    plain_row = _row("t", "B01003e1", "Total Population: Total")
    assert _base_group(plain_row) == "B01003"


def test_is_race_iteration_group_structural_check():
    assert _is_race_iteration_group("B01002A") is True
    assert _is_race_iteration_group("B01002") is False
    # Not a false positive on a base group that happens to end in a
    # digit-adjacent letter as part of its real, non-race number.
    assert _is_race_iteration_group("B01002F") is True


def test_derive_label_strips_trailing_total_segments():
    assert _derive_label("Total Population: Total") == "Total Population"
    assert _derive_label("Median Age By Sex: Median age: Total") == "Median Age By Sex: Median age"
    # No colon at all -- left as-is.
    assert _derive_label("Median Household Income In The Past 12 Months") == (
        "Median Household Income In The Past 12 Months"
    )


def test_validate_raises_on_a_source_pointer_that_does_not_exist():
    concepts = [{"concept": "x", "sources": {"2019": {"table": "t", "column": "does_not_exist"}}}]
    try:
        _validate(concepts, metadata_by_key={})
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "does_not_exist" not in str(exc) or "drifted" in str(exc) or "doesn't exist" in str(exc)


def test_validate_passes_when_every_pointer_resolves():
    concepts = [{"concept": "x", "sources": {"2019": {"table": "t", "column": "c"}}}]
    _validate(concepts, metadata_by_key={("t", "c"): {}})  # no raise


def test_curated_table_produces_one_renter_occupied_units_concept():
    metadata = [_row("2019_TOTAL_RENTAL_GEO", "Total: Renter-occupied housing units", "irrelevant")]
    concepts = _build_curated_concepts(metadata)
    assert list(concepts.keys()) == ["renter_occupied_housing_units"]
    assert concepts["renter_occupied_housing_units"]["sources"]["2019"]["table"] == "2019_TOTAL_RENTAL_GEO"


def test_curated_table_absent_produces_no_concept():
    assert _build_curated_concepts([]) == {}
