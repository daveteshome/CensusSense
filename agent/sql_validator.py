"""Independent hard backstop, regardless of what the Gatekeeper LLM
decided: every generated query is re-parsed and checked against the
metadata allow-list before it ever reaches Snowflake. A prompt-injection
attempt that somehow talked its way past the Gatekeeper still cannot
produce SQL outside this allow-list, because sql_builder.py never
accepts free text in the first place — this is defense-in-depth on top
of that, not the only line of defense.
"""

import sqlglot
from sqlglot import exp

from metadata.metadata_store import MetadataStore

# CENSUS_BLOCK_GROUP is a real column on every fact table. VALUE and
# GROUP_FIPS are aliases sql_builder.py itself generates (e.g. `SELECT
# SUM(...) AS VALUE ... ORDER BY VALUE` in ranking queries) -- not real
# Census columns, but referencing them back is exactly how SQL aliases
# work, and they're never influenced by user input, only by our own
# fixed query templates.
ALWAYS_ALLOWED_COLUMNS = {"CENSUS_BLOCK_GROUP", "VALUE", "GROUP_FIPS"}


class SqlValidationError(Exception):
    pass


def _build_allowlist(store: MetadataStore) -> tuple[set[str], dict[str, set[str]]]:
    tables: set[str] = set()
    columns_by_table: dict[str, set[str]] = {}
    for entry in store.entries:
        tables.add(entry.table)
        columns_by_table.setdefault(entry.table, set()).add(entry.column)
        if entry.moe_column:
            columns_by_table[entry.table].add(entry.moe_column)
    return tables, columns_by_table


def validate(sql: str, store: MetadataStore) -> None:
    """Raises SqlValidationError if `sql` is anything other than a single
    SELECT against allow-listed tables/columns. Returns None (silently)
    if the query is safe to execute."""
    try:
        statements = sqlglot.parse(sql, dialect="snowflake")
    except sqlglot.errors.ParseError as exc:
        raise SqlValidationError(f"could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlValidationError(f"expected exactly one statement, got {len(statements)}")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SqlValidationError(f"only SELECT statements are allowed, got {type(statement).__name__}")

    allowed_tables, allowed_columns_by_table = _build_allowlist(store)

    referenced_tables = {t.name for t in statement.find_all(exp.Table)}
    if not referenced_tables:
        raise SqlValidationError("no table referenced")
    unknown_tables = referenced_tables - allowed_tables
    if unknown_tables:
        raise SqlValidationError(f"unknown table(s): {sorted(unknown_tables)}")

    allowed_columns = set(ALWAYS_ALLOWED_COLUMNS)
    for table in referenced_tables:
        allowed_columns |= allowed_columns_by_table.get(table, set())

    referenced_columns = {c.name for c in statement.find_all(exp.Column)}
    unknown_columns = referenced_columns - allowed_columns
    if unknown_columns:
        raise SqlValidationError(f"unknown column(s): {sorted(unknown_columns)}")

    if statement.args.get("limit") is None:
        raise SqlValidationError("query must include a LIMIT clause")
