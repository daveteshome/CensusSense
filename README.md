# CensusSense

A chat agent that answers natural-language questions about US population and
demographics, grounded in the US Open Census Data (Neighborhood Insights)
dataset on Snowflake Marketplace.

## Introduction

A production-oriented, chat-based agent that answers natural-language
questions about US population and demographics. The core architectural
idea: the LLM never generates SQL. It only produces structured intent — a
concept, geography, year, and (for ranking questions) a sort direction and
row count — which is resolved against a verified metadata catalog. SQL is
then built deterministically and independently re-validated before it ever
reaches Snowflake, making hallucinated table/column names structurally
impossible rather than just discouraged by prompting.

## At a Glance

| Item | Value |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| Data source | US Open Census Data (Neighborhood Insights), Snowflake Marketplace |
| LLM | Groq — `qwen/qwen3-32b` |
| Deployment | Railway |
| SQL generation | Deterministic (LLM never generates SQL or names a table/column) |
| Conversation memory | LangGraph checkpointer, SQLite-backed, per-account persisted |
| Tests | 226 automated (unit + integration) |

## Live Demo

**URL**: https://web-production-e991c.up.railway.app

**Sign in** (3 separate accounts, each with its own persisted conversation
history — sign out and back in with the same account and your prior
conversation is still there):
- `evaluator1` / `census-sense-2026`
- `evaluator2` / `census-sense-2026`
- `evaluator3` / `census-sense-2026`

No local setup required.

## Features

- Chat-based, multi-turn conversation with persistent, per-account memory
- Deterministic, metadata-driven SQL generation (see Introduction)
- Guardrails: off-topic, inappropriate/adversarial input, ambiguous or conflicting queries
- Ranking/superlative queries (single winner, top/bottom N, or all — state or county scope)
- Typo-tolerant geography and metric matching, with a "did you mean X?" confirmation flow
- Graceful degradation on any failure — never a crash, empty response, or invented number
- Independent SQL validation (defense-in-depth against prompt injection)
- Definitional questions answered from a closed, human-written glossary — never LLM-generated

## Architecture Overview

```
User (Streamlit chat UI, signed in as one of 3 accounts)
        │
        ▼
LangGraph pipeline (shared process, per-account persisted state)
        │
   ┌────┴─────┐
   │ Gatekeeper │  LLM: scope/safety classification + structured query intent
   └────┬──────┘
        │ (short-circuits to a guardrail message if out of scope,
        │  inappropriate, or conflicting)
        ▼
   ┌───────────┐
   │ Resolver   │  concept/geography/year -> verified metadata (typo-corrected)
   └────┬──────┘
        ▼
   ┌───────────┐
   │ SQL Builder│  deterministic template, only from already-verified metadata
   └────┬──────┘
        ▼
   ┌───────────┐
   │ Validator  │  independent re-check via sqlglot: allow-list, LIMIT enforced
   └────┬──────┘
        ▼
   ┌───────────┐
   │ Executor   │  Snowflake, least-privilege role, timeout + retry
   └────┬──────┘
        ▼
   ┌───────────┐
   │ Responder  │  LLM: computed number(s) -> natural-language answer
   └────┬──────┘
        ▼
   Final answer + updated conversation memory
```

**Why LangGraph**: the pipeline is several deterministic stages with
conditional routing (short-circuit to a guardrail on any non-happy path),
shared conversation state, and per-node failure isolation. Modeling this as
an explicit state graph keeps the routing readable as it grows, and the
same graph state is what the checkpointer snapshots for conversation
memory and per-account persistence — no separate machinery needed.

A few implementation notes, in brief:
- **Data**: ACS 5-year estimates (2019 and 2020 vintages, block-group
  grain) plus 2020 decennial redistricting counts. Geography is resolved
  from the `CENSUS_BLOCK_GROUP` ID's state/county FIPS prefix, not a named
  column.
- **Concept catalog**: the resolver matches against a curated ~200-entry
  concept catalog (`data/concept_catalog.json`), not the raw ~8,300-column
  schema — the Gatekeeper never sees table/column names at all.
- **Ranking**: `operation`/`sort`/`limit` are parameters the Gatekeeper
  fills in from a closed set; one query builder handles every ranking
  phrasing (single winner, top/bottom N, or all — capped and validated
  server-side, never an unbounded query).
- **Persistence**: conversation history is keyed by a stable per-account
  thread ID and stored via a SQLite-backed LangGraph checkpointer —
  signing back in with the same account restores the full conversation.
- **Median/rate metrics** (income, age, rent) are never summed across
  geographies — aggregated with `AVG` and explicitly caveated in the
  response as an approximation.

### Example Request Flow

"What are the 10 most populated states?"

```
Gatekeeper (LLM) — structured intent, never SQL:
{
  "status": "IN_SCOPE",
  "concept": "population",
  "operation": "RANKING",
  "sort": "DESC",
  "limit": "10",
  "ranking_scope": "STATE"
}
        ↓
Resolver — concept_catalog.json → metadata.json:
  "population" → table="2020_REDISTRICTING_CBG_DATA", column="P0010001"
        ↓
SQL Builder — deterministic, no LLM involved:
  SELECT SUBSTR("CENSUS_BLOCK_GROUP", 1, 2) AS GROUP_FIPS, SUM("P0010001") AS VALUE
  FROM "2020_REDISTRICTING_CBG_DATA"
  GROUP BY SUBSTR("CENSUS_BLOCK_GROUP", 1, 2)
  ORDER BY VALUE DESC NULLS LAST
  LIMIT 10
        ↓
Validator — re-checks table/column against the same allow-list, confirms a LIMIT is present
        ↓
Executor — Snowflake, 10 rows back
        ↓
Responder (LLM) — "California is the most populous state, with about 39.5
million people, followed by Texas (29.1M), Florida (21.5M), ..."
```

## Repository Structure

```
CensusSense/
├── app.py                       Streamlit entrypoint (sign-in, chat UI)
├── config.py                     Environment/config loading
├── logging_config.py              Structured JSON logging
├── snowflake_client.py             Low-level Snowflake connection helper
├── requirements.txt
├── Procfile                        Railway process definition
├── .env.example                     Required environment variables (template)
│
├── agent/                        The LangGraph pipeline
│   ├── gatekeeper.py                LLM #1: scope/safety classification + structured query intent
│   ├── resolver.py                   Concept/geography/year resolution against verified metadata
│   ├── sql_builder.py                 Deterministic SQL construction
│   ├── sql_validator.py                Independent allow-list re-check (sqlglot)
│   ├── executor.py                      Snowflake execution + retry
│   ├── responder.py                      LLM #2: natural-language answer
│   ├── pipeline.py                        LangGraph wiring + conversation memory
│   ├── guardrails.py                       Canned copy + glossary + rate limiter
│   ├── llm_client.py                        Shared Groq client setup
│   └── state.py                              Conversation turn representation
│
├── metadata/                     Offline-built metadata + concept catalog
│   ├── metadata_store.py            Loads metadata.json + concept_catalog.json + geography_fips.json
│   ├── metric_ranker.py               TF-IDF/coverage ranking + typo correction
│   ├── build_metadata.py               Offline metadata.json builder (raw ~8,300-entry index)
│   ├── build_concept_catalog.py         Offline concept_catalog.json builder (curated ~200-entry index)
│   ├── normalize.py                      Census variable ID <-> Snowflake column transform
│   └── us_states.py                       Static state name/FIPS reference
│
├── data/                         Committed, offline-built (no live Snowflake needed to load)
│   ├── metadata.json                ~8,300-entry raw column index
│   ├── concept_catalog.json           ~200-entry curated concept index
│   ├── geography_fips.json             State/county FIPS crosswalk
│   └── ...                              raw export dumps consumed by build_metadata.py
│
├── scripts/                      One-off data export scripts (need live Snowflake credentials)
│   ├── export_schema.py
│   ├── export_geography.py
│   └── export_field_descriptions.py
│
├── sql/
│   └── setup_role.sql             Least-privilege Snowflake role setup
│
└── tests/
    ├── unit/                      One module per agent/metadata file, mocked at the LLM/Snowflake boundary
    ├── integration/                Full pipeline + SQLite-checkpointer tests
    └── fixtures/                    Small hand-built metadata/geography/concept-catalog fixtures
```

## Local Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # fill in real values
streamlit run app.py
```

Required environment variables (see `.env.example`): `GROQ_API_KEY`,
`SNOWFLAKE_ACCOUNT`/`USER`/`PASSWORD`/`ROLE`/`WAREHOUSE`/`DATABASE`,
`DEMO_ACCOUNTS` (comma-separated `user:pass` pairs, one per demo account —
e.g. `evaluator1:pw1,evaluator2:pw2,evaluator3:pw3`). Optionally
`CONVERSATIONS_DB_PATH` (default `data/conversations.db`) to relocate
where per-account conversation history is persisted.

`sql/setup_role.sql` creates the dedicated, least-privilege Snowflake role
used at runtime (not `ACCOUNTADMIN`).

`data/metadata.json` and `data/concept_catalog.json` are committed and
don't need rebuilding to run the app locally. To rebuild them after a
schema change (requires live Snowflake credentials):

```bash
python scripts/export_schema.py
python scripts/export_geography.py
python scripts/export_field_descriptions.py
python -m metadata.build_metadata
python -m metadata.build_concept_catalog
```

## Testing

```bash
pytest
```

226 automated tests (unit + integration), all mocked at the LLM/Snowflake
boundary so the suite runs in seconds with no live credentials required.
Several tests run against the real, committed `data/metadata.json` and
`data/concept_catalog.json` as a regression check, not just small
fixtures, and `tests/integration/test_persistence.py` verifies the
SQLite-backed checkpointer specifically (history surviving a simulated
process restart, per-account isolation). See `REFLECTION.md` for testing
philosophy, trade-offs, and what's still missing.

## Requirements → Design Mapping

| Assignment requirement | Where it's satisfied |
|---|---|
| Interactive, chat-based agent grounded in the Census dataset | Streamlit UI over the LangGraph pipeline; SQL only ever built from verified metadata |
| Web-based interface, publicly accessible, no local setup to evaluate | Single Streamlit service on Railway (see Live Demo above) |
| Preserve conversation context across multiple turns | LangGraph's checkpointer, keyed by a stable per-account thread_id, persisted to SQLite so it survives sign-out/sign-in and server restarts |
| Complete responses within 60 seconds under normal conditions | Explicit per-stage timeout budget (Gatekeeper/Responder ~12s each, Snowflake ~25s); typical real latency 0.3-2s |
| Guardrails against off-topic or inappropriate responses | Gatekeeper classifies `OUT_OF_SCOPE` / `INAPPROPRIATE` distinctly; SQL validator allow-list as independent defense-in-depth |
| Degrade gracefully — no hallucination, empty response, or unhandled error | Every node wraps LLM/Snowflake calls; failures route to a deterministic guardrail message, never a raw error or invented number |
| Handle ambiguous, partially-matching, conflicting/underspecified, or reasonable-but-unanswerable queries | `agent/resolver.py` + `agent/guardrails.py` — see Guardrails in practice, below |
| Meaningful tests | 226 automated tests (unit + integration) plus extensive live testing — see Testing above and `REFLECTION.md` |

## Guardrails in practice

- **Out of scope, generic** ("who won the super bowl") → a clear, fixed decline.
- **Out of scope, specific reason** ("most populated city", "average salary
  of a software engineer", non-US geography, pre-2019 years) → the
  Gatekeeper names the *specific* limitation rather than a generic refusal.
- **Definitional questions** ("what is a Census Block Group", "median vs
  average") → answered from a small, human-written, closed glossary —
  never an LLM-generated definition.
- **Ambiguous metric** (two curated concepts scoring closely) → asks which
  was meant, listing real candidates by name. A bare "income" resolves
  cleanly to median household income by default rather than asking, via a
  deliberately curated alias.
- **Conflicting details** ("population in 2018 and 2025") → asks for
  clarification, naming the specific conflict.
- **Metric not in the curated concept catalog** (a fine-grained
  demographic slice, not a headline figure) → declines, naming the real
  scope cut, not a guess at a similarly-worded but wrong metric.
- **Unsupported year** → states the years actually available.
- **Unrecognized/misspelled geography** → typo-corrected when confident
  enough; when plausible but not confident, asks "Did you mean
  California?" rather than guessing — a "yes" on the next turn resolves
  it via existing conversation memory; clearly wrong geography gets a
  not-found message.
- **Ranking/superlative questions** (single winner, top/bottom N, "all
  ordered", state or county scope — see Example Request Flow above) →
  summable metrics only; median/rate metrics decline with a specific
  explanation regardless of phrasing.
- **Empty result** → an honest "couldn't find matching records" message,
  never a bare zero.
- **Unhandled exception anywhere in the pipeline** → a clean guardrail
  message, never a raw traceback, and never persisted into history — so a
  transient failure doesn't compound into future turns.

## Known Limitations

- No city/place-level geography — the dataset is block-group/county/state
  level only.
- Ranking is supported for summable metrics only (population, housing
  counts); median/rate metrics decline, since ranking many
  already-approximate numbers would compound one approximation on
  another. Row count is capped at `MAX_RANKING_LIMIT` (60) — a nationwide
  county "all" legitimately exceeds this (~3,143 US counties), and the
  response discloses the truncation rather than silently dropping rows.
- No multi-geography *comparison* ("Cook County vs. the national
  average") — a different, still-unsupported case from ranking above;
  deliberately deferred, explained in `REFLECTION.md`.
- One continuous, persisted conversation thread per account — no "start a
  new chat but keep the old one" option, and no multiple saved
  conversations per account. Scoped out as a separate follow-on feature.
- On Railway, the SQLite conversation-history file lives on the
  container's local filesystem, wiped on redeploy unless a [Railway
  Volume](https://docs.railway.com/reference/volumes) is mounted at
  `CONVERSATIONS_DB_PATH`'s directory. History survives sign-out/sign-in
  and in-place restarts either way, just not a redeploy without the
  volume.
- Median/rate-type metrics aggregated above block-group level are an
  unweighted average, explicitly caveated in the response, not a true
  recomputed median.
- The concept catalog covers ~200 curated headline concepts, not every
  one of the ~8,300 raw Census variables — a fine-grained demographic
  slice gets an honest decline naming the scope cut, not a guess. See
  `metadata/build_concept_catalog.py` and `REFLECTION.md` for which
  tables were excluded and why.

## Future Improvements

- Multi-geography *comparison* ("Cook County vs. the national average")
  — a distinct capability from ranking, deliberately deferred; the
  `operation` field is already a closed enum designed to grow into this
  as a third value (`COMPARISON`) alongside `LOOKUP`/`RANKING`.
- Compound multi-metric questions in one turn ("population and median
  age of Texas") — currently resolves one metric per turn.
- City/place-level geography — would need a TIGER/Line place-boundary
  crosswalk layered on top of the current block-group/county/state data.
- Expand the concept catalog beyond ~200 curated headline concepts to
  cover more fine-grained demographic slices.


See `REFLECTION.md` for the full reasoning behind each of these — what's
already been tried, what's a real trade-off vs. just unfinished, and
what I'd prioritize first with more time.

## Additional Documentation

- **[`REFLECTION.md`](REFLECTION.md)** — development process and key
  architectural decisions, what I'd improve with more time, edge cases and
  failure modes identified but not fully addressed, and testing approach —
  mirrors the assignment's reflection requirements directly.
- **[`TESTING_ANALYSIS.md`](TESTING_ANALYSIS.md)** — a detailed,
  supplementary live-testing writeup: strengths and weaknesses observed
  against the real pipeline, and every bug found and fixed during
  development.
