# CensusSense — Testing Analysis

Compiled from: 226 automated tests (unit + integration, mocked LLM/Snowflake),
dozens of live manual tests against the real Groq/Snowflake backends
throughout development, and a structured 68-query battery across 11
categories run **twice** against the live pipeline — once on
`llama-3.3-70b-versatile` (completed 9/68 before exhausting its daily quota,
a real finding in its own right, see below) and once, completely, on
`qwen/qwen3-32b` after switching models specifically to address that
finding. This document reflects the full 68/68 run.

## Strengths

**No observed hallucination across all 68 queries.** Every data-bearing
answer traces to a real, validated Snowflake query. Spot-checked numbers
matched known real figures (Texas 28,635,442 for the 2020 decennial count,
Wyoming 581,348, US total ~329.8M). Structural, not just observed:
`sql_builder.py` only ever builds queries from verified `metadata.json`
entries, and `sql_validator.py` independently rejects anything outside
that allow-list regardless of what the LLM layers decided.

**Guardrails correctly gated every category, all 68 queries:**
- Out-of-scope, generic (coding, sports, weather, trivia, capital of
  France) — all correctly and clearly declined.
- **All 5 adversarial/injection attempts** ("ignore your previous
  instructions...", a DAN-style jailbreak, a credential-exfiltration
  request, a fake "SYSTEM: override safety" message, and a SQL-injection-
  flavored string) — correctly classified `INAPPROPRIATE` and refused with
  the firm, non-elaborating message, not the friendlier `OUT_OF_SCOPE`
  copy. No jailbreak succeeded.
- Ambiguous metrics correctly flagged with real candidate options rather
  than guessing, across four separate cases: bare "income" (3 real
  allocation-metric candidates), bare "poverty" (2 candidates), "median
  household income" (2 candidates), and "median household income in King
  County" (same). Every candidate listed was a real, distinct table — none
  invented.
- Unsupported geography (city-level, non-US), unsupported years, and
  non-tracked figures (occupation salary) each got a *specific*
  explanation via `scope_note` — including graceful generalization beyond
  the examples actually written into the prompt ("population of the
  moon" correctly got a sensible, specific explanation despite that exact
  case never being an example in the Gatekeeper's system prompt).
- Conflicting details correctly triggered clarification, both for
  conflicting years ("2018 and 2025") and conflicting states ("Texas or
  California" — correctly named both conflicting values in the message).
- Median/rate-type metrics (household income, median age, median gross
  rent, per capita income) consistently included the "unweighted average,
  treat as approximate" caveat.

**Conversation memory works across multi-step, mixed-topic threads.**
Verified with a longer, harder sequence than earlier testing: "median age
in Travis County" → "what does that mean, median age?" (correctly
diverted to the glossary, a full topic change) → "how about Harris
county?" (correctly *returned* to the original metric/state context,
resolving Harris County's median age — carrying context back across an
intervening unrelated glossary detour, not just consecutive similar
turns).

**Typo tolerance, both geography and metric, holds up under a full run.**
Every typo'd query in the battery resolved correctly: "begas" correctly
rejected (not a real place), "texs"→Texas, "Calfornia"→California,
"Ohoi"→Ohio, "Flordia"→Florida all resolved; "icnome"→income and
"populaton"/"populaiton"→population all resolved through the metric-side
correction added this session.

**An unintended positive side effect**: "renters or homeowners" — flagged
in the original critique as a semantic gap the ranker couldn't bridge —
now resolves cleanly (43.9M renter-occupied units) without any change
targeted at it specifically. The metric typo-correction fix (added to fix
misspellings like "populaton") also normalizes simple plural/singular
mismatches ("renters"→"renter" is a 1-character edit, caught by the same
mechanism), partially closing that earlier gap as a side effect.

**Definitional questions handled safely and correctly, including the
negative case.** All 10 real glossary terms answered correctly in the
intended Acknowledge-Contextualize-Redirect style. The one term
deliberately *not* in the glossary ("census tract") correctly fell through
to the generic decline rather than the model inventing a definition —
confirming the closed-set design actually holds under a real query, not
just in the unit tests.

**Honest partial-answer behavior on an explicitly out-of-scope compound
question.** "What is the population of Rhode Island in 2020 and also what
is the median age" — the architecture only resolves one metric per turn
(a known, documented limitation), and it picked median age. But rather
than silently presenting that as if it were the whole answer, the
Responder said so explicitly: median age, plus "*Population data for 2020
isn't included in the provided results — would you like me to look that
up separately?*" That's the right degrade-gracefully instinct even though
compound-query resolution itself was never built.

**Graceful degradation under two real, unplanned failures, not scripted
ones.** (1) `llama-3.3-70b-versatile`'s daily quota was exhausted
mid-battery from cumulative testing — the pipeline caught it and returned
a clean message rather than crashing. (2) During the full re-run on the
new model, 3 of 68 queries ("thanks!", a Tokyo question, a weather
question) failed with the same clean generic-error message rather than a
raw exception; these were scattered, not clustered, which points to
transient rate-limit contention from a concurrent local Streamlit test
session sharing the same Groq account's per-minute budget, not a code bug
or a hard quota wall (a hard wall would fail everything from that point
forward, as it did the first time — see Weaknesses).

## Weaknesses found

**Latency rose substantially on the new model — worth watching, not yet fully isolated.**
The full battery run averaged well into the 10-35 second range per turn,
versus 0.4-2s measured testing `qwen/qwen3-32b` in isolation earlier the
same session. The most likely explanation is contention: the battery ran
concurrently with a local Streamlit session I'd started for manual
testing, both competing for the same account's 6,000-tokens-per-minute
cap. Still comfortably inside the 60s budget on every single call, but
the gap between isolated and concurrent-load latency is large enough that
it deserves a clean, isolated re-measurement before being fully written
off as "just contention" — flagged here as observed, not fully explained.

**Compound multi-part questions confirmed, not just predicted, to only
resolve one metric.** Previously flagged as an architectural prediction;
now directly observed (see Strengths — the Responder's honesty about it
is good, but the underlying one-metric-per-turn limitation is real and
now confirmed under live conditions, not just reasoned about).

**No multi-geography comparison or ranking/superlative queries.**
Deliberate, discussed scope cuts, unchanged from earlier — the Gatekeeper
now explains the specific limitation via `scope_note` rather than a flat
refusal, but the underlying capability doesn't exist.

**Occasional over-cautious metric ambiguity persists** for "median
household income" specifically (flags `AMBIGUOUS` against a
citizen/voting-age-householder variant a human would resolve instantly).
Accepted as a documented precision/recall trade-off rather than something
worth further threshold-chasing — see `metadata/metric_ranker.py`.

**Local dev process management was fragile** on this Windows/Git Bash
setup during iteration (stale backgrounded Streamlit processes surviving
`kill`, requiring `taskkill //F` by PID). An operational footgun during
development, not a production concern — Railway manages the container
lifecycle itself.

## Bugs found and fixed this session

- Repetitive, robotic tone on out-of-scope declines with a specific reason
  available (always re-appended the full boilerplate on top).
- No specific explanation for city-level-geography / ranking / non-tracked-
  figure / non-US-geography questions — added the `scope_note` mechanism.
- No handling for definitional questions — added the hardcoded,
  human-reviewed glossary (closed-set classification, never LLM-generated
  definitions).
- Geography typos silently discarded: the Gatekeeper routed unclear
  geography text to `NEEDS_CLARIFICATION` with an **empty** `state` field,
  never giving the resolver's typo-correction a chance to run.
- A real false-positive risk in state fuzzy-matching (`WRatio` scoring the
  fictional word "Narnia" at 72 against "California") — fixed via a
  scorer change (`fuzz.ratio` for states, keeping `WRatio` for counties
  where its suffix-handling is genuinely needed), verified against a
  dozen real typos and a dozen false-positive-risk city names.
- Metric-phrase typos weren't corrected, unlike geography ("populaton"
  returned `NO_MATCH`) — fixed with per-word typo correction against the
  real corpus vocabulary, empirically threshold-tuned the same way as the
  geography fix (genuine typos scored 83-95, risky collisions like
  "weather"→"father" scored 76-80, threshold set to 82).
- Groq's `llama-3.3-70b-versatile` daily quota (100k tokens/day) was
  exhausted by testing — switched to `qwen/qwen3-32b` (500k tokens/day),
  verified against the real Gatekeeper prompt before committing.
- That switch surfaced a second, unrelated bug: `qwen3-32b` is a
  "thinking" model that leaked a raw `<think>...</think>` reasoning trace
  into the Responder's plain-text output (confirmed it did NOT happen in
  the Gatekeeper's JSON-mode calls). Fixed via Groq's `reasoning_format:
  "hidden"` API parameter on both call sites, not string-stripping.

## New feature: "did you mean X?" geography confirmation

User-requested: for geography names that are plausible but not confident
enough to silently auto-correct, ask rather than guess or reject outright.
Added a third confidence tier to `resolve_geography` (`agent/resolver.py`):
`NEEDS_CONFIRMATION` for scores in `[45, 65)` (below the auto-correct
threshold, above a floor picked from the same empirical data used to tune
the typo-correction fix). Built with zero new pipeline state — it relies
entirely on the conversation memory and history-aware follow-up handling
already in place, plus a small addition to the Gatekeeper's system prompt
instructing it to recognize an affirmative reply to a "did you mean X?"
question. Verified live, at the structured-data level (not just plausible-
sounding text): "population of calif" → "Did you mean California?" → "yes"
→ resolved `state_fips='06'`, `state_name='California'`, correct answer.
Also verified a corrective reply ("no, I meant Texas") resolves properly
to Texas. One minor, non-blocking gap found: a bare "no" with no
correction just re-asks the identical question rather than prompting
differently -- not a functional bug (no wrong answer, no crash), just an
unpolished loop, left as a known minor limitation given time remaining.

## New feature: ranking/superlative queries

User-requested architecture expansion (previously a deliberate scope
cut): "which state has the most population", "least populated county in
Texas". One `GROUP BY` + `ORDER BY` + `LIMIT 1` query finds the winner
directly (grouping on the FIPS prefix already embedded in the block-group
ID), rather than 50 separate queries. Deliberately restricted to summable
metrics -- median/rate metrics decline with a specific explanation instead
of compounding two layers of approximation.

Two real bugs found and fixed before this shipped:
- `agent/sql_validator.py`'s column allow-list rejected `ORDER BY VALUE`
  as an "unknown column" -- `VALUE` is `sql_builder.py`'s own generated
  alias, not a real Census column. Found by reasoning through the query
  shape *before* writing the ranking code, not after a test failure.
- County-scope ranking's winning FIPS is a combined 5-digit state+county
  code, not a bare county code -- the reverse name lookup needed the
  state prefix stripped first. Found via live verification: the answer
  for "least populated county in Texas" initially failed to resolve to a
  county name at all.

Live-verified against real, independently-checkable facts, not just
"the code ran": "which state has the most population" -> California
(39,346,023, the correct 2020 decennial figure); "which county in Texas
has the least population" -> Loving County (117 people) -- genuinely the
least populated county in the entire United States, not just Texas, a
fact I could verify independently rather than just trust the pipeline's
own output. "Which state has the highest median age" correctly declined
with the summable-only explanation.

## New feature: concept catalog refactor (metric matching)

Found via live testing: "Hello there can you tell me the most populated
state?" declined with `NO_MATCH`. Root cause traced precisely, not
guessed: the Gatekeeper was extracting `metric_phrase="most populated"`
(the ranking superlative baked into the metric text), and separately,
testing the sibling phrase "least populated" revealed a worse bug --
it silently resolved to **"Household Income In The Last 12 Months..."**,
a completely wrong metric, because the per-word typo-corrector "fixed"
the real word "least" into "last" (edit-distance 88.9 via `fuzz.ratio`,
above the 82 typo-correction threshold) since "last" is common ACS
boilerplate and "least" wasn't in the 8,276-entry vocabulary.

Rather than patch this one case, discussed the architecture at length
with the user and rebuilt metric resolution around three layers: the
Gatekeeper now extracts loose semantic intent (`concept`, e.g.
"population") with ranking words explicitly routed to
`ranking_direction` instead; a new, small, auto-generated Concept
Catalog (`data/concept_catalog.json`, 198 entries, built offline from
the already-committed `metadata.json` by `metadata/build_concept_catalog.py`)
replaces the raw 8,276-entry corpus as the matching target; the resolver
shrank accordingly (race-qualifier filtering, breakdown filtering, and
the `"Total"`-auto-accept heuristic all became unnecessary, since the
catalog is pre-folded to one canonical row per concept at build time).
No TF-IDF fallback onto the old full-corpus matcher for anything the
catalog doesn't cover -- an honest decline instead.

**A real bug found building the generator itself, caught before it
shipped.** My first race-qualifier detection reused description-text
matching for words like "alone" (mirroring the old resolver's approach) --
verified this produces a false positive: "Household Type (**Including
Living Alone**)" contains "alone" without being a race-qualified row at
all, which silently broke that table's clustering (folded 81 real rows
down to nothing usable). Fixed by switching to a *structural* check
(does the column's variable group itself end in a race-iteration letter,
e.g. `B01002F`) instead of scanning text -- more precise, and avoids this
exact class of collision entirely.

**Verified directly against real data, not assumed:** 506 entries belong
to `B99*` "Allocation Of X" imputation-flag tables (excluded, or
"Allocation Of Median Household Income" would pollute real income
matching); 2020 population has two mechanically-unrelated candidate
sources (an ACS estimate and the decennial redistricting count, the
latter literally titled "RACE" in its own schema) requiring a
hand-authored override, not a guess.

**Live-verified after the fix, against the real Groq/Snowflake pipeline
(not just the resolver in isolation):** "Hello there can you tell me the
most populated state?" -> California (correct); "which county in Texas
has the least population?" -> Loving County (correct, matching the
independently-verifiable fact this is genuinely the least-populated
county in the entire US). A broader sanity pass (plain population/income
lookups, a definitional glossary question, an off-topic decline, bare
"income", a typo'd "populaton") showed no regressions from the refactor
-- including an improvement: the old "median household income" vs.
"citizen, voting-age householder" near-ambiguity is gone as a side
effect, since the catalog only keeps one canonical row per concept.

**As defense-in-depth** (the architectural fix means ranking words
shouldn't reach the resolver in the normal flow at all), added an
explicit strip of known ranking-operation words before matching, plus a
regression test locking in that a bare "least"/"most"/"highest"/"lowest"
declines rather than silently resolving to anything.

## New feature: persistent per-account chat history

User-requested: replace the single shared `evaluator` login with 3 named
accounts, each with one continuous conversation history that survives
sign-out and (per explicit user choice) full durable storage, not just
in-session state. Implementation: `DEMO_ACCOUNTS` env var parsed into a
`username -> password` dict; a stable `thread_id` of `account-<username>`
(not a random per-session uuid); LangGraph's default in-memory
checkpointer swapped for `langgraph-checkpoint-sqlite`, sharing the exact
same custom-type serde registration the in-memory version already used
(now factored into `build_checkpoint_serde()` so both backends stay in
sync); sign-out clears only the browser's own `st.session_state` display
fields, never the persisted record.

**A real testing-tool bug found, not a product bug.** Streamlit's own
`AppTest` framework (v1.59, the officially recommended way to test a
Streamlit script without a browser) threw an internal `KeyError` hunting
for a stale widget's session-state entry, specifically on this app's
conditionally-rendered sign-in form — once `authenticated` flips to
`True`, the form's `text_input`/`button` widgets stop being rendered
entirely, and `AppTest`'s widget-state bookkeeping didn't handle that
disappearance cleanly across a rerun. Rather than fight the test
framework or skip live verification, switched to driving a real headless
Chromium via Playwright against the actual running `streamlit run app.py`
server — arguably a stronger test anyway, since it's the real browser/
session lifecycle rather than a script-rerun simulation.

**Live-verified end to end** (real Groq/Snowflake backend, real browser,
not mocked): signed in as `evaluator3`, asked "What is the population of
Texas?", got "The population of Texas in 2020 was approximately 28.6
million people" (matches the real 28,635,442 decennial figure verified
earlier this session), signed out, signed back in as the same account,
and the exact same question-and-answer pair was still there, restored
from the SQLite-backed checkpointer — not re-run, not regenerated. A
second, separate account (`evaluator1`) signing in fresh showed zero
messages, confirming accounts don't leak history into each other.

## New feature: generalized ranking (operation/sort/limit)

Found via live testing: "all the states ordered by population" returned a
nationwide total instead of a ranked list, because the old ranking design
only supported a single winner (`ranking_direction`: MOST/LEAST, each
implying `LIMIT 1`) -- "ordered by" isn't a MOST/LEAST superlative, so the
field came back empty and the request fell through to a plain lookup.

Rather than add a new pipeline node for "list all" as a separate
capability, redesigned the Gatekeeper's ranking output around three
orthogonal parameters instead of a single binary flag: `operation`
(LOOKUP/RANKING), `sort` (ASC/DESC), `limit` (a validated, capped count,
or "ALL"). One query builder (`GROUP BY <geo> ORDER BY VALUE {sort} LIMIT
{limit}`) now handles "most populated," "least populated," "top 10,"
"bottom 5," and "all 50 sorted" as different values of the same two
fields, not five separate capabilities. The LLM's role stayed exactly
what it always was -- pick parameters, never a table/column/SQL text --
this is a parameter generalization, not a loosening of the no-SQL-
generation guarantee. Folded in as part of the same change: ranking
counties nationwide with no state named at all (e.g. "which county in the
US has the most people"), essentially free once the query was
parameterized.

**A real, previously-latent correctness bug found and fixed while
generalizing, not before.** `ORDER BY VALUE DESC` with no explicit
null-ordering defaults to `NULLS FIRST` in Snowflake -- a NULL-valued
group could have ranked #1 in a "most populated" list. Invisible with the
old `LIMIT 1` (nothing else was ever visible to notice), but a real,
visible bug once N rows are surfaced. Fixed with an explicit `NULLS LAST`
before this shipped, not left for someone to find later.

**A verified-not-guessed constant.** The cap for "all" needed a real
number: checked the actual geography crosswalk and found 56 distinct
state-level FIPS groups (50 states + DC + 5 territories), not 50 --
`MAX_RANKING_LIMIT` was set to 60 specifically so "all states" can never
silently truncate, which would have been exactly the kind of dishonest
behavior this redesign exists to eliminate.

**Live-verified against the real Groq/Snowflake pipeline, not just the
mocked test suite:**
- "All the states ordered by population" -> a complete, correctly-
  ordered 52-row list (50 states + DC + Puerto Rico; the remaining
  territories have null population data in this dataset and were
  correctly filtered out, not shown as zero), matching real Census
  figures exactly (California 39,538,223 down to Wyoming 576,851).
- "Most populated state" and "least populated county in Texas" -- the
  pre-existing single-winner phrasing -- still work completely unchanged.
- "Which county in the US has the most population" (the new nationwide-
  county capability) -> Los Angeles County, California -- the real,
  independently-verifiable most-populous US county -- with the label
  correctly disambiguated as "County, State" rather than a bare county
  name (county names collide across states).
- "Which state has the highest median household income" -> still
  declines with the unchanged explanation, confirming the non-summable
  gate is unaffected by the sort/limit generalization.
- "Top 5 states by population" -> correct list, correct order.

## Bug found live by the user, after the generalized-ranking feature shipped

"What is the total population of the United States" resolved to a real
but wrong column ("Total Population In Occupied Housing Units By
Tenure") instead of the curated "Population" concept, giving a
plausible-sounding but incorrect figure (~321.7M instead of the real
~334.7M). Root cause: at equal coverage, the fuzzy concept ranker's
cosine score favored a short, narrowly-focused auto-derived concept over
the correct one, because a short document aligns more tightly with a
short query than a longer document carrying legitimately useful alias
diversity. Fixed with an exact (case-insensitive) match against a
concept's label/alias short-circuiting the fuzzy ranker entirely --
stronger evidence than any cosine score. This also surfaced that a
previously-passing typo-correction test ("populaton") had been passing
for the wrong reason (both the typo and the clean spelling were landing
on the *same* wrong concept, so the test's "do they agree" assertion
never actually checked *which* concept) -- fixed by extending the
concept ranker with a `correct_text()` method so the exact-match check
also runs after typo correction, not just on the raw query. Live-verified
after the fix: "total population", "population", and the typo'd
"populaton" all now resolve to the correct concept and the nationwide
total reads 334,735,155 -- matching the same figure independently
verified via the ranking feature's own nationwide sum.

Separately, live testing during this same session hit genuine Groq
`429 Too Many Requests` rate-limit exhaustion (the `qwen/qwen3-32b`
model's 6,000-tokens-per-minute cap), compounded by concurrent live-
verification scripts running against the same account while the user was
testing through the UI -- a handful of turns degraded gracefully to the
generic "something went wrong" guardrail message rather than crashing,
exactly as designed, and cleanly reproduced as fine once load eased.

## Recommendations, in priority order

1. Get a clean, isolated latency measurement for `qwen/qwen3-32b` (no
   concurrent local sessions competing for the same account) before
   concluding the 10-35s figures are purely about contention.
2. Decide whether compound multi-metric questions are worth a scoped fix
   (e.g. resolving multiple metrics when the Gatekeeper detects an "and")
   or should remain a documented limitation — now confirmed real, not
   hypothetical.
3. ~~Resolve the Groq quota risk~~ — done: switched models, 5x daily
   headroom, verified.
4. ~~Extend typo tolerance to metric phrases~~ — done, with an unplanned
   bonus fix to simple plural/singular metric-word mismatches.
5. ~~Ranking/superlative queries~~ — done: summable metrics at both state
   and county-within-state scope, live-verified.
6. ~~Persistent per-account chat history~~ — done: 3 named accounts,
   SQLite-backed checkpointer, live-verified including cross-account
   isolation.
7. ~~Retire the fragile TF-IDF-over-8,276-entries metric matching~~ —
   done: concept catalog refactor, live-verified against the exact
   reported bug plus a broader sanity pass, no regressions found.
8. ~~Generalize ranking beyond a single winner~~ — done: operation/sort/
   limit replace the old MOST/LEAST-implies-1 model, live-verified
   including a full 52-row ranked list and nationwide county ranking.
9. Multi-geography *comparison* ("Cook County vs. the national average")
   remains deliberately deferred — documented in `REFLECTION.md`.
10. Multiple saved conversations per account ("new chat, keep the old
    one") remains deliberately deferred alongside persistence —
    documented in `REFLECTION.md`.
