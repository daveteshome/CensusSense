# Reflection

## Development process and key architectural decisions

I started by reading the assignment closely and pressure-testing my own
initial design doc before writing any code — the original plan assumed a
`WHERE STATE='TX'`-style column existed on the fact tables and used
`SUM()` for median household income. Neither survived contact with the
real dataset. I logged into the Snowflake trial and queried
`INFORMATION_SCHEMA` directly rather than assuming a schema: the real
tables (`US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET`) turned
out to be ACS 5-year block-group data (2019, 2020 vintages) plus 2020
decennial redistricting counts, with geography encoded in a
`CENSUS_BLOCK_GROUP` ID (state/county FIPS as a prefix) rather than a
named column, and mixed-case column identifiers (`B19013e1`, not
`B19013E1`) that Snowflake's unquoted-identifier folding would silently
break against.

That discovery drove the core architectural choice: a **metadata-driven,
deterministic pipeline** rather than letting an LLM write SQL directly.
Two Snowflake tables I found mid-build —
`2019/2020_METADATA_CBG_FIELD_DESCRIPTIONS` — turned out to be keyed
exactly by the real Snowflake column name and gave a human-readable
breadcrumb (table title + field hierarchy), which was more precise than
bridging through the external Census Bureau API dictionary I'd originally
planned to use. I built `metadata/build_metadata.py` to join that against
`INFORMATION_SCHEMA.COLUMNS` once, offline, producing a single
`metadata.json` (8,276 entries) that the app loads at startup — no live
Snowflake dependency for query resolution itself, only for execution.

The pipeline is: **Gatekeeper (LLM, structured JSON output) → resolver
(deterministic: year/geography/metric) → SQL builder (deterministic,
only ever accepts already-verified metadata) → SQL validator
(independent allow-list re-check via `sqlglot`, defense-in-depth against
prompt injection regardless of what the Gatekeeper decided) → Snowflake
executor → Responder (LLM, given only the already-computed number, never
free-generates a fact)**. LangGraph wires this as a graph with conditional
edges to a shared `guardrail` node for every non-happy-path case, and its
checkpointer gives conversation memory (a `thread_id` persists resolved
slots across turns — "population of Texas" → "how about California?"
correctly carries the metric forward). That checkpointer is now
SQLite-backed and keyed by a stable per-account thread_id rather than a
random per-browser-session one, so history survives sign-out/sign-in and
server restarts — see the persistence write-up below.

The metric-matching approach changed under evidence, not intuition: I
tried plain fuzzy string matching (rapidfuzz) first, and it failed
badly — "average salary of software engineers" (which should be
unanswerable) scored 85.5 against an unrelated geographic-mobility field,
because character-level similarity over ~4,000 verbose, boilerplate-heavy
ACS descriptions rewards shared filler words ("Total", "In The Past 12
Months") over actual topical relevance. I switched to TF-IDF/cosine with
a coverage-first ranking (documents must share the query's *distinctive*
words, not just score well on edit distance), which fixed it — verified
by testing before/after, not just asserting it was better.

**LLM provider took three attempts.** I designed for Gemini first; the
API key hit a hard `429 quota=0` and 404s for deprecated model versions on
one project, then a working key hit 15-26s latency with a 503 mid-session.
A second, older key worked well (`gemini-2.5-flash`, sub-1s), but the user
didn't want to pay Google's required minimum billing top-up. I switched to
Groq (free tier, no card required, OpenAI-compatible API so the standard
`openai` SDK worked unmodified) running `llama-3.3-70b-versatile` — fast
(0.2-2s typical) and free, at the cost of a tighter daily token quota that
became the biggest late-session finding (see below).

## What I would improve or do differently with more time

**Ranking/superlative queries** ("which state has the highest population",
"least populated county in Texas") were originally scoped out as a
deferred feature, then built later in the session once there was time:
one `GROUP BY` + `ORDER BY` + `LIMIT 1` query (grouping on the FIPS prefix
already embedded in the block-group ID) finds the winner without a join or
50 separate queries. Deliberately restricted to summable metrics
(population, housing counts) -- ranking 50+ already-approximate
AVG-of-block-group-medians numbers against each other would compound one
approximation on top of another, so median/rate metrics still decline
with a specific explanation. Building this surfaced a real, separate bug
in `agent/sql_validator.py`: the query's own `SELECT ... AS VALUE` alias,
referenced again in `ORDER BY VALUE`, was being flagged as an "unknown
column" by the allow-list check, since the validator didn't yet
distinguish a self-generated alias from a real Census column -- fixed by
allow-listing the small set of aliases `sql_builder.py` itself generates.
Live verification also caught a second real bug (county-scope ranking
returns a combined 5-digit state+county FIPS, not a bare county FIPS, and
the reverse name lookup needed the state prefix stripped first) before
it shipped -- confirmed via real answers I could independently verify
(California as the most populous state; Loving County, TX, which is
genuinely the least populated county in the entire US, not just Texas).

**Persistent per-account chat history** was also originally scoped as
in-session-only, then built later once there was time. The single shared
`evaluator` login became 3 distinct named accounts (`DEMO_ACCOUNTS`,
parsed into a `username -> password` dict in `config.py`), each mapped to
a stable LangGraph `thread_id` (`account-<username>`) rather than a random
per-session uuid, backed by `langgraph-checkpoint-sqlite` instead of the
default in-memory saver. Signing out only clears the browser's own
`st.session_state` display fields; the real conversation lives server-side
keyed by that thread_id, so signing back in with the same account replays
it in full via `pipeline.get_state(...)` — verified live with a real
headless-browser session (not mocked): asked "What is the population of
Texas?", got the real answer, signed out, signed back in with the same
account, and the exact prior turn was still there; a second, different
account signing in fresh correctly saw no history at all, confirming
isolation. One genuine testing-methodology finding here: Streamlit's own
`AppTest` framework (1.59) turned out to error on this app's specific
shape — a login form that fully unmounts once authenticated — with an
internal `KeyError` hunting for a stale widget's session-state entry. Real
bug in AppTest's bookkeeping for conditionally-rendered widget trees, not
in the app, but it meant I couldn't use Streamlit's own recommended
testing tool for the live check and switched to driving a real headless
Chromium via Playwright against the actual running server instead — which
arguably verifies the feature more convincingly anyway, since it exercises
the real browser/cookie/session lifecycle rather than Streamlit's script-
rerun simulation. The one deliberately-not-built adjacent feature: multiple
saved conversations per account (a "new chat, keep the old one" picker) —
doesn't combine cleanly with "one continuous restored history" and was
scoped out given remaining time. And on Railway specifically, the SQLite
file lives on the container's ephemeral local disk, so it survives
sign-out/sign-in and in-place restarts but not a redeploy unless a Railway
Volume is mounted — a dashboard action I've flagged in the README rather
than silently claiming full survival.

**A real bug led to an architectural rethink of metric matching, not
another threshold patch.** Live testing found "which state is the most
populated?" failing: the Gatekeeper was handing the resolver
`metric_phrase="most populated"` (the ranking superlative baked into the
metric text itself), so it correctly declined -- but root-causing it
surfaced a worse sibling bug: "least populated" didn't decline, it
silently resolved to a *wrong* metric ("Household Income..."), because
the per-word rapidfuzz typo-corrector "fixed" the real, correctly-spelled
word "least" into "last" (edit-distance 88.9, above its 82 threshold),
since "last" is extremely common ACS boilerplate ("...In The Last 12
Months...") and "least" wasn't in the 8,276-entry vocabulary at all. I
discussed this at length with the user, who pushed back hard (correctly)
on my first instinct to just add another special case: the underlying
problem is that matching free-form natural language against 8,276 raw,
verbose ACS column descriptions is inherently fragile, and I'd already
been patching around it all session (a typo threshold, a stopword list, a
phrase-match bonus, race-qualifier filtering, breakdown filtering) without
addressing why it kept needing patches.

The user's proposed fix, which I implemented: split metric resolution
into three layers instead of one. The **Gatekeeper** now extracts loose
semantic intent (a `concept` field, e.g. "population") with any ranking
word explicitly instructed to go in `ranking_direction` instead, never
`concept` -- and critically, the Gatekeeper's prompt never sees the
catalog at all, so this didn't re-couple schema knowledge into the prompt
(a coupling problem the user specifically flagged and asked me to avoid).
A new, small **Concept Catalog** (`data/concept_catalog.json`, ~198
entries, built offline by `metadata/build_concept_catalog.py` from the
already-committed `metadata.json`) replaces the raw 8,276-entry corpus as
the actual matching target. The **resolver** shrank substantially --
`resolve_metric` now just matches against ~198 clean, curated documents
instead of 8,276 overlapping ones, and the old race-qualifier/breakdown-
filtering/`"Total"`-auto-accept logic it needed is gone, because the
catalog is pre-folded to one canonical row per real concept at build time
instead. No TF-IDF fallback for anything the catalog doesn't cover --
per the user's explicit call, two matching systems that can silently
disagree is a worse failure mode than one system with an honest,
documented edge (a clean decline: "this version supports headline Census
metrics, not fine-grained demographic breakdowns").

Building the generator surfaced several real findings I verified directly
against the data rather than assuming: grouping by raw Snowflake table
name doesn't cleanly separate concepts (one table can bundle several,
e.g. "Total Population" and "Median Age By Sex" in the same physical
table) -- the real grouping key is the ACS variable group, with race-
iteration letters folded off *structurally* (checking whether the column
group itself ends in a race-iteration letter) rather than by scanning
description text for words like "alone", which I initially tried and
found produces a real false positive: "Household Type (**Including
Living Alone**)" contains "alone" without being a race qualifier at all,
silently breaking that table's clustering until caught and fixed. 506
entries turned out to belong to `B99*` "Allocation Of X" tables -- Census
data-quality/imputation flags, not real metrics -- and needed a hard
exclusion, or "Allocation Of Median Household Income" would have polluted
matching for genuine income queries. And a real, non-obvious collision:
2020 population has two mechanically-unrelated candidate sources (an ACS
estimate and the decennial redistricting count, the latter literally
titled "RACE" in its own schema since it's the base row of the
race-breakdown table) that no automatic rule could ever unify -- this
needed a hand-authored override, verified against real column
descriptions, not a guess. As defense-in-depth (since the architectural
fix means ranking words shouldn't reach the resolver at all in the normal
flow), I also added an explicit strip of known ranking-operation words
before matching, and a regression test locking in that a bare "least" or
"most" declines rather than resolving to anything.

Live-verified after the fix: "Hello there can you tell me the most
populated state?" now correctly answers California; "which county in
Texas has the least population?" now correctly answers Loving County --
both against the real Groq/Snowflake pipeline, not just the resolver in
isolation. A broader sanity pass (population, income, definitional
questions, off-topic questions, bare "income", a typo'd query) showed no
regressions from the refactor.

**Ranking itself then needed generalizing, and the fix taught me something
about how I'd been thinking about the LLM's role.** Live testing found a
real gap: "all the states ordered by population" returned a nationwide
total instead of a ranked list. Root cause: `ranking_direction` only had
two values (MOST/LEAST), each secretly meaning "return exactly 1 row" via
a hardcoded `LIMIT 1` -- "ordered by population" isn't a MOST/LEAST
superlative, so the field came back empty and the whole request fell
through to a plain lookup. My first instinct was to treat this as a
missing capability needing new pipeline nodes. The user pushed back, and
correctly: "most populated," "least populated," "top 10," "bottom 5," and
"all 50 sorted" are all the *same* SQL shape (`GROUP BY <geo> ORDER BY
VALUE {ASC|DESC} LIMIT N`) with only the sort direction and row count
changing -- the Gatekeeper should be a *planner* extracting parameters,
not a parser needing a new rule per phrasing.

This led to a real, if brief, disagreement I want to record honestly. The
user's initial framing was "the LLM should build the query" -- I pushed
back hard on that specific phrasing, because it's the one line this
entire architecture has held since the first design decision: the LLM
never picks a table, a column, or writes SQL, and the SQL validator's
allow-list defense only means anything because of that. We converged on
the actual right answer, which turned out to satisfy both concerns at
once: generalize the *parameters* (`operation`: LOOKUP/RANKING, `sort`:
ASC/DESC, `limit`: a validated, capped count), not the SQL. The LLM picks
a direction and a count -- never a table/column/SQL text -- and one
query builder handles every phrasing of "rank/sort/list" as a value of
those two fields instead of a new code path. Folded into the same
generalization, since it was essentially free once the query was
parameterized: ranking counties *nationwide* with no state named at all
("which county in the US has the most people"), not just within one
named state.

Building this surfaced two real things worth naming. First, a
correctness bug that was already latent and invisible: `ORDER BY VALUE
DESC` without an explicit null-ordering clause defaults to `NULLS FIRST`
in Snowflake, so a NULL-valued group could theoretically have ranked #1
in a "most populated" list -- harmless with the old `LIMIT 1` (nothing
else was ever visible to notice), but a real, visible bug once N rows
are surfaced. Added `NULLS LAST` explicitly rather than leaving it to
find later. Second, the cap for "all" needed an actual verified number,
not a guess: I checked the real geography crosswalk and found 56 distinct
state-level FIPS groups (50 states + DC + 5 territories), not 50 --
`MAX_RANKING_LIMIT` had to exceed that or "all states" would silently
truncate exactly the kind of way this whole redesign was trying to make
honest and visible.

Live-verified end to end against the real Groq/Snowflake pipeline: "all
the states ordered by population" now returns a complete, correctly-
ordered 52-row list (50 states + DC + Puerto Rico, the territories with
null population data correctly filtered out) matching real Census
figures (California 39,538,223 down to Wyoming 576,851); "most populated
state" and "least populated county in Texas" still work exactly as
before, unchanged single-winner phrasing; "which county in the US has
the most population" correctly answers Los Angeles County, California
(the real most-populous US county, with distinct-state disambiguation in
the label -- a real design point, since county names collide across
states); a median-type metric still declines with the unchanged
explanation regardless of sort/limit.

The `operation` field was deliberately designed as a small, closed enum
with room to grow rather than a one-off `is_ranking` boolean --
`LOOKUP`/`RANKING` are the only two built and verified today, but the
same "LLM fills in parameters, never SQL" pattern extends naturally to
future operations I didn't build: `TREND` (a metric across multiple
years for one geography, e.g. "how has Texas's population changed since
2019" -- the dataset only has 2019/2020 vintages, so this would be thin
today, but the shape is there), `FILTER` (geographies matching a
threshold, e.g. "which states have a median income over $70k"), and the
`COMPARISON` case that's currently a clean decline ("Cook County vs. the
national average") could become a real third operation instead of a
guardrail message. Each would need its own SQL template and Responder
phrasing, but none would need the Gatekeeper to see schema or write SQL
-- the safety property holds by construction, not by convention.

**Testing the generalized ranking end to end surfaced a real, unrelated
bug in the concept catalog itself, found by the user live, not by me.**
"What is the total population of the United States" resolved to "Total
Population In Occupied Housing Units By Tenure" -- a real column, but
the wrong one -- instead of the curated "Population" concept, and the
answer read as a plausible-sounding but wrong number (~321.7 million
instead of the real ~334.7 million). Root-caused precisely: at equal
coverage (1.0), the fuzzy ranker's cosine score favored the short,
narrowly-focused auto-derived concept over the correct one, because a
short document naturally aligns more tightly with a short query than a
longer document carrying more (legitimately useful) alias diversity --
the same class of length-bias problem coverage-first ranking was
originally built to fix, just resurfacing at a tie rather than a clear
win. My first attempted fix (deduping a redundant alias that exactly
repeated the label) turned out to be a mathematical no-op -- L2
normalization makes uniformly duplicating a whole document's text
invisible to cosine similarity, which I verified directly with a 3-line
calculation before trusting the fix. The actual fix: an exact
(case-insensitive) match against a concept's own label or a curated
alias is stronger evidence than any fuzzy cosine score, so `resolve_metric`
now short-circuits directly to that concept when the query matches one
exactly, skipping the fuzzy ranker's document-length bias entirely. This
in turn broke a *different*, previously-passing test: a typo'd query
("populaton") no longer benefited from the new short-circuit (it doesn't
match anything verbatim) and fell back to the same wrong-concept bug the
fix was supposed to close -- revealing that the old passing test had
been accidentally correct for the wrong reason (both the typo and the
clean spelling had been landing on the *same* wrong concept, so the
test's narrow "do they agree" assertion passed without ever checking
*which* concept). Fixed by extending the ranker with a public
`correct_text()` method so the exact-match check can also run after
typo-correction, not just on the raw query. Also had to recover from a
real self-inflicted process mistake while debugging this: I ran `git
stash` to isolate a test case and it silently reverted every uncommitted
change from the entire session, not just the one file I meant to
isolate -- caught immediately (the stash contents were still fully
recoverable) and restored with `git stash pop`, but it's a reminder that
`git stash` with no path argument touches the whole working tree, not
just what you're looking at.

**A genuine Groq rate-limit failure, live, turned out to be self-
reinforcing in a way worth understanding precisely rather than just
retrying.** The user hit "something went wrong" repeatedly on the exact
same query. I traced the full pipeline state rather than assuming
rate-limiting was the whole story: the resolver, geography, and metric
resolution all succeeded correctly every time -- the failure was
isolated to the Responder call hitting Groq's 6,000-tokens-per-minute
cap. But *why* the same query kept failing specifically, not just
occasionally: `guardrail_node` was persisting the generic failure
message into that account's conversation history exactly like a real
answer, and since both LLM prompts include the last 5 turns for context,
a thread that had already failed once was carrying that dead-weight
failure text into every subsequent attempt's prompt -- shrinking the
token budget actually available and making the *next* attempt more
likely to collide with the same rate limit, not less. Verified this
concretely: `account-evaluator3`'s persisted history had accumulated
five consecutive "something went wrong" turns in a row, each one making
the next retry of the same question worse. Fixed narrowly: a guardrail
message triggered by a genuine transient error (`state["error"]`) is
shown to the user but not persisted into history; deliberate guardrail
decisions (declines, clarifications, "did you mean X?") still are, since
those genuinely are meaningful context for a natural follow-up turn.

**A Snowflake connector concurrency bug surfaced only after deploying to
Railway, not in local testing.** The executor cached a single Snowflake
connection at module scope and reused it across every query in the
process, on the reasoning that opening a fresh connection per query was
wasted latency. That's safe for one person testing locally but not for a
Streamlit app serving multiple signed-in accounts from the same process:
two calls reaching the shared connection within a short window of each
other (not necessarily truly concurrent -- a different account active a
few seconds later was enough) collided on the same underlying Snowflake
session. The user reported "something went wrong" recurring across
different accounts and different questions on the deployed app; two
rounds of pasted Railway logs made the mechanism concrete rather than
theoretical -- the connector itself logging `Session with id X is
already connected! Connecting to a new session.`, and a clear
before/after pair where `account-evaluator3` succeeded on a question and
`account-evaluator1` failed on the same question roughly 30 seconds
later. Fixed by removing the cached connection entirely: `execute()` now
opens a dedicated connection per call and always closes it in a
`finally` block, on both the first attempt and the retry, so no two
callers can ever share a session (`agent/executor.py`). The latency cost
is sub-second and worth it for correctness in a multi-account deployment.

**A county mentioned without its state was silently answered as
nationwide, not declined.** Found by accident while re-verifying the
connection fix above: asking "median household income in Travis County"
(no "Texas" in the question) returned $72,918 -- the real *nationwide*
median -- while the answer text still said "Travis County" (Travis
County's real value is $92,616, confirmed by running both the filtered
and unfiltered SQL directly against Snowflake). Traced to
`resolve_geography`: `gk.state` was an empty string (the Gatekeeper
correctly hadn't invented a state the user never said), and
`resolve_geography(gk.state or None, gk.county or None, store)` turned
that empty string into `None`, hitting the function's original
short-circuit for "no state mentioned → nationwide, no filter" -- which
discarded the county entirely without even looking at it. This is worse
than a plain decline: the response *sounds* specific and confident but
answers a different question than the one asked. Most county names
aren't nationally unique ("Washington County" exists in 31 states), so
the fix couldn't just silently pick a state either. `resolve_geography`
now searches every state's county list when no state is given: exactly
one plausible match resolves directly (`"Travis County"` is unique to
Texas), more than one asks which state instead of guessing
(`AMBIGUOUS_COUNTY`), and none found declines rather than falling back to
nationwide. The naive version of this cross-state search was itself
briefly a false-positive generator worth naming: `fuzz.WRatio` against
all ~3,100 counties at once matched a wholly nonsense query
("Xyzzyplex County") in 23 different states, because nearly every county
name shares the same " County" suffix and WRatio's partial-match scoring
rewards that shared suffix on its own. Stripping the trailing
designator word ("County"/"Parish"/"Borough"/etc.) before comparing, then
scoring the core name with plain `fuzz.ratio`, fixed it -- verified
empirically before settling on the threshold, the same way the state-name
matcher's threshold was tuned earlier in the project.

**Multi-geography *comparison*** ("How does Cook County compare to the
national average?") is a different, still-unsupported case from ranking
-- it needs two independently resolved geographies compared side by side
in one answer, not a single winner picked from a group. That's a real
feature addition (Gatekeeper changes to detect comparison intent, running
two queries, a Responder change to synthesize across them), and I chose
not to build it given the remaining time before submission. The
Gatekeeper explains specifically why via `scope_note` ("I can only look
up a metric for one geography at a time, I don't compare across two named
geographies") rather than a flat refusal.

**City/place-level geography isn't supported.** The dataset is block-group
level; there's no table mapping incorporated places (cities) to block
groups in what I found. With more time I'd look for a TIGER/Line
place-boundary crosswalk to layer on top, though it's a genuinely separate
data-acquisition problem, not just more engineering time on what's here.

**The Groq free-tier daily quota was a real risk I hit mid-build, not a
hypothetical one.** `llama-3.3-70b-versatile`'s free tier caps at 100,000
tokens/day, and my own testing exhausted it during this session — at
roughly 1,000-1,500 tokens per conversation turn, that ceiling is only
~60-70 total turns per day shared across everyone hitting the deployed
URL, which was genuinely too tight for an evaluation window with multiple
reviewers. Rather than just documenting the risk and hoping it didn't come
up, I switched models to `qwen/qwen3-32b` (500,000 tokens/day on Groq's
free tier, 5x the headroom), verifying it against the real Gatekeeper
prompt before committing to it. That switch surfaced a second, unrelated
bug: `qwen3-32b` is a "thinking" model that leaks its raw
`<think>...</think>` reasoning trace into plain-text completions — it
happened to not show up in the Gatekeeper's JSON-mode calls, only the
Responder's plain-text ones, which I only caught because I looked at the
actual output text rather than trusting the happy-path unit tests (which
mock the LLM response and would never have caught this). Fixed via Groq's
`reasoning_format: "hidden"` parameter rather than post-hoc string
stripping.

**Compound multi-part questions** ("population and median age of Rhode
Island") only resolve one metric per turn (whichever the Gatekeeper's
single `metric_phrase` extraction picks) — confirmed live, not just
predicted: asking exactly that question resolved only median age. The
Responder's behavior on it was better than I expected, though: rather than
silently presenting that as the whole answer, it said so explicitly
("population data wasn't included in the provided results — would you
like me to look that up separately?"). That's the right instinct even on
a capability I never explicitly built, but the underlying limitation
(one metric per turn) is real and would need a genuine pipeline change —
detecting multi-metric intent, resolving each separately, executing
multiple queries — to actually fix, not a quick patch.

**Metric ambiguity is occasionally over-cautious.** "Median household
income" alone sometimes flags `AMBIGUOUS` against a "...for households
with a citizen, voting-age householder" variant of the same base metric —
a real, close TF-IDF score a human would resolve instantly by picking the
obviously-more-general one. I tuned several similar cases out (race/
ethnicity iteration tables, "Total" vs. "Male"/"Female" sub-variants,
literal-phrase-match boosting) but stopped chasing this particular one
rather than keep narrowing thresholds against an ever-smaller set of
remaining edge cases with diminishing returns and rising risk of
overfitting the ranker to specific test phrases.

**If I were rebuilding this from scratch knowing what I know now**, I'd
verify the actual Snowflake schema and pull real column/description data
*before* writing the original design doc at all, rather than writing a
plausible-sounding design and then discovering it didn't match reality —
though arguably that's a reasonable one-two punch for a 24-hour project:
sketch fast, verify immediately, revise before committing real
implementation time.

## Edge cases and failure modes identified but not fully addressed

- Multi-geography *comparison* (above) — explicitly deferred, not
  silently missing. Ranking/superlative queries and persistent per-account
  history, originally in this same category, were both built later in the
  session (see above).
- Multiple saved conversations per account ("new chat, keep the old
  one") — explicitly deferred alongside persistence, not silently missing.
- Persisted history depends on the SQLite file surviving on disk; on
  Railway that means a Volume needs to be mounted, or a redeploy resets
  everyone's history (documented in the README, not a silent gap).
- Compound multi-metric questions in a single message — found late,
  unverified.
- The old "median household income" vs. "citizen, voting-age householder"
  near-ambiguity is gone as a side effect of the concept catalog refactor
  (the catalog only ever keeps one canonical row per real concept, so
  that near-duplicate variant isn't a competing candidate anymore) —
  verified live, not just assumed: "median household income" now resolves
  without asking for clarification.
- A known, low-practical-impact residual risk in metric-word typo
  correction, unchanged by the refactor: "country" and "county" are a
  genuine 1-character edit apart (92.3% similarity). A bare, isolated
  "country" string does still get corrected toward a county-related
  concept if it somehow reached the resolver directly — verified this is
  still theoretically true against the new, smaller catalog too — but in
  practice "country" is caught by the Gatekeeper's glossary/out-of-scope
  handling before a bare concept ever reaches the resolver (verified live:
  "what is a country" hits the glossary, "population of the country"
  correctly extracts concept="population", not "country"). Naming the
  risk rather than claiming the mitigation is airtight against every
  conceivable input.
- The concept catalog covers ~200 curated headline concepts, not the full
  ~8,300 raw variables — documented in the README's Known limitations,
  not a silent gap; a fine-grained demographic slice declines cleanly with
  a scope-cut message instead of a guess.
- Local dev server process management was fragile on Windows/Git Bash
  during this build (backgrounded processes didn't reliably die on `kill`,
  leaving stale servers on the same port more than once) — purely an
  operational annoyance during development, not a production concern since
  Railway manages the container lifecycle itself, but worth naming since
  it did cause me to test against stale code at least twice mid-session.

## Testing approach and what I'd add

**235 automated tests**, unit + integration, all mocked at the LLM/
Snowflake boundary so the suite runs in a few seconds with no live
credentials: normalization/ID-transform round-trips against the real,
live-fetched Census dictionary (not synthetic fixtures); SQL builder and
validator correctness (including that a median-type metric is never
silently `SUM`med, and that the validator independently rejects anything
outside the metadata allow-list — a prompt-injection defense test, not
just a happy-path test); resolver behavior against both small fixtures
*and* the real, committed metadata/concept catalog (deliberately — the
"Narnia scores 72 against California" false-positive, and the "least"→
"last" wrong-metric bug, only reproduced against real data, since a tiny
fixture doesn't reproduce a large corpus's collision surface);
`tests/unit/test_build_concept_catalog.py`'s coverage of the cluster-
selection algorithm itself (singleton/race-fold/total-fold/unresolvable-
excluded cases) against small synthetic rows, independent of data
changes; the full LangGraph pipeline's routing through every guardrail
path (out-of-scope, inappropriate/injection, ambiguous, no-match,
unsupported year, unknown geography, empty result, executor failure) with
mocked Gatekeeper/Snowflake responses; the generalized ranking builder
(both scopes, both sort directions, limit clamping at every boundary,
`ALL` resolving to the verified cap, nationwide county ranking with no
state named, multi-row answers preserving exact given order) plus the
non-summable decline path unaffected by sort/limit; conversation-memory
threading and cross-session isolation
through the actual compiled graph, not just the `Turn` dataclass in
isolation; and, for the SQLite-backed checkpointer specifically, history
surviving a simulated process restart (a fresh sqlite3 connection against
the same db file, not just the same in-memory object), per-account
isolation, and a second turn after "restart" correctly seeing the first
turn's history (`tests/integration/test_persistence.py`).

**Beyond the automated suite**, I ran extensive live testing throughout
development against the real Snowflake trial and real Groq API — this is
where most of the actual bugs were found, not in the unit tests: the
mixed-case Snowflake identifier quoting issue, the median-aggregation
statistical error in my own first design doc, the TF-IDF ranking failure
on nonsense queries, the Gatekeeper routing typo'd geography to
`NEEDS_CLARIFICATION` with an empty `state` field instead of letting the
resolver's correction run, and the `Narnia`→`California` false-positive
in fuzzy state matching. Every threshold in this codebase (metric-match
coverage cutoff, state-typo-match cutoff, metric-typo-match cutoff) is
set from an actual before/after table of real query scores against real
data, documented inline in the code, not picked by feel. The persistence
feature got the same live-verification treatment: a real headless-browser
session against the real running app (real sign-in form, real chat
input, real Groq/Snowflake round trip), signing out and back in and
confirming the exact prior turn reappeared, plus a second browser session
confirming a different account started with no history at all.

**What I'd add with more time:**
- A CI-safe, opt-in live smoke test suite (`RUN_LIVE_SNOWFLAKE_TESTS=1`,
  already scaffolded as a config flag but not fully built out) that
  exercises the real Snowflake connection and a real (rate-limited) LLM
  call on a schedule, to catch drift if the underlying Snowflake schema or
  Groq's model availability changes.
- Load/concurrency testing — the current rate limiter is a simple
  per-session token bucket, untested under genuinely concurrent multi-user
  load.
- Broader, more adversarial red-teaming of the Gatekeeper's
  `INAPPROPRIATE`/prompt-injection classification beyond the handful of
  manual attempts I tried (basic "ignore your instructions", DAN-style
  jailbreak, credential-exfiltration requests, a SQL-injection-flavored
  string) — all correctly caught, but a handful of manual tries is not the
  same as systematic coverage.
- Automated tests for the exact live-battery categories I ran manually
  this session (greetings, typos, glossary terms, adversarial prompts,
  multi-turn context) as a proper mocked integration suite, rather than a
  one-off script — I ran ~68 of these live and found three real bugs, which
  strongly suggests this category of test has a good bug-per-test-written
  ratio and deserves to be permanent, not disposable.

## A note on AI tool usage

This was built end-to-end with Claude Code, used as an active pair-
programming partner rather than a one-shot code generator: it drafted an
initial architecture, I pushed back and had it verify assumptions against
the live Snowflake schema before committing to them, we iterated on
several genuine bugs together (the fuzzy-matching false positives, the
Gatekeeper's mishandling of typo'd geography, the LLM provider pivots) with
me directing priority and reviewing tradeoffs at each step, and it ran a
live testing pass at my direction that surfaced the Groq quota exhaustion
as a real, unplanned finding rather than a scripted one. I'm comfortable
discussing any part of this build in more depth.
