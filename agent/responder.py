"""Second LLM call in the pipeline: turns a already-executed, already-
aggregated query result into a natural-language answer. Never sees raw
multi-row dumps -- our queries are always single-row aggregates -- and
never invents numbers: the actual value always comes from `result`, the
model's only job is phrasing, unit context, and surfacing caveats.
"""

from dataclasses import dataclass
from typing import Optional

from agent.llm_client import LLMError, MODEL_NAME, get_client
from agent.state import Turn, format_history
from config import Config

_SYSTEM_PROMPT = """You are the response-writing layer of a Census data chat agent. \
You are given: the user's question, recent conversation history, and a single \
aggregate query result already computed from official US Census Bureau data \
(ACS 5-year estimates or 2020 decennial counts). Write a short, clear, \
conversational answer.

Rules:
- Use ONLY the number(s) given to you. Never invent, adjust, or estimate a \
figure yourself.
- State the figure once, clearly. Do not repeat the same number again in a \
different phrasing later in the answer.
- If a caveat is provided in the facts below (e.g. about approximation or a \
defaulted year), include it plainly. Do NOT add any other caveat, disclaimer, \
or mention of margin of error / data quality that wasn't explicitly given to \
you -- only state what you were told.
- Round large numbers to something readable (e.g. "about 28.3 million") while \
still being accurate; for dollar amounts, format as currency.
- Keep it to 1-3 sentences for a single-result answer (including a single- \
winner ranking). For a multi-result ranking list, a one-sentence intro \
followed by the list is fine -- no headers, but a numbered/bulleted list is \
allowed for these specifically (see the ranking rules below).
- Do not mention SQL, tables, columns, or internal system details.
- If the facts describe a RANKING answer with exactly one result, phrase it \
as a single sentence naming that geography (e.g. "The state with the most \
population is California, with about 39.3 million people."), same as a plain \
lookup.
- If the facts describe a RANKING answer with multiple results, present them \
as a short numbered or bulleted list, in EXACTLY the order given -- never \
re-sort, re-rank, renumber, recompute, invent, drop, or reorder entries. A \
one-sentence intro followed by the list is fine -- do not artificially \
compress a 10+ item list into 1-3 sentences to satisfy the length rule above. \
If a truncation note is present in the facts, mention briefly that the list \
was capped, without dwelling on it or apologizing."""


@dataclass(frozen=True)
class RankingRow:
    label: str  # already-resolved display name, e.g. "California" or "Cook County, Illinois"
    value: float


@dataclass(frozen=True)
class ResponderInput:
    question: str
    metric_description: str
    value: Optional[float]
    block_group_count: int
    year: str
    state_name: Optional[str]
    county_name: Optional[str]
    is_approximate: bool
    caveat: Optional[str]
    year_was_defaulted: bool
    ranking_rows: Optional[list[RankingRow]] = None
    ranking_sort: Optional[str] = None  # "ASC" | "DESC" | None
    ranking_truncated: bool = False


def generate_answer(payload: ResponderInput, history: list[Turn], cfg: Config) -> str:
    geography = payload.county_name or payload.state_name or "the United States"
    facts = [
        f"Question: {payload.question}",
        f"Metric: {payload.metric_description}",
        f"Geography: {geography}",
        f"Year: {payload.year}" + (" (defaulted -- user did not specify a year)" if payload.year_was_defaulted else ""),
        f"Value: {payload.value}",
        f"Number of Census block groups aggregated: {payload.block_group_count}",
    ]
    if payload.is_approximate:
        facts.append(f"Caveat to include: {payload.caveat}")
    if payload.ranking_rows:
        direction_word = "highest" if payload.ranking_sort == "DESC" else "lowest"
        if len(payload.ranking_rows) == 1:
            row = payload.ranking_rows[0]
            facts.append(
                f"This is a RANKING answer: {row.label} is the specific geography with the "
                f"{direction_word} value of this metric ({row.value}), out of all the ones compared."
            )
        else:
            other_direction = "lowest" if direction_word == "highest" else "highest"
            listing = "\n".join(f"{i}. {row.label}: {row.value}" for i, row in enumerate(payload.ranking_rows, start=1))
            facts.append(
                f"This is a RANKING answer with {len(payload.ranking_rows)} results, sorted from "
                f"{direction_word} to {other_direction} (already in the correct order, do not re-sort):\n{listing}"
            )
            if payload.ranking_truncated:
                facts.append(
                    f"Note: there are more matching geographies than shown; only the top "
                    f"{len(payload.ranking_rows)} are listed. Mention this briefly."
                )

    prompt = (
        f"Conversation so far:\n{format_history(history)}\n\n"
        f"Data to answer with:\n" + "\n".join(facts)
    )

    client = get_client(cfg)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            # qwen3-32b is a "thinking" model that otherwise leaks a raw
            # <think>...</think> reasoning trace into plain-text
            # completions (verified: it does NOT happen in the
            # Gatekeeper's JSON-mode calls, only here). This suppresses
            # it at the API level rather than string-stripping after the
            # fact.
            extra_body={"reasoning_format": "hidden"},
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - normalized for guardrails.py to catch
        raise LLMError(f"response generation failed: {exc}") from exc
