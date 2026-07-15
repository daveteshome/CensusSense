from dataclasses import dataclass
from typing import Optional


@dataclass
class Turn:
    """One resolved conversation turn, checkpointed by LangGraph. Only
    resolved slots + the short answer are kept -- never raw SQL or full
    result rows -- so prompts stay bounded over a long conversation."""

    question: str
    concept: Optional[str] = None
    state: Optional[str] = None
    county: Optional[str] = None
    year: Optional[str] = None
    answer: Optional[str] = None


def format_history(turns: list[Turn], max_turns: int = 5) -> str:
    # A Turn field rename (e.g. this dataclass's own metric_phrase ->
    # concept migration) can leave old persisted checkpoints holding a
    # Turn shape the current deserializer can't reconstruct -- LangGraph's
    # serde returns None for those rather than raising. Skip them rather
    # than crash on a stale entry from before a schema change; this never
    # masks a real bug since newly-appended turns are always a proper
    # Turn, built fresh by _turn_from_state.
    turns = [t for t in turns if t is not None]
    if not turns:
        return "(no prior conversation)"
    lines = []
    for t in turns[-max_turns:]:
        resolved = ", ".join(
            f"{label}={value}"
            for label, value in [
                ("metric", t.concept),
                ("state", t.state),
                ("county", t.county),
                ("year", t.year),
            ]
            if value
        )
        lines.append(f'User asked: "{t.question}" (resolved: {resolved or "none"}) -> {t.answer or "(no answer)"}')
    return "\n".join(lines)
