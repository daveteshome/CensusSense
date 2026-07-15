from dataclasses import dataclass
from typing import Optional


@dataclass
class Turn:
    """One resolved conversation turn, checkpointed by LangGraph. Only
    resolved slots + the short answer are kept -- never raw SQL or full
    result rows -- so prompts stay bounded over a long conversation."""

    question: str
    metric_phrase: Optional[str] = None
    state: Optional[str] = None
    county: Optional[str] = None
    year: Optional[str] = None
    answer: Optional[str] = None


def format_history(turns: list[Turn], max_turns: int = 5) -> str:
    if not turns:
        return "(no prior conversation)"
    lines = []
    for t in turns[-max_turns:]:
        resolved = ", ".join(
            f"{label}={value}"
            for label, value in [
                ("metric", t.metric_phrase),
                ("state", t.state),
                ("county", t.county),
                ("year", t.year),
            ]
            if value
        )
        lines.append(f'User asked: "{t.question}" (resolved: {resolved or "none"}) -> {t.answer or "(no answer)"}')
    return "\n".join(lines)
