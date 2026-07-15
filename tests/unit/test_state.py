from agent.state import Turn, format_history


def test_format_history_empty():
    assert format_history([]) == "(no prior conversation)"


def test_format_history_includes_resolved_slots_and_answer():
    turns = [Turn(question="population of Texas", metric_phrase="population", state="Texas", year="2020", answer="~28M")]
    text = format_history(turns)
    assert "population of Texas" in text
    assert "metric=population" in text
    assert "state=Texas" in text
    assert "~28M" in text


def test_format_history_truncates_to_max_turns():
    turns = [Turn(question=f"q{i}") for i in range(10)]
    text = format_history(turns, max_turns=3)
    assert "q9" in text
    assert "q7" in text
    assert "q6" not in text
    assert "q0" not in text
