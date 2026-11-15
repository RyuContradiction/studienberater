from __future__ import annotations
from dataclasses import dataclass

LOW_MATCH_LABEL = "Geringe Übereinstimmung"

@dataclass
class ProgramScore:
    program_id: str
    name: str
    cosine: float
    percent: float
    below_floor: bool

def bar_percent(score: ProgramScore) -> float:
    return max(5.0, score.percent)

def percent_label(score: ProgramScore) -> str:
    return f"{score.percent:.1f}%"

def low_match_note(score: ProgramScore) -> str:
    if score.below_floor:
        return "Dieser Studiengang weist nur eine geringe inhaltliche Überschneidung auf."
    return ""