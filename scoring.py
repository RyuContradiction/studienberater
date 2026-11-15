COS_FLOOR = 0.436
COS_CEIL = 0.758


def cosine_to_percent(similarity: float) -> float:
    """Skaliert eine Kosinus-Ähnlichkeit linear auf einen Prozentwert von 0 bis 100."""
    percent = (similarity - COS_FLOOR) / (COS_CEIL - COS_FLOOR) * 100
    return max(0.0, min(100.0, percent))
