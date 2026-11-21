import numpy as np
import finder as F
from loader import UNWICHTIG


def finder():
    module = [{"studiengang": "Test", "titel": "Modul", "inhalt": ""}]
    return F.Finder(np.ones((1, 12)), module, lambda _: np.ones(12))


def test_codes_are_removed():
    assert F.ohne_codes("D1.3: Code (D1.4) BMI 03: Test") == "Code Test"
    assert F.ohne_codes("Vertiefung Akustik") == "Akustik"


def test_electives_are_kept_but_practice_and_final_semesters_are_removed():
    assert UNWICHTIG.search("Praxissemester") and UNWICHTIG.search("Abschlusssemester")
    assert not UNWICHTIG.search("Wahlpflichtmodul 3") and not UNWICHTIG.search("Wahlmodul Gestaltung")


def test_first_question_is_text_only():
    key, question, _, kind, _ = finder().next_question(np.zeros(12), {})
    assert key == "interessen" and kind == "text" and "interessierst" in question.lower()


def test_broad_question_asks_about_interest():
    _, question, kind, _ = finder().next_question(np.zeros(12), {"interessen": "Kunst"})
    assert kind == "buttons" and question.startswith("Interessierst du dich dafür")


def test_open_questions_are_not_repeated():
    f = finder()
    answers = {"interessen": "Kunst", "thema|Mathematik": 1, "thema|Cybersecurity": 0,
               "thema|Programmierung": 0, "thema|Daten und KI": 0}
    key, _, kind, _ = f.next_question(np.zeros(12), answers)
    assert key == "offen|erfahrung" and kind == "text"
    answers[key] = "Ich mag kreative Projekte."
    assert f.next_question(np.zeros(12), answers)[0] != key
