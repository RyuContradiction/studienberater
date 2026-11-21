import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader
DATEN_ORDNER = Path(__file__).parent / "data"
NAMEN = {
    "BMI": "Medieninformatik (B.Sc.)",
    "BMT(Medientechnik)": "Medientechnik (B.Eng.)",
    "BTB(Bild_und_Ton)": "Ton und Bild (B.Eng.)",
    "DAISY": "Data Science, AI und Intelligente Systeme (B.Sc.)",
}
MODUL_KOPF = re.compile(
    r"\b(?:[A-Z]{1,4}\s?\d{1,2}(?:\.\d{1,2})?\s*[:–-]|Modul\s+\d{1,2}(?:\.\d{1,2})?\s*:|PF\s?\d\.\d{1,2})"
    r"\s+(?P<titel>.{3,100})$"
)

def _abschnitt(ueberschrift: str) -> re.Pattern:
    return re.compile(
        rf"^(?:\d\s+)?(?:{ueberschrift})\W*(?P<text>.{{30,}}?)(?=Literatur|Lehr\s?und\s?Lernformen|Lehrformen|Prüfungsform|\Z)",
        re.S | re.M,
    )
INHALT = _abschnitt("Lehrinhalte|Inhalte")
LERNZIELE = _abschnitt("Lernergebnisse|Lernziele")
UNWICHTIG = re.compile(
    r"\b(?:Bachelorarbeit|Abschlussarbeit|Abschlusssemester|Externes Semester|Praxis(?:semester|phase)?)\b",
    re.I,
)


@dataclass
class Modul:
    studiengang: str
    titel: str
    inhalt: str


def lade_pdf(pfad: Path) -> list[Modul]:
    seiten = [
        [z.strip() for z in (seite.extract_text() or "").splitlines() if z.strip()]
        for seite in PdfReader(pfad).pages
    ]
    randzeilen = Counter(z.lower() for s in seiten for z in s[:3] + s[-3:])
    schwelle = max(2, (len(seiten) + 1) // 2)
    kopf_fuss = {z for z, n in randzeilen.items() if n >= schwelle} | {str(n) for n in range(1, len(seiten) + 1)}
    seiten = [[z for z in s if z.lower() not in kopf_fuss] for s in seiten]
    texte = {}  # Modultitel -> gesamter Text des Moduls
    aktuell = None
    for zeilen in seiten:
        for zeile in zeilen[:10]:  # Überschrift steht oben auf der Seite
            if re.search(r"\.{4,}", zeile):  # Inhaltsverzeichnis ("Titel ........ 12")
                continue
            treffer = MODUL_KOPF.search(zeile)
            if treffer:
                aktuell = treffer["titel"].split(": ")[-1].strip()  # "Digital Health: X" -> "X"
                break
        if aktuell:
            texte[aktuell] = texte.get(aktuell, "") + "\n" + "\n".join(zeilen)
    module = []
    for titel, text in texte.items():
        if UNWICHTIG.search(titel):
            continue
        treffer = INHALT.search(text) or LERNZIELE.search(text)
        inhalt = re.sub(r"\s+", " ", treffer["text"]).strip() if treffer else ""
        inhalt = re.sub(r"(\w)- (?=[a-zäöüß])", r"\1", inhalt)  # Silbentrennung "Infor- matik" -> "Informatik"
        module.append(Modul(NAMEN.get(pfad.stem, pfad.stem), titel, inhalt[:500]))
    return module


def lade_alle_module() -> list[Modul]:
    module = []
    for pfad in sorted(DATEN_ORDNER.glob("*.pdf")):
        module.extend(lade_pdf(pfad))
    return module
if __name__ == "__main__":
    alle = lade_alle_module()
    for name in sorted({m.studiengang for m in alle}):
        print(f"{name}: {sum(m.studiengang == name for m in alle)} Module")
    print("\nBeispiel:", alle[0])
