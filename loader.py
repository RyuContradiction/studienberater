import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk:
    program_id: str
    module: str
    page: int
    text: str

# Regex, um typische Modul-Überschriften zu erkennen (z.B. "D1.1: Einführung..." oder "PF 5.3...")
_MODULE_HEADING = re.compile(r"^(?P<id>(?:[A-Z]\d+(?:\.\d+)?|PF\s?\d+(?:\.\d+)?))\s*[:–-]\s*(?P<title>.{2,100})$")
_DOT_LEADER = re.compile(r"\.{4,}")  # Erkennt Inhaltsverzeichnisse mit Punktlinien (. . . .)


def _normalize_edge_line(line: str) -> str:
    line = re.sub(r"Seite\s+\d+(?:\s+von\s+\d+)?", "Seite", line)
    if line.isdigit():
        return ""
    return re.sub(r"\s{2,}\d+$", "", line).strip()


def detect_module(lines: list[str]) -> str | None:
    """Sucht im Seitenkopf nach einer Modul-Überschrift."""
    for line in lines[:8]:  # Schaut in den ersten 8 Zeilen nach
        if _DOT_LEADER.search(line):
            continue
        match = _MODULE_HEADING.match(line)
        if match:
            data = match.groupdict()
            return f"{data['id']}: {data['title'].strip()}"
    return None


def _flatten(text: str) -> str:
    """Behebt Silbentrennungen am Zeilenende und formt fließenden Text."""
    text = re.sub(r"(\w)-\n(?=[a-zäöüß])", r"\1", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def load_pdf(path: Path, program_id: str) -> list[Chunk]:
    pages = PyPDFLoader(str(path)).load()
    
    # 1. Unerwünschte Seiten direkt aussortieren
    pages = [
        page for page in pages
        if "praxissemester" not in page.page_content.casefold()
        and "prüfungsausschuss" not in page.page_content.casefold()
    ]

    # 2. Wiederkehrende Kopf- und Fußzeilen ermitteln (deine bewährte Logik)
    edge_lines = Counter()
    page_lines_list = []
    
    for page in pages:
        lines = [line.strip() for line in page.page_content.splitlines() if line.strip()]
        page_lines_list.append(lines)
        edge_lines.update({_normalize_edge_line(line) for line in lines[:4] + lines[-4:]})

    min_repetitions = max(2, (len(pages) + 1) // 2)
    repeated_edge_lines = {line for line, count in edge_lines.items() if count >= min_repetitions}

    # 3. Kopf-/Fußzeilen entfernen und Text bereinigen
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = []
    current_module = path.stem  # Fallback-Modulname ist der Dateiname

    for doc, lines in zip(pages, page_lines_list):
        # Inhaltsverzeichnis-Seiten überspringen
        if lines and sum(bool(_DOT_LEADER.search(l)) for l in lines) / len(lines) > 0.3:
            continue

        # Prüfen, ob auf dieser Seite ein neues Modul beginnt
        found_mod = detect_module(lines)
        if found_mod:
            current_module = found_mod

        # Boilerplate oben/unten löschen
        cleaned_lines = [
            line for index, line in enumerate(lines)
            if not ((index < 4 or index >= len(lines) - 4) and _normalize_edge_line(line) in repeated_edge_lines)
        ]
        
        page_text = _flatten("\n".join(cleaned_lines))
        
        if not page_text:
            continue

        # In Chunks zerteilen und Modul-Kontext mitnehmen
        for text in splitter.split_text(page_text):
            chunks.append(
                Chunk(
                    program_id=program_id,
                    module=current_module,
                    page=doc.metadata.get("page", 0) + 1,
                    text=f"{current_module} – {text}",
                )
            )

    return chunks


def load_all_chunks() -> list[Chunk]:
    data_dir = Path(__file__).parent / "data"
    chunks = []

    for path in sorted(data_dir.rglob("*.pdf")):
        chunks.extend(load_pdf(path, path.stem))

    return chunks


if __name__ == "__main__":
    print("Starte PDF-Extraktion mit erweitertem Cleaning...")
    alle_chunks = load_all_chunks()
    print(f"Fertig! Insgesamt {len(alle_chunks)} Chunks geladen.")
    
    if alle_chunks:
        print("\nBeispiel für den ersten Chunk:")
        print(f"Studien-ID: {alle_chunks[0].program_id}")
        print(f"Erkanntes Modul: {alle_chunks[0].module}")
        print(f"Seite: {alle_chunks[0].page}")
        print(f"Text-Ausschnitt: {alle_chunks[0].text[:150]}...")