from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Suche nach einer PDF-Datei im data-Ordner deines Projekts
# (Pass den Pfad an, je nachdem wo deine PDFs liegen)
pdf_pfade = list(Path("data").glob("*.pdf"))

if pdf_pfade:
    erste_pdf = pdf_pfade[0]
    print(_ := f"Lese Datei: {erste_pdf.name} ...\n")
    
    # Lade die PDF mit dem LangChain Loader
    loader = PyPDFLoader(str(erste_pdf))
    pages = loader.load()
    
    print(f"Gesamtanzahl Seiten in der PDF: {len(pages)}")
    print("\n--- Inhalt der 1. Seite (erste 300 Zeichen) ---")
    print(pages[0].page_content[:300])
else:
    print("Keine PDF-Datei im 'data'-Ordner gefunden. Bitte lege eine ab!")