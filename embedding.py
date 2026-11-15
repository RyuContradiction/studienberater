from sentence_transformers import SentenceTransformer
import chromadb

from loader import load_all_chunks

# 1. Das Embedding-Modell laden
print("Lade Embedding-Modell...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Einen lokalen ChromaDB Client erstellen (speichert die Daten im Ordner '.chroma_test')
client = chromadb.PersistentClient(path=".chroma_test")
collection = client.get_or_create_collection(name="hsd_test")

# 3. PDF-Dateien laden und ihre Chunks für die Embeddings vorbereiten
print("Lade und bereinige PDF-Chunks über den Loader...")
chunks = load_all_chunks()
if not chunks:
    raise RuntimeError("Keine Text-Chunks in den PDF-Dateien im data-Ordner gefunden.")

texte = [chunk.text for chunk in chunks]
print(inova := f"Erzeuge Vektoren für {len(texte)} Chunks (Das kann einen Moment dauern)...")
vektoren = model.encode(texte, show_progress_bar=True).tolist()

# 4. Chunks, Embeddings und Metadaten in ChromaDB speichern
print("Speichere in ChromaDB...")
collection.upsert(
    documents=texte,
    embeddings=vektoren,
    metadatas=[
        {
            "program_id": chunk.program_id,
            "module": chunk.module,
            "page": chunk.page,
        }
        for chunk in chunks
    ],
    ids=[f"chunk-{index}" for index in range(len(chunks))], # IDs über den Index erzeugen
)
print(f"Fertig! Alle {len(chunks)} PDF-Chunks erfolgreich in ChromaDB gespeichert!")
# 5. Eine Test-Abfrage machen (Wie eine echte Suche)
frage = "Ich interessiere mich für Webentwicklung und Frontend."
frage_vektor = model.encode([frage]).tolist()

ergebnisse = collection.query(
    query_embeddings=frage_vektor,
    n_results=1 # Gib mir den bestmöglichen Treffer
)

print("\n--- Suchergebnis ---")
print(f"Suche nach: '{frage}'")
print(f"Bester Treffer: {ergebnisse['documents'][0][0]}")
print(f"Metadaten dazu: {ergebnisse['metadatas'][0][0]}")