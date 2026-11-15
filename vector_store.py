from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from loader import load_all_chunks

def initialize_database():
    print("1. Lade alle echten PDF-Chunks über den Loader...")
    chunks = load_all_chunks()
    print(f"-> {len(chunks)} Chunks geladen.")

    print("\n2. Lade das Embedding-Modell...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("\n3. Verbinde mit ChromaDB...")
    # Wir speichern die echte Datenbank im Ordner '.chroma_db'
    client = chromadb.PersistentClient(path=".chroma_db")
    
    # Falls die Collection schon da ist, löschen wir sie sicher (jetzt mit Exception-Abfang)
    try:
        client.delete_collection("hsd_study_finder")
    except Exception:
        pass
        
    collection = client.get_or_create_collection(
    name="hsd_study_finder", 
    metadata={"hnsw:space": "cosine"} # <--- Das ist der entscheidende Fix!
)

    print("\n4. Bereite Daten für die Datenbank vor...")
    ids = [f"chunk-{index}" for index in range(len(chunks))] # Nutzt saubere Index-IDs
    texts = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "program_id": chunk.program_id,
            "module": chunk.module,
            "page": chunk.page
        } 
        for chunk in chunks
    ]

    print("5. Erzeuge Vektoren (Das kann je nach Laptop einen Moment dauern)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("\n6. Speichere alles in ChromaDB...")
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Fertig! Alle {len(chunks)} Chunks sind nun fest in ChromaDB gespeichert.")

if __name__ == "__main__":
    initialize_database()