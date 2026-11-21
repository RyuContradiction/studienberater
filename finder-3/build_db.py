import chromadb
from sentence_transformers import SentenceTransformer
from finder import DB_PFAD, MODELL, SAMMLUNG
from loader import lade_alle_module
module = lade_alle_module()
print(f"{len(module)} Module geladen.")
texte = [f"{m.titel}. {m.inhalt}" for m in module]
vektoren = SentenceTransformer(MODELL).encode(texte, normalize_embeddings=True, show_progress_bar=True)
client = chromadb.PersistentClient(path=DB_PFAD)
try:
    client.delete_collection(SAMMLUNG)
except Exception:
    pass  # gab es noch nicht
sammlung = client.create_collection(SAMMLUNG, metadata={"hnsw:space": "cosine"})
sammlung.add(
    ids=[f"modul-{i}" for i in range(len(module))],
    embeddings=vektoren.tolist(),
    documents=texte,
    metadatas=[{"studiengang": m.studiengang, "titel": m.titel, "inhalt": m.inhalt} for m in module],
)
print(f"Fertig: {len(module)} Module in '{DB_PFAD}' gespeichert.")
