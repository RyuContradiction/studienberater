import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from scoring import cosine_to_percent

# Seiten-Konfiguration
st.set_page_config(page_title="HSD Studiengangs-Finder (Akinator)", page_icon="🎓", layout="centered")

st.title("🎓 HSD Studiengangs-Finder: Der Akinator-Modus")
st.write("Antworte im Chat auf meine Fragen. Ich finde heraus, welcher Studiengang am besten zu dir passt!")

# 1. Embedding-Modell und ChromaDB einmalig laden
@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=".chroma_db")
    collection = client.get_or_create_collection(name="hsd_study_finder", metadata={"hnsw:space": "cosine"})
    return model, collection

model, collection = load_resources()

# 2. Session State für den Chat-Verlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hallo! Erzähl mir erst mal: Wofür interessierst du dich in deiner Freizeit oder im Bereich Technik, Medien und Wirtschaft besonders?"
        }
    ]

if "user_answers" not in st.session_state:
    st.session_state.user_answers = []

if "step" not in st.session_state:
    st.session_state.step = 1  # Steuert, bei welcher Frage wir sind

# 3. Den bisherigen Chat-Verlauf auf dem Bildschirm anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat-Eingabe unten für den Nutzer (solange wir noch im Fragen-Modus sind)
if st.session_state.step <= 3:
    if prompt := st.chat_input("Deine Antwort..."):
        
        # Nutzer-Nachricht speichern und anzeigen
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.user_answers.append(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # KI-Logik: Nächste Frage basierend auf dem Schritt stellen
        if st.session_state.step == 1:
            next_question = "Spannend! Magst du eher logisch-mathematische Aufgaben (wie Programmieren, Algorithmen und Datenanalyse) oder bist du eher kreativ unterwegs (Design, Benutzeroberflächen, visuelle Gestaltung)?"
            st.session_state.step = 2
            
        elif st.session_state.step == 2:
            next_question = "Alles klar, danke dir! Gibt es noch einen speziellen Anwendungsbereich, der dich reizt (z. B. Gesundheit, Energie, Roboter/Industrie, Wirtschaft oder gesellschaftliche Themen)?"
            st.session_state.step = 3
            
        elif st.session_state.step == 3:
            next_question = "Super, ich habe alle Infos gesammelt! Ich werte jetzt die Modulhandbücher für dich aus..."
            st.session_state.step = 4  # Springt zum Auswertungs-Modus

        # KI-Antwort speichern und Seite neu laden, damit die Frage erscheint
        st.session_state.messages.append({"role": "assistant", "content": next_question})
        st.rerun()

# 5. Auswertung (Sobald alle Fragen beantwortet sind -> Step 4)
if st.session_state.step == 4:
    # Prüfen, ob das Ergebnis schon im Chat steht, sonst generieren
    if not any("Dein persönliches Studiengangs-Ranking" in m["content"] for m in st.session_state.messages):
        
        with st.spinner("Durchsuche Modulhandbücher und berechne Ähnlichkeiten..."):
            
            # Alle Antworten des Users zu einem fetten Suchtext zusammenfügen
            full_user_query = " ".join(st.session_state.user_answers)
            query_vector = model.encode([full_user_query]).tolist()
            
            # ChromaDB abfragen (wir holen viele Chunks, um alle Studiengänge abzudecken)
            ergebnisse = collection.query(
                query_embeddings=query_vector,
                n_results=150,
                include=["metadatas", "distances"]
            )
            
            # Besten Wert pro Studiengang ermitteln
            studiengang_bestwerte = {}
            for metadata, distance in zip(ergebnisse["metadatas"][0], ergebnisse["distances"][0]):
                prog_id = metadata["program_id"]
                similarity = 1.0 - distance 
                
                if prog_id not in studiengang_bestwerte or similarity > studiengang_bestwerte[prog_id]:
                    studiengang_bestwerte[prog_id] = similarity

            # Durch deine Scoring-Funktion jagen
            rankings = []
            for prog_id, raw_sim in studiengang_bestwerte.items():
                prozent = cosine_to_percent(raw_sim)
                rankings.append((prog_id, prozent))
                
            rankings.sort(key=lambda x: x[1], reverse=True)

            # Ergebnis-Text für den Chat zusammenbauen
            result_md = "### 🎯 Dein persönliches Studiengangs-Ranking:\n\n"
            for prog_id, prozent in rankings:
                result_md += f"- **Studiengang {prog_id}**: **{prozent:.1f}%** Übereinstimmung\n"
            
            # Im Chat als Assistenz-Nachricht hinzufügen
            st.session_state.messages.append({"role": "assistant", "content": result_md})
            st.rerun()

    # Button zum Neustarten des Chats
    st.write("")
    if st.button("🔄 Noch einmal von vorne beginnen"):
        st.session_state.clear()
        st.rerun()