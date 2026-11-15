import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from scoring import cosine_to_percent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
 
# Seiten-Konfiguration
st.set_page_config(page_title="HSD Studiengangs-Finder (Studienberater-Akinator)", page_icon="🎓", layout="centered")
 
st.title("🎓 HSD Studiengangs-Finder: Der intelligente Studienberater")
st.write("Erzähl mir von deinen Interessen. Ich helfe dir, den passenden HSD-Studiengang zu finden, und gleiche deine Erwartungen mit der Realität ab!")
 
# 1. Embedding-Modell und ChromaDB einmalig laden
@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=".chroma_db")
    collection = client.get_or_create_collection(name="hsd_study_finder", metadata={"hnsw:space": "cosine"})
    return model, collection
 
model, collection = load_resources()
 
# 2. Session State initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hallo! Welcher Bereich oder Traumberuf geht dir aktuell durch den Kopf? (Egal ob vage oder konkret – erzähl einfach mal, was dich fasziniert!)"
        }
    ]
 
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
 
if "question_count" not in st.session_state:
    st.session_state.question_count = 1  # Zählt die Fragen (max. 12)
 
if "show_results" not in st.session_state:
    st.session_state.show_results = False
 
# 3. Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# 4. Frage-Modus
if not st.session_state.show_results:
    if prompt := st.chat_input("Deine Antwort..."):
        # Nutzer-Nachricht speichern und anzeigen
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.user_answers.append(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
 
        # SCHRITT A: Vektor-Ranking und breite Themenbereiche aus ChromaDB abfragen
        full_user_query = " ".join(st.session_state.user_answers)
        query_vector = model.encode([full_user_query]).tolist()
        ergebnisse = collection.query(query_embeddings=query_vector, n_results=150, include=["metadatas", "documents", "distances"])
        studiengang_bestwerte = {}
        top_themes = []
 
        for metadata, doc, distance in zip(ergebnisse["metadatas"][0], ergebnisse["documents"][0], ergebnisse["distances"][0]):
            prog_id = metadata["program_id"]
            similarity = 1.0 - distance 
            if prog_id not in studiengang_bestwerte or similarity > studiengang_bestwerte[prog_id]:
                studiengang_bestwerte[prog_id] = similarity
 
            # Sammle verallgemeinerte Modul-Inhalte
            snippet = doc.replace("\n", " ").strip()[:100]
            if snippet not in top_themes and len(top_themes) < 6:
                top_themes.append(snippet)
 
        rankings = []
        for prog_id, raw_sim in studiengang_bestwerte.items():
            prozent = cosine_to_percent(raw_sim)
            rankings.append((prog_id, prozent))
        rankings.sort(key=lambda x: x[1], reverse=True)
 
        # ABBRUCHBEDINGUNG (Max 12 Fragen oder klarer Favorit ab Frage 5)
        top_score = rankings[0][1] if rankings else 0
        second_score = rankings[1][1] if len(rankings) > 1 else 0
        abstand = top_score - second_score
        max_fragen_erreicht = st.session_state.question_count >= 12
        sicherer_treffer = (top_score >= 68.0 and abstand >= 12.0 and st.session_state.question_count >= 5)
 
        if sicherer_treffer or max_fragen_erreicht:
            st.session_state.show_results = True
            if sicherer_treffer:
                conclusion_msg = f"🎯 Ich habe ein klares Bild! **{rankings[0][0]}** passt mit **{top_score:.1f}%** am besten zu deinem Profil. Hier ist dein Auswertungs-Ranking:"
            else:
                conclusion_msg = f"🏁 Nach {st.session_state.question_count} Fragen ist unser Beratungsgespräch beendet! Hier ist dein Studiengangs-Ranking:"
            st.session_state.messages.append({"role": "assistant", "content": conclusion_msg})
            st.rerun()
 
        # SCHRITT B: Nächste Frage via Ollama – Als realistischer Studienberater!
        with st.spinner("Der Studienberater analysiert deine Angaben und Vorerfahrungen..."):
            try:
                llm = ChatOllama(model="llama3", temperature=0.6)
                themes_text = "\n".join([f"- {t}" for t in top_themes])
 
                system_prompt = (
                    "Du bist ein erfahrener, nahbarer Studienberater und Akinator an der HSD (Hochschule Düsseldorf).\n"
                    "DEINE ROLLE & STRATEGIE:\n"
                    "1. **REALITY-CHECK & WISSENSSTAND PRÜFEN:** Wenn der Nutzer ein vages oder populäres Ziel nennt (z.B. 'Hacker werden', 'Spiele entwickeln', 'KI bauen'), hinterfrage das charmant. Prüfe seinen Wissensstand: Hat er schon mal programmiert? Gab es Berührungspunkte mit Mathe, Logik, Elektrotechnik oder Wirtschaft? Finde heraus, ob er realistische Erwartungen hat oder ob er noch Grundlagenberatung braucht.\n"
                    "2. **VERALLGEMEINERTE THEMEN (MAKR O- TO MIKRO):** Nutze verständliche, breite Kategorien (z.B. Software & IT, Wirtschaft, Gestaltung/Medien, Technik) statt trockener Modul-Titel.\n"
                    "3. **KEIN TUNNELBLICK & RÜCKSPRUNG:** Verrenne dich nicht in ein Detail. Wenn der Nutzer sich in eine Nische verrennt, hake nach, ob er nicht vielleicht doch breit gefächerte Interessen (z.B. eine Mischung aus Design und Technik oder Wirtschaft und IT) hat.\n"
                    "4. **ORIENTIERUNG AN DEN HSD-INHALTEN:** Nutze diese echten HSD-Themenbereiche im Hintergrund als Kompass:\n"
                    f"{themes_text}\n\n"
                    "STRIKTE REGELN:\n"
                    "- Nenne KEINE Studiengangsnamen (wie DAISY, BMI, BMT).\n"
                    "- Starte NIEMALS mit 'Wie wichtig ist es dir...'.\n"
                    "- Antworte IMMER in der Sprache des Nutzers (Deutsch oder Englisch).\n"
                    "- Gib AUSSCHLIESSLICH deine Rückfrage oder deinen Beratungsimpuls aus!"
                )
 
                lc_messages = [SystemMessage(content=system_prompt)]
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant" and not m["content"].startswith("🎯") and not m["content"].startswith("🏁"):
                        lc_messages.append(AIMessage(content=m["content"]))
                response = llm.invoke(lc_messages)
                next_question = response.content.strip()
            except Exception as e:
                next_question = f"⚠️ Fehler bei der Verbindung zu Ollama ({e}). Ist dein Server gestartet?"
 
        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "assistant", "content": next_question})
        st.rerun()
 
# 5. Finale Auswertung
if st.session_state.show_results:
    if not any("Studiengang " in m["content"] and "Übereinstimmung" in m["content"] for m in st.session_state.messages[-2:]):
        full_user_query = " ".join(st.session_state.user_answers)
        query_vector = model.encode([full_user_query]).tolist()
        ergebnisse = collection.query(query_embeddings=query_vector, n_results=150, include=["metadatas", "distances"])
        studiengang_bestwerte = {}
        for metadata, distance in zip(ergebnisse["metadatas"][0], ergebnisse["distances"][0]):
            prog_id = metadata["program_id"]
            similarity = 1.0 - distance 
            if prog_id not in studiengang_bestwerte or similarity > studiengang_bestwerte[prog_id]:
                studiengang_bestwerte[prog_id] = similarity
 
        rankings = []
        for prog_id, raw_sim in studiengang_bestwerte.items():
            prozent = cosine_to_percent(raw_sim)
            rankings.append((prog_id, prozent))
        rankings.sort(key=lambda x: x[1], reverse=True)
 
        result_md = "### 📊 Dein detailliertes Studiengangs-Ranking:\n\n"
        for prog_id, prozent in rankings:
            result_md += f"- **Studiengang {prog_id}**: **{prozent:.1f}%** Übereinstimmung\n"
        st.session_state.messages.append({"role": "assistant", "content": result_md})
        st.rerun()
 
    st.write("")
    if st.button("🔄 Noch einmal von vorne beginnen"):
        st.session_state.clear()
        st.rerun()
