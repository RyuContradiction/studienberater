# HSD Studiengang-Akinator

Die App gleicht Interessen mit den Studiengängen aus vier HSD-Modulhandbüchern ab. Sie startet mit einer offenen Interessenfrage, fragt danach breit und vertieft positiv bewertete Themen. Fragen sind alltagsnah und fragen nach Interesse oder Tätigkeiten, nicht nach der Wichtigkeit eines Fachbegriffs.

## Starten

```bash
pip install -r requirements.txt
python build_db.py
ollama pull llama3
ollama serve
streamlit run app.py
```

`build_db.py` bereinigt wiederkehrende Kopf- und Fußzeilen, erkennt einzelne Module und schließt Praxis- und Abschlusssemester aus. Wahlpflicht- und Wahlmodule bleiben enthalten. Titel und Lehrinhalte werden mit `sentence-transformers` eingebettet und zusammen mit Metadaten in ChromaDB gespeichert. Nach Änderungen an PDFs oder Loader die Datenbank neu aufbauen.

## Ablauf

1. Die erste Frage ist offen. Nach vier breiten Fragen und anschließend etwa alle drei Antworten kommen zusätzliche, unterschiedliche Freitextfragen zu Erfahrungen, Abneigungen und Berufswünschen. Jede Textfrage zeigt nur ein Textfeld.
2. Breite Themen beginnen mit Mathematik und digitaler Sicherheit. Die Fragen dazu kommen aus dem Code und werden höchstens einmal gestellt.
3. Nur positiv bewertete Themen werden vertieft. Detailfragen kommen aus alltagsnahen Interessen und Tätigkeiten; jedes Detail wird höchstens einmal gefragt.
4. Ja/Nein-Fragen zeigen nur die fünf Akinator-Buttons. Ollama entscheidet keine Fragen. Es kann am Ende mit einem `SystemMessage` einen einfachen Satz zu einem passenden Handbuchmodul erklären.
5. Rocchio-Profil und zentrierte Moduldaten ergeben unabhängige Übereinstimmungswerte pro Studiengang; es gibt kein Softmax. Bei klarem Favoriten endet die Runde automatisch, spätestens nach 12 Fragen oder auf Wunsch früher.

Ist Ollama nicht erreichbar, bleibt das Ranking funktionsfähig; nur die optionale Modulerklärung am Ende fehlt.

## Dateien

- `loader.py`: Kopf- und Fußzeilenbereinigung und Modulerkennung.
- `build_db.py`: Embeddings erstellen und ChromaDB aufbauen.
- `themen.py`: breite Themenfelder samt Beschreibungen.
- `finder.py`: Themenbreite, Trichterlogik, Profile und Ranking.
- `app.py`: Streamlit-Oberfläche.
- `test_finder.py`: kleine Regressionstests.
