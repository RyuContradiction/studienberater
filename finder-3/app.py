import streamlit as st
from finder import ANTWORTEN, MAX_FRAGEN, Finder, erklaere, ohne_codes

st.set_page_config(page_title="HSD Studiengang-Akinator", page_icon="🎓", layout="centered")
st.title("🎓 HSD Studiengang-Akinator")


@st.cache_resource
def lade_finder():
    return Finder.aus_datenbank()


@st.cache_data
def erklaerung(modul):
    return erklaere(modul)


finder = lade_finder()
s = st.session_state
if s.get("version") != 4:
    s.clear()
    s.version = 4
    s.antworten, s.fertig, s.frage_key = {}, False, None
profil = finder.profile(s.antworten)
rang = finder.ranking(profil)
fertig = s.fertig or finder.ist_fertig(profil, s.antworten)
if fertig:
    st.sidebar.subheader("Übereinstimmung")
    for name, score in rang:
        st.sidebar.write(f"**{name}** · {score:.0f} %")
        st.sidebar.progress(score / 100)
elif len(s.antworten) >= 3 and st.sidebar.button("Ergebnis jetzt anzeigen"):
    s.fertig = True
    st.rerun()

if not fertig:
    if s.frage_key is None:
        s.frage_key, s.frage, s.frage_typ, s.ollama_status = finder.next_question(profil, s.antworten)
    if s.frage_key is None:
        s.fertig = True
        st.rerun()
    st.caption(f"Frage {len(s.antworten) + 1} von höchstens {MAX_FRAGEN}")
    st.subheader(ohne_codes(s.frage))
    if s.ollama_status:
        st.warning(s.ollama_status)
    if s.frage_typ == "text":
        with st.form(f"freitext_{len(s.antworten)}", clear_on_submit=True):
            text = st.text_area("Erzähl kurz, was dich interessiert")
            if st.form_submit_button("Weiter") and text.strip():
                s.antworten[s.frage_key] = text
                s.frage_key = None
                st.rerun()
    else:
        for col, (label, value) in zip(st.columns(5), ANTWORTEN.items()):
            if col.button(label, use_container_width=True):
                s.antworten[s.frage_key] = value
                s.frage_key = None
                st.rerun()
    st.stop()

st.subheader(f"📊 Dein Ergebnis nach {len(s.antworten)} Fragen")
if len(rang) > 1 and rang[0][1] - rang[1][1] >= 30:
    st.success(f"Am besten passt **{rang[0][0]}** zu dir.")
for name, score in rang:
    st.markdown(f"**{name}** – {score:.0f} % Übereinstimmung")
    st.progress(score / 100)
    reasons = finder.gruende(name, s.antworten)
    if reasons:
        st.caption("Passt zu dir wegen: " + ", ".join(reasons))
if rang and any(profil):
    modul = finder.bestes_modul(rang[0][0], profil)
    if modul:
        st.caption("Beispiel aus dem Modulhandbuch: " + ohne_codes(modul["titel"]))
        text = erklaerung(modul)
        if text:
            st.write(text)
if st.button("🔄 Von vorne beginnen"):
    s.clear()
    st.rerun()
