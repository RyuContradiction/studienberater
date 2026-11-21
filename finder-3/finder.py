import re
import chromadb
import numpy as np
from themen import DETAILS, FRAGE_TEXTE, THEMEN, OFFENE_FRAGEN

MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PFAD, SAMMLUNG = ".chroma_db", "hsd_module"
ANTWORTEN = {"Ja": 1.0, "Eher ja": 0.5, "Weiß nicht": 0.0, "Eher nein": -0.5, "Nein": -1.0}
MIN_FRAGEN, MAX_FRAGEN, BETA, GAMMA = 8, 12, 1.0, 0.5
TRENNER = re.compile(r"[,.;!?\n]|\baber\b|\bund\b|\bsondern\b|\boder\b", re.I)
VERNEINUNG = re.compile(r"\b(?:kein\w*|nicht|nie|nichts|hasse\w*|ungern|langweilig\w*)\b", re.I)


def normiere(v):
    return v / (np.linalg.norm(v) or 1.0)


def ohne_codes(text):
    text = re.sub(r"\s*\((?:[A-Z]{1,4}\s*)?\d+(?:\.\d+)*\)", "", text)
    text = re.sub(r"\b(?:[A-Z]{1,4}\s*)?\d+(?:\.\d+)+\s*[:–-]\s*", "", text)
    text = re.sub(r"\b(?:[A-Z]{1,4}\s*\d{1,3}|Modul\s+\d+(?:\.\d+)*)\s*[:–-]\s*", "", text)
    return re.sub(r"^(?:Vertiefung|Fokusmodul)\s+", "", text, flags=re.I).strip()


class Finder:
    def __init__(self, vectors, modules, embed):
        raw = np.asarray(vectors, dtype=float)
        self.middle = raw.mean(axis=0)
        self.V = np.array([normiere(v - self.middle) for v in raw])
        self.modules, self.embed = modules, embed
        self.programs = sorted({m["studiengang"] for m in modules})
        self.centroids = {p: normiere(self.V[self.indices(p)].mean(axis=0)) for p in self.programs}
        self.T = {t: self.embed_text(f"{t}: {d}") for t, d in THEMEN.items()}
        self.D = {f"{t}|{i}": self.embed_text(f"{t}: {d}") for t, values in DETAILS.items() for i, d in enumerate(values)}
        self.topic_by_module = [max(self.T, key=lambda t: self.V[i] @ self.T[t]) for i in range(len(modules))]
        self.broad_topics = sorted(THEMEN, key=lambda t: (t in {"Mathematik", "Cybersecurity"}, self.breadth(t)), reverse=True)

    @classmethod
    def aus_datenbank(cls):
        from sentence_transformers import SentenceTransformer
        collection = chromadb.PersistentClient(path=DB_PFAD).get_collection(SAMMLUNG)
        data = collection.get(include=["embeddings", "metadatas"])
        return cls(data["embeddings"], data["metadatas"], SentenceTransformer(MODELL).encode)

    def indices(self, program):
        return [i for i, m in enumerate(self.modules) if m["studiengang"] == program]

    def embed_text(self, text):
        return normiere(np.asarray(self.embed(text), dtype=float) - self.middle)

    def breadth(self, topic):
        return sum(np.max(self.V[self.indices(p)] @ self.T[topic]) >= 0.42 for p in self.programs), sum(t == topic for t in self.topic_by_module)

    def profile(self, answers):
        signals = []
        for key, answer in answers.items():
            if isinstance(answer, str):
                for part in TRENNER.split(answer):
                    if len(part.strip()) < 3 or re.search(r"\b(?:weiß nicht|keine ahnung|unsicher)\b", part, re.I):
                        continue
                    negative = bool(VERNEINUNG.search(part))
                    asked_to_avoid = key == "offen|abneigung"
                    sign = -1 if negative or asked_to_avoid else 1
                    signals.append((self.embed_text(VERNEINUNG.sub("", part).strip()), sign))
            else:
                vector = self.T[key[6:]] if key.startswith("thema|") else self.D[key[7:]] if key.startswith("detail|") else self.V[int(key)]
                signals.append((vector, answer))
        liked = [v * w for v, w in signals if w > 0]
        disliked = [-w * v for v, w in signals if w < 0]
        dim = self.V.shape[1]
        pos = np.mean(liked, axis=0) if liked else np.zeros(dim)
        neg = np.mean(disliked, axis=0) if disliked else np.zeros(dim)
        return normiere(BETA * pos - GAMMA * neg)

    def ranking(self, profile):
        if not np.linalg.norm(profile):
            return [(p, 0.0) for p in self.programs]
        return sorted(((p, float(np.clip(50 + 50 * profile @ self.centroids[p], 0, 100))) for p in self.programs), key=lambda x: x[1], reverse=True)

    def next_question(self, profile, answers):
        if "interessen" not in answers:
            return "interessen", "Wofür interessierst du dich in deiner Freizeit oder im Bereich Technik, Medien und Gestaltung?", "text", ""
        topics = {k[6:]: v for k, v in answers.items() if isinstance(k, str) and k.startswith("thema|")}
        asked_open = {k[6:] for k in answers if isinstance(k, str) and k.startswith("offen|")}
        open_topics = [t for t in self.broad_topics if t not in topics]
        closed = sum(not isinstance(k, str) or k.startswith(("thema|", "detail|")) for k in answers)
        if len(topics) >= 4 and closed // 3 > len(asked_open):
            name = next((n for n in OFFENE_FRAGEN if n not in asked_open), None)
            if name:
                return f"offen|{name}", OFFENE_FRAGEN[name], "text", ""
        if open_topics and (len(topics) < 4 or (closed % 3 == 0 and open_topics)):
            topic = open_topics[0]
            return f"thema|{topic}", f"Interessierst du dich dafür, {FRAGE_TEXTE[topic]}?", "buttons", ""
        for topic, value in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            if value <= 0:
                continue
            for i, detail in enumerate(DETAILS[topic]):
                key = f"detail|{topic}|{i}"
                if key not in answers:
                    return key, f"Interessierst du dich dafür, {detail}?", "buttons", ""
        if open_topics:
            topic = open_topics[0]
            return f"thema|{topic}", f"Interessierst du dich dafür, {FRAGE_TEXTE[topic]}?", "buttons", ""
        return None, "", "", ""

    def ist_fertig(self, profile, answers):
        rank = self.ranking(profile)
        clear = rank[0][1] >= 72 and rank[0][1] - rank[1][1] >= 15
        return len(answers) >= MAX_FRAGEN or (len(answers) >= MIN_FRAGEN and clear)

    def gruende(self, program, answers):
        topics = [k[6:] for k, v in answers.items() if isinstance(k, str) and k.startswith("thema|") and v > 0]
        topics += [DETAILS[k.split("|")[1]][int(k.rsplit("|", 1)[1])] for k, v in answers.items() if isinstance(k, str) and k.startswith("detail|") and v > 0]
        modules = [ohne_codes(self.modules[i]["titel"]) for i, v in answers.items() if isinstance(i, int) and v > 0 and self.modules[i]["studiengang"] == program]
        return topics + modules

    def bestes_modul(self, program, profile):
        ids = self.indices(program)
        return self.modules[max(ids, key=lambda i: self.V[i] @ profile)] if ids else None


def erklaere(modul):
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        system = "Erkläre den Handbuchinhalt in einem einfachen deutschen Satz mit höchstens 18 Wörtern. Keine Modulcodes und kein Fachjargon."
        user = f"{ohne_codes(modul['titel'])}: {ohne_codes(modul['inhalt'][:300])}"
        return ohne_codes(ChatOllama(model="llama3", temperature=0.2).invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip())
    except Exception:
        return ""
