from __future__ import annotations
import os

def get_chat_model():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            return "OpenAI GPT-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        except Exception:
            pass
    return "Lokaler Vektor-Modus (Akinator-Logik aktiv)", None

def provider_notes() -> list[str]:
    return [
        "Das System nutzt die Vektor-Raum-Verschiebung für dynamische Fragen aus den Modulhandbüchern.",
        "Optional kann ein LLM-Key für sprachliche Optimierungen hinterlegt werden."
    ]