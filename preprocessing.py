import re
import unicodedata

def normalize_text(text: str) -> str:
    text=str(text)
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return normalize_text(text).split()