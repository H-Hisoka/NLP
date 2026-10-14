import re
import unicodedata
import pandas as pd
from config import DATA_FILE, TRAIN_RATIO, SEED


def normalize_text(text: str) -> str:
    text=str(text)
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return normalize_text(text).split()

# Dữ liệu 
def load_data(file_path=DATA_FILE):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df.dropna(subset=["wrong", "right"]).copy()
    df["wrong_norm"] = df["wrong"].apply(normalize_text)
    df["right_norm"] = df["right"].apply(normalize_text)
    return df[(df["wrong_norm"] != "") & (df["right_norm"] != "")]


def split_train_test(df, ratio=TRAIN_RATIO, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_index = int(len(df) * ratio)
    return df.iloc[:split_index], df.iloc[split_index:]
