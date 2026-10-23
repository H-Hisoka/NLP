import re
import unicodedata
import pandas as pd
from config import DATA_FILE, TRAIN_RATIO, SEED, SPECIAL_PATTERNS


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
    df = df.drop_duplicates(subset=["right_norm"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_index = int(len(df) * ratio)
    return df.iloc[:split_index], df.iloc[split_index:]
# Lỗi gõ Telex / VNI
TELEX_TONES = {"s": "\u0301", "f": "\u0300", "r": "\u0309", "x": "\u0303", "j": "\u0323"}
VNI_TONES = {"1": "\u0301", "2": "\u0300", "3": "\u0309", "4": "\u0303", "5": "\u0323"}
VIETNAMESE_VOWELS = "aăâeêioôơuưy"


def apply_tone(base, tone_mark, vocab):
    """Đặt dấu thanh lên từng nguyên âm của `base`, chỉ giữ các từ có trong từ vựng."""
    if not tone_mark:
        return [base] if base in vocab else []
    candidates = []
    for i, ch in enumerate(base):
        if ch in VIETNAMESE_VOWELS:
            word = unicodedata.normalize("NFC", base[:i + 1] + tone_mark + base[i + 1:])
            if word in vocab:
                candidates.append(word)
    return candidates


def telex_candidates(token, vocab):
    """Chuyển chuỗi gõ Telex (ví dụ 'vieejt', 'dduwowcj') về các từ tiếng Việt có trong vocab."""
    token = token.lower()
    if not (token.isascii() and token.isalpha()):
        return []
    results = []
    for pos in [None] + [i for i, ch in enumerate(token) if ch in TELEX_TONES]:
        if pos is None:
            raw, mark = token, ""
        else:
            # phím dấu phải đứng sau nguyên âm và phía sau chỉ còn phụ âm cuối
            if not any(c in "aeiouyw" for c in token[:pos]) or any(c in "aeiouyw" for c in token[pos + 1:]):
                continue
            raw, mark = token[:pos] + token[pos + 1:], TELEX_TONES[token[pos]]
        base = raw.replace("dd", "đ").replace("uwow", "ươ").replace("uow", "ươ")
        for old, new in (("aa", "â"), ("ee", "ê"), ("oo", "ô"), ("aw", "ă"), ("ow", "ơ"), ("uw", "ư")):
            base = base.replace(old, new)
        results += [w for w in apply_tone(base, mark, vocab) if w != token]
    return results


def vni_candidates(token, vocab):
    """Chuyển chuỗi gõ VNI (ví dụ 'vie65t', 'd9uo7c5') về các từ tiếng Việt có trong vocab."""
    token = token.lower()
    if not (token.isascii() and token.isalnum() and any(c.isdigit() for c in token)
            and any(c.isalpha() for c in token)):
        return []
    letters, tone = [], ""
    for ch in token:
        if ch.isalpha():
            letters.append(ch)
        elif ch in VNI_TONES:
            tone = VNI_TONES[ch]
        elif ch in "6789":
            targets = {"6": "aeo", "7": "ou", "8": "a", "9": "d"}[ch]
            index = max((i for i, c in enumerate(letters) if c in targets), default=None)
            if index is None:
                return []
            if ch == "7" and letters[index] == "o" and index > 0 and letters[index - 1] == "u":
                letters[index - 1], letters[index] = "ư", "ơ"
            else:
                letters[index] = {"6": {"a": "â", "e": "ê", "o": "ô"}, "7": {"o": "ơ", "u": "ư"},
                                  "8": {"a": "ă"}, "9": {"d": "đ"}}[ch][letters[index]]
        else:
            return []
    return apply_tone("".join(letters), tone, vocab)
# Nhận diện email, link, số điện thoại (không phải chữ cần sửa chính tả) 
SPECIAL_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in SPECIAL_PATTERNS.items()))
def find_special_spans(text):
    """Danh sách (bắt đầu, kết thúc, loại) của các đoạn email, url, số điện thoại, ngày, giờ, hashtag."""
    return [(m.start(), m.end(), m.lastgroup) for m in SPECIAL_REGEX.finditer(text)]

def mask_special_spans(text):
    """Thay các đoạn đặc biệt bằng ký tự '\x00' (độ dài giữ nguyên) để không bị tách thành từ."""
    chars = list(text)
    for start, end, _ in find_special_spans(text):
        chars[start:end] = "\x00" * (end - start)
    return "".join(chars)