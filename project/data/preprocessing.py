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
    """Thay các đoạn đặc biệt (kể cả đoạn SAI định dạng) bằng '\x00' (độ dài giữ nguyên) để không bị tách thành từ."""
    chars = list(text)
    spans = [(s, e) for s, e, _ in find_special_spans(text)]
    spans += [(item["start"], item["end"]) for item in find_invalid_format_spans(text)]
    for start, end in spans:
        chars[start:end] = "\x00" * (end - start)
    return "".join(chars)


# Phát hiện email / ngày / giờ SAI định dạng
import calendar
import difflib
from datetime import date as _date

KNOWN_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "yahoo.com.vn", "outlook.com",
                       "hotmail.com", "icloud.com", "live.com"]
EMAIL_TYPO_CUTOFF = 0.85    # độ giống tối thiểu để coi 'gmial.com' là gõ sai của 'gmail.com'
EMAIL_CANDIDATE = re.compile(r"[\w.+-]+@+[\w.-]*[\w-]")
DATE_CANDIDATE = re.compile(r"(?<![\w/])(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})(?![\w/])")
TIME_CANDIDATE = re.compile(r"(?<![\w:])(\d{1,2})([:hH])(\d{2})(?![\w:])")
EMAIL_OK = re.compile(r"^[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$")

def _strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _domain_typo(email):
    """Email đúng cấu trúc nhưng tên miền giống (mà không trùng) một nhà cung cấp quen thuộc."""
    domain = email.split("@")[-1].lower()
    return domain not in KNOWN_EMAIL_DOMAINS and bool(
        difflib.get_close_matches(domain, KNOWN_EMAIL_DOMAINS, n=1, cutoff=EMAIL_TYPO_CUTOFF))


def _email_ok(email):
    local = email.split("@")[0]
    return bool(EMAIL_OK.match(email)) and ".." not in email and not local.startswith(".") and not local.endswith(".")


def _fix_email(raw):
    local, _, domain = re.sub(r"@+", "@", raw).partition("@")
    local = re.sub(r"\.{2,}", ".", _strip_accents(local)).strip(".")
    domain = re.sub(r"\.{2,}", ".", domain.lower()).strip(".-")
    fixes = []

    def add(d):
        email = f"{local}@{d}"
        if local and _email_ok(email) and email != raw and email not in fixes:
            fixes.append(email)

    if "." in domain:
        add(domain)
        if domain not in KNOWN_EMAIL_DOMAINS:       # có thể gõ sai tên miền: gmial.com, gmail.con...
            for d in difflib.get_close_matches(domain, KNOWN_EMAIL_DOMAINS, n=2, cutoff=EMAIL_TYPO_CUTOFF):
                add(d)
    else:                                   # thiếu đuôi, ví dụ 'user@gmail'
        known = difflib.get_close_matches(domain + ".com", KNOWN_EMAIL_DOMAINS, n=1, cutoff=EMAIL_TYPO_CUTOFF)
        if known:
            add(known[0])
        else:
            for tld in (".com", ".vn", ".edu.vn"):
                add(domain + tld)
    return fixes[:3]

def _fix_date(day, sep, month, year_text):
    day, month = int(day), int(month)
    year = int(year_text) + (2000 if len(year_text) == 2 else 0)
    year = year if year >= 1 else 2000
    fmt = lambda d, m: f"{d:02d}{sep}{m:02d}{sep}{year_text}"
    fixes = []
    if 1 <= month <= 12 and day >= 1:       # ngày vượt quá số ngày của tháng -> lấy ngày cuối tháng
        fixes.append(fmt(min(day, calendar.monthrange(year, month)[1]), month))
    try:                                    # có thể bị đảo ngày và tháng
        _date(year, day, month)
        fixes.append(fmt(month, day))
    except ValueError:
        pass
    return list(dict.fromkeys(fixes))[:3]


def _fix_time(hour, sep, minute):
    hour, minute = int(hour), int(minute)
    fixes = [f"{min(hour, 23)}{sep}{min(minute, 59):02d}"]
    if minute > 59 and hour < 23 and minute - 60 <= 59:
        fixes.append(f"{hour + 1}{sep}{minute - 60:02d}")
    return list(dict.fromkeys(fixes))[:3]


def find_invalid_format_spans(text):
    """Các đoạn trông giống email / link / ngày / giờ nhưng viết SAI định dạng.
    Mỗi mục: {start, end, kind, text, suggestions} (suggestions: tối đa 3 cách sửa)."""
    found = []
    url_spans = [(s, e) for s, e, kind in find_special_spans(text) if kind == "url"]
    for m in EMAIL_CANDIDATE.finditer(text):
        start, end = m.span()
        if any(a <= start and end <= b for a, b in url_spans):    # email nằm trong link thì bỏ qua
            continue
        if not _email_ok(m.group()) or _domain_typo(m.group()):
            found.append({"start": start, "end": end, "kind": "email", "text": m.group(),
                          "suggestions": _fix_email(m.group())})
    for m in DATE_CANDIDATE.finditer(text):
        day, sep, month, year_text = m.groups()
        year = int(year_text) + (2000 if len(year_text) == 2 else 0)
        try:
            _date(max(year, 1), int(month), int(day))
        except ValueError:
            found.append({"start": m.start(), "end": m.end(), "kind": "date", "text": m.group(),
                          "suggestions": _fix_date(day, sep, month, year_text)})
    for m in TIME_CANDIDATE.finditer(text):
        hour, sep, minute = m.groups()
        if int(hour) > 23 or int(minute) > 59:
            found.append({"start": m.start(), "end": m.end(), "kind": "time", "text": m.group(),
                          "suggestions": _fix_time(hour, sep, minute)})
    return sorted(found, key=lambda item: item["start"])