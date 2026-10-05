import os
import random
import re
import unicodedata
from collections import defaultdict

import pandas as pd

try:
    from config import RANDOM_SEED, ERRORS_PER_SENTENCE
except ImportError:
    RANDOM_SEED, ERRORS_PER_SENTENCE = 42, 3


# --------------------------------------------------------------------------- tiện ích tiếng Việt
def _nfd(s):
    return unicodedata.normalize("NFD", s)


def _nfc(s):
    return unicodedata.normalize("NFC", s)


TONES = ["", "\u0301", "\u0300", "\u0309", "\u0303", "\u0323"]   # ngang, sắc, huyền, hỏi, ngã, nặng
TONE_SET = set(TONES[1:])
MODIFIERS = {"\u0302", "\u0306", "\u031b"}                        # mũ (â ê ô), trăng (ă), móc (ơ ư)
VOWEL_LIST = list("aeiouy")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)                    # một "từ" gồm toàn chữ cái


def get_tone(s):
    return next((c for c in _nfd(s) if c in TONE_SET), "")


def remove_tone(s):
    return _nfc("".join(c for c in _nfd(s) if c not in TONE_SET))


def remove_modifiers(s):
    """Bỏ dấu mũ/móc/trăng, giữ dấu thanh: 'ường' -> 'ùng' (không đụng tới đ)."""
    return _nfc("".join(c for c in _nfd(s) if c not in MODIFIERS))


def strip_marks(s):
    d = "".join(c for c in _nfd(s) if not unicodedata.combining(c))
    return d.replace("đ", "d").replace("Đ", "D")


def _tone_index(s):
    """Vị trí nguyên âm mang dấu thanh (s không có dấu thanh, chữ thường)."""
    chars = list(s)
    idx = [i for i, c in enumerate(chars) if strip_marks(c) in VOWEL_LIST]
    if not idx:
        return None
    if len(idx) > 1 and idx[0] > 0 and ((chars[idx[0] - 1] == "q" and chars[idx[0]] == "u") or
                                        (chars[idx[0] - 1] == "g" and chars[idx[0]] == "i")):
        idx = idx[1:]
    if len(idx) == 1:
        return idx[0]
    modified = [i for i in idx if chars[i] not in VOWEL_LIST]
    if modified:
        return modified[-1]
    if len(idx) == 3:
        return idx[1]
    if idx[-1] < len(chars) - 1:
        return idx[-1]
    return idx[0]


def set_tone(s, tone):
    s = remove_tone(_nfc(s))
    i = _tone_index(s)
    if not tone or i is None:
        return s
    return _nfc(s[:i] + _nfd(s[i]) + tone + s[i + 1:])


# Telex / VNI
_TELEX_KEY = {"\u0301": "s", "\u0300": "f", "\u0309": "r", "\u0303": "x", "\u0323": "j"}
_VNI_KEY = {"\u0301": "1", "\u0300": "2", "\u0309": "3", "\u0303": "4", "\u0323": "5"}


def _telex_vni_parts(base, vni):
    out = []
    for ch in base:
        if ch == "đ":
            out.append("d9" if vni else "dd"); continue
        d = _nfd(ch); b, m = d[0], d[1:]
        if m == "\u0302":
            out.append(b + "6" if vni else b * 2)
        elif m == "\u0306":
            out.append(b + "8" if vni else b + "w")
        elif m == "\u031b":
            out.append(b + "7" if vni else b + "w")
        else:
            out.append(ch)
    return out


def to_telex(w, tone_after_vowel=False):
    tone = get_tone(w); parts = _telex_vni_parts(remove_tone(w), vni=False); key = _TELEX_KEY.get(tone, "")
    if tone_after_vowel and key:
        last_v = max((i for i, p in enumerate(parts) if strip_marks(p[0]) in VOWEL_LIST), default=len(parts) - 1)
        parts.insert(last_v + 1, key)
        return "".join(parts)
    return "".join(parts) + key


def to_vni(w):
    return "".join(_telex_vni_parts(remove_tone(w), vni=True)) + _VNI_KEY.get(get_tone(w), "")


# Phụ âm đầu / vần
ONSETS = sorted(["ngh", "ng", "nh", "ch", "tr", "th", "ph", "kh", "gh", "gi", "qu", "b", "c", "d", "đ",
                 "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x"], key=len, reverse=True)
FINALS = ["nh", "ng", "ch", "c", "t", "n", "m", "p"]
ONSET_GROUPS = [("ch", "tr"), ("s", "x"), ("d", "gi", "r"), ("l", "n"), ("c", "k"), ("ng", "ngh"), ("g", "gh")]
FINAL_GROUPS = [("c", "t"), ("n", "ng"), ("nh", "n"), ("ch", "c")]


def _partners(groups):
    m = defaultdict(set)
    for g in groups:
        for a in g:
            m[a] |= set(g) - {a}
    return m


def split_onset(w):
    for o in ONSETS:
        if w.startswith(o):
            rest = w[len(o):]
            if o in ("gi", "qu") and (not rest or strip_marks(rest[0]) not in VOWEL_LIST):
                continue
            return o, rest
    return "", w


def split_final(w):
    for f in FINALS:
        if w.endswith(f) and len(w) > len(f) and strip_marks(w[-len(f) - 1]) in VOWEL_LIST:
            return w[:-len(f)], f
    return w, ""


def confusion_variants(w):
    out = set()
    o, rest = split_onset(w)
    if o:
        out |= {p + rest for p in _partners(ONSET_GROUPS).get(o, ())}
    stem, f = split_final(w)
    if f:
        out |= {stem + p for p in _partners(FINAL_GROUPS).get(f, ())}
    out.discard(w)
    return out


class TuDien:
    """Tập âm tiết hợp lệ lấy từ các câu đúng (dùng để sinh lỗi real-word)."""

    def __init__(self, sentences):
        self.words = {m.group().lower() for s in sentences for m in WORD_RE.finditer(str(s))}
        self.by_key = defaultdict(list)
        for w in sorted(self.words):
            self.by_key[strip_marks(w)].append(w)
        self._near = {}

    def gan_giong(self, w):
        """Các âm tiết hợp lệ khác `w` đúng 1 thao tác chèn/xoá/thay ký tự."""
        if w not in self._near:
            self._near[w] = [v for v in sorted(self.words) if v != w and _cach_1_ky_tu(w, v)]
        return self._near[w]


def _cach_1_ky_tu(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


# hàm tạo lỗi cho MỘT từ (chữ thường)
KEY_NEIGHBORS = {'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tuhg', 'u': 'yihj',
                 'i': 'uojk', 'o': 'iplk', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
                 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop', 'z': 'asx', 'x': 'zsdc',
                 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'}
_ALLOWED_MOD = {"a": ["", "\u0302", "\u0306"], "o": ["", "\u0302", "\u031b"], "u": ["", "\u031b"],
                "e": ["", "\u0302"]}


def _missing_tone(w, rng, vocab):
    r = remove_tone(w); return r if r != w else None


def _wrong_tone(w, rng, vocab):
    base, cur = remove_tone(w), get_tone(w)
    return set_tone(base, rng.choice([t for t in TONES[1:] if t != cur]))


def _missing_diacritic(w, rng, vocab):
    r = remove_modifiers(w); return r if r != w else None


def _missing_all_marks(w, rng, vocab):
    r = strip_marks(w); return r if r != w else None


def _d_stroke(w, rng, vocab):
    return w.replace("đ", "d") if "đ" in w else None


def _wrong_vowel_mark(w, rng, vocab):
    idx = [i for i, c in enumerate(w) if _nfd(c)[0] in _ALLOWED_MOD]
    if not idx:
        return None
    i = rng.choice(idx); d = _nfd(w[i]); base, rest = d[0], d[1:]
    cur = next((c for c in rest if c in MODIFIERS), "")
    tone = "".join(c for c in rest if c in TONE_SET)
    new = rng.choice([m for m in _ALLOWED_MOD[base] if m != cur])
    return w[:i] + _nfc(base + new + tone) + w[i + 1:]


def _wrong_character(w, rng, vocab):
    i = rng.randrange(len(w)); pool = KEY_NEIGHBORS.get(strip_marks(w[i]))
    return w[:i] + rng.choice(pool) + w[i + 1:] if pool else None


def _extra_character(w, rng, vocab):
    i = rng.randrange(1, len(w) + 1)
    return w[:i] + rng.choice(KEY_NEIGHBORS.get(strip_marks(w[i - 1]), "a")) + w[i:]


def _duplicate_character(w, rng, vocab):
    i = rng.randrange(len(w)); return w[:i] + w[i] * 2 + w[i + 1:]


def _missing_character(w, rng, vocab):
    if len(w) < 3:
        return None
    i = rng.randrange(len(w)); return w[:i] + w[i + 1:]


def _swap_character(w, rng, vocab):
    idx = [i for i in range(len(w) - 1) if w[i] != w[i + 1]]
    if not idx:
        return None
    i = rng.choice(idx); return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _make_initial(group_pairs):
    partners = _partners(group_pairs)

    def f(w, rng, vocab):
        o, rest = split_onset(w); ps = sorted(partners.get(o, ()))
        return rng.choice(ps) + rest if o and ps else None
    return f


def _make_final(group_pairs):
    partners = _partners(group_pairs)

    def f(w, rng, vocab):
        stem, fin = split_final(w); ps = sorted(partners.get(fin, ()))
        return stem + rng.choice(ps) if fin and ps else None
    return f


def _telex_leftover(w, rng, vocab):
    if strip_marks(w) == w:
        return None
    return to_telex(w, tone_after_vowel=rng.random() < 0.5)


def _vni_leftover(w, rng, vocab):
    return to_vni(w) if strip_marks(w) != w else None


def _telex_tone_key_left(w, rng, vocab):
    t = get_tone(w)
    return remove_tone(w) + _TELEX_KEY[t] if t else None


def _real_word_tone(w, rng, vocab):
    c = [v for v in vocab.by_key.get(strip_marks(w), []) if v != w]
    return rng.choice(c) if c else None


def _real_word_similar(w, rng, vocab):
    """Thay bằng một âm tiết HỢP LỆ khác gần giống: ưu tiên nhầm phụ âm đầu/vần, không có thì cách 1 ký tự."""
    c = sorted(v for v in confusion_variants(w) if v in vocab.words)
    return rng.choice(c or vocab.gan_giong(w) or [None])


def _mixed_tone_char(w, rng, vocab):
    first = rng.choice([_missing_tone, _wrong_tone, _missing_diacritic, _wrong_vowel_mark])(w, rng, vocab)
    if not first:
        return None
    second = rng.choice([_wrong_character, _missing_character, _swap_character, _extra_character])(first, rng, vocab)
    return second if second and second != w else None


#  hàm tạo lỗi cho cả câu
def _restore_case(orig, new):
    if len(orig) > 1 and orig.isupper():
        return new.upper()
    return new[:1].upper() + new[1:] if orig[:1].isupper() else new


def _apply_word_error(text, rng, fn, vocab):
    spans = [m for m in WORD_RE.finditer(text) if len(m.group()) >= 2]
    rng.shuffle(spans)
    for m in spans:                                   # thử lần lượt các từ cho tới khi có từ áp dụng được
        w = m.group().lower()
        new = fn(w, rng, vocab)
        if new and new != w:
            return text[:m.start()] + _restore_case(m.group(), new) + text[m.end():]
    return text


def _missing_space(text, rng, vocab):
    pos = [m.start() for m in re.finditer(r"(?<=[^\W\d_]) (?=[^\W\d_])", text)]
    if not pos:
        return text
    p = rng.choice(pos); return text[:p] + text[p + 1:]


def _extra_space_in_word(text, rng, vocab):
    spans = [m for m in WORD_RE.finditer(text) if len(m.group()) >= 4]
    if not spans:
        return text
    m = rng.choice(spans); p = m.start() + rng.randrange(2, len(m.group()) - 1)
    return text[:p] + " " + text[p:]


WORD_ERROR_FUNCS = {
    "missing_tone": _missing_tone, "wrong_tone": _wrong_tone, "missing_diacritic": _missing_diacritic,
    "missing_all_marks": _missing_all_marks, "d_stroke_error": _d_stroke, "wrong_vowel_mark": _wrong_vowel_mark,
    "wrong_character": _wrong_character, "extra_character": _extra_character,
    "duplicate_character": _duplicate_character, "missing_character": _missing_character,
    "swap_character": _swap_character,
    "initial_ch_tr": _make_initial([("ch", "tr")]), "initial_s_x": _make_initial([("s", "x")]),
    "initial_d_gi_r": _make_initial([("d", "gi", "r")]), "initial_l_n": _make_initial([("l", "n")]),
    "initial_ng_ngh_g_gh": _make_initial([("ng", "ngh"), ("g", "gh")]), "initial_c_k": _make_initial([("c", "k")]),
    "final_c_t": _make_final([("c", "t")]), "final_n_ng": _make_final([("n", "ng")]),
    "final_nh_n": _make_final([("nh", "n")]), "final_ch_c": _make_final([("ch", "c")]),
    "telex_leftover": _telex_leftover, "vni_leftover": _vni_leftover, "telex_tone_key_left": _telex_tone_key_left,
    "real_word_tone": _real_word_tone, "real_word_similar": _real_word_similar,
    "mixed_tone_char": _mixed_tone_char,
}
_MULTI_POOL = ["missing_tone", "wrong_tone", "missing_diacritic", "wrong_character", "missing_character",
               "swap_character", "initial_ch_tr", "initial_s_x", "initial_d_gi_r", "initial_l_n", "final_n_ng",
               "final_c_t", "real_word_tone"]


def _multiple_errors(text, rng, vocab):
    """2..ERRORS_PER_SENTENCE từ khác nhau trong cùng một câu, mỗi từ một loại lỗi ngẫu nhiên."""
    n = rng.randint(2, max(2, ERRORS_PER_SENTENCE))
    spans = [m for m in WORD_RE.finditer(text) if len(m.group()) >= 2]
    rng.shuffle(spans)
    edits = []
    for m in spans:
        if len(edits) == n:
            break
        w = m.group().lower()
        for name in rng.sample(_MULTI_POOL, len(_MULTI_POOL)):
            new = WORD_ERROR_FUNCS[name](w, rng, vocab)
            if new and new != w:
                edits.append((m.start(), m.end(), _restore_case(m.group(), new))); break
    if len(edits) < 2:
        return text
    for s, e, new in sorted(edits, reverse=True):
        text = text[:s] + new + text[e:]
    return text


DANH_SACH_LOI = list(WORD_ERROR_FUNCS) + ["missing_space", "extra_space_in_word", "multiple_errors"]
# giữ thứ tự nhóm cho dễ đọc: dấu -> ký tự -> phụ âm đầu -> vần -> Telex/VNI -> real-word -> tách từ -> tổng hợp
_ORDER = ["missing_tone", "wrong_tone", "missing_diacritic", "missing_all_marks", "d_stroke_error",
          "wrong_vowel_mark", "wrong_character", "extra_character", "duplicate_character", "missing_character",
          "swap_character", "initial_ch_tr", "initial_s_x", "initial_d_gi_r", "initial_l_n",
          "initial_ng_ngh_g_gh", "initial_c_k", "final_c_t", "final_n_ng", "final_nh_n", "final_ch_c",
          "telex_leftover", "vni_leftover", "telex_tone_key_left", "real_word_tone", "real_word_similar",
          "missing_space", "extra_space_in_word", "mixed_tone_char", "multiple_errors"]
assert sorted(_ORDER) == sorted(DANH_SACH_LOI) and len(_ORDER) == 30
DANH_SACH_LOI = _ORDER


def tao_loi(text, loai_loi, rng=None, vocab=None):
    """Tạo câu lỗi từ câu đúng `text` theo loại lỗi `loai_loi` (xem DANH_SACH_LOI)."""
    if not isinstance(text, str):
        return text
    rng = rng or random
    if loai_loi == "missing_space":
        return _missing_space(text, rng, vocab)
    if loai_loi == "extra_space_in_word":
        return _extra_space_in_word(text, rng, vocab)
    if loai_loi == "multiple_errors":
        return _multiple_errors(text, rng, vocab)
    if loai_loi in WORD_ERROR_FUNCS:
        return _apply_word_error(text, rng, WORD_ERROR_FUNCS[loai_loi], vocab)
    return text


#  đọc dulieu.csv, tạo file lỗi
def tao_file_loi():
    duong_dan = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dulieu.csv")
    df = pd.read_csv(duong_dan, encoding="utf-8-sig")
    for cot in ("id", "correct_text"):
        if cot not in df.columns:
            raise ValueError(f"File dulieu.csv phải có cột '{cot}'")

    rng = random.Random(RANDOM_SEED)
    vocab = TuDien(df["correct_text"])
    data = []
    for _, row in df.iterrows():
        for loai_loi in DANH_SACH_LOI:
            data.append({"id": row["id"], "correct_text": row["correct_text"],
                         "error_text": tao_loi(row["correct_text"], loai_loi, rng, vocab),
                         "error_type": loai_loi})
    error_df = pd.DataFrame(data)
    error_df.to_csv("dulieu_loi.csv", index=False, encoding="utf-8-sig")

    print("Đã tạo file: dulieu_loi.csv")
    print("Số câu gốc:", len(df))
    print("Số dòng lỗi:", len(error_df))
    print("Số loại lỗi:", error_df["error_type"].nunique())
    unchanged = (error_df["correct_text"] == error_df["error_text"]).groupby(error_df["error_type"]).mean()
    thieu = unchanged[unchanged > 0.2]
    if len(thieu):
        print("Loại lỗi có hơn 20% câu không tạo được lỗi (câu không có từ phù hợp):")
        print(thieu.round(2).to_string())


if __name__ == "__main__":
    tao_file_loi()