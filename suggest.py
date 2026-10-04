"""
suggest.py - Hàm dùng chung cho app.py và error_analysis.py.

Lấy Top-K phương án sửa cho từng vị trí nghi sai của một câu / đoạn văn, dựa trên chính
các thành phần của đồ án: NGramLanguageModel (baseline.py), generate_candidates_fast
(candidate_generation.py), rank_candidates / get_top_k_candidates (ranking.py) và
ImproveSpellChecker (improved_model.py).

Quy tắc phát hiện giữ nguyên như hai checker hiện có: một từ bị nghi sai khi
"không có trong từ điển (lm.vocab) và chỉ gồm chữ cái". Khác biệt duy nhất: dấu câu dính ở
đầu/cuối từ (vd "tập.") được tách ra trước khi kiểm tra, để đoạn văn thật cũng dùng được.
"""
import heapq
import math
import re
import unicodedata

import pandas as pd

from config import PROCESSED_DATA_FILE, TOP_K, MAX_CANDIDATES
from baseline import NGramLanguageModel, BaselineSpellChecker
from improved_model import ImproveSpellChecker
from candidate_generation import generate_candidates_fast
from error_detection import levenshtein
from ranking import rank_candidates, get_top_k_candidates

EDIT_PENALTY = 1.5      # cùng hệ số phạt edit distance với baseline.py / improved_model.py
TEXT_COLUMN_CANDIDATES = ["sentence", "correct_sentence", "text", "clean_sentence",
                          "processed_sentence", "content", "câu"]


# ----------------------------------------------------------------------------- tiện ích
def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_marks(s):
    """Bỏ toàn bộ dấu (thanh, mũ, móc, trăng), đ -> d."""
    d = unicodedata.normalize("NFD", s)
    d = "".join(ch for ch in d if not unicodedata.combining(ch))
    return d.replace("đ", "d").replace("Đ", "D")


def match_case(orig, cand):
    if len(orig) > 1 and orig.isupper():
        return cand.upper()
    if orig[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


_TOKEN_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def split_core(token):
    """'tập.' -> ('', 'tập', '.')"""
    m = _TOKEN_RE.match(token)
    return m.group(1), m.group(2), m.group(3)


# --------------------------------------------------------------------- nạp mô hình ngôn ngữ
def load_language_model(path=PROCESSED_DATA_FILE, text_col=None, exclude=None):
    """
    Huấn luyện NGramLanguageModel từ file dữ liệu đã xử lý (câu đúng).
    - text_col: tên cột chứa câu; None thì tự dò.
    - exclude: tập câu cần loại khỏi dữ liệu huấn luyện (dùng để tránh rò rỉ khi đánh giá).
    Trả về (lm, số câu huấn luyện).
    """
    df = pd.read_csv(path)
    if text_col is None:
        lower = {c.lower(): c for c in df.columns}
        text_col = next((lower[c] for c in TEXT_COLUMN_CANDIDATES if c in lower), None)
        if text_col is None:
            text_col = next((c for c in df.columns if df[c].dtype == object), df.columns[0])
    sentences = [str(s) for s in df[text_col].dropna()]
    if exclude:
        sentences = [s for s in sentences if s not in exclude]
    lm = NGramLanguageModel()
    lm.train(sentences)
    return lm, len(sentences)


def build_checkers(lm):
    return {"baseline": BaselineSpellChecker(lm), "improved": ImproveSpellChecker(lm)}


# ------------------------------------------------------------------------ gợi ý cho một từ
def baseline_suggest_word(lm, word, prev_word, next_word, top_k=TOP_K):
    """Cùng cách chấm điểm với BaselineSpellChecker, nhưng trả về Top-k thay vì chỉ top-1."""
    cands = generate_candidates_fast(word.lower(), lm.vocab, lm.unigram_counts,
                                     max_distance=2, top_n=MAX_CANDIDATES)
    scored = []
    for cand, dist, freq in cands:
        p1 = lm.get_bigram_prob(prev_word, cand)
        p2 = lm.get_bigram_prob(cand, next_word)
        scored.append((cand, math.log(p1) + math.log(p2) - EDIT_PENALTY * dist))
    return get_top_k_candidates(rank_candidates(scored), top_k)


def suggest_word(checker, word, prev_word, next_word, top_k=TOP_K):
    if hasattr(checker, "suggest_word"):                      # ImproveSpellChecker
        return checker.suggest_word(word, prev_word, next_word, top_k)
    return baseline_suggest_word(checker.lm, word, prev_word, next_word, top_k)


def _fill(lm, word, cands, need):
    """Bổ sung ứng viên gần nhất trong từ điển để luôn đủ `need` phương án (dùng cho giao diện)."""
    have = {c for c, _ in cands}
    pool = heapq.nsmallest(
        need * 3,
        (v for v in lm.vocab if v.isalpha() and v not in have),
        key=lambda v: (levenshtein(word, v), -lm.unigram_counts[v]))
    base = min((s for _, s in cands), default=-30.0)
    for v in pool:
        if len(cands) >= need:
            break
        cands.append((v, base - levenshtein(word, v)))
    return cands


# --------------------------------------------------------------------- gợi ý cho cả câu/đoạn
def suggest_sentence(checker, text, top_k=TOP_K, min_candidates=0):
    """
    Trả về danh sách lỗi, mỗi lỗi là dict:
        index       chỉ số token (theo text.split())
        start, end  vị trí ký tự của từ lỗi trong text (không gồm dấu câu dính kèm)
        original    từ lỗi
        candidates  [(phương án, điểm), ...] đã xếp hạng, tối đa top_k (có thể rỗng)
    """
    text = nfc(text)
    lm = checker.lm
    toks = list(re.finditer(r"\S+", text))
    parts = [split_core(m.group()) for m in toks]
    errors = []
    for i, m in enumerate(toks):
        lead, core, trail = parts[i]
        low = core.lower()
        if not core or not core.isalpha() or low in lm.vocab:
            continue
        sentence_start = i == 0 or parts[i - 1][2][-1:] in (".", "!", "?", "…")
        sentence_end = i == len(toks) - 1 or trail[-1:] in (".", "!", "?", "…")
        prev_word = "<s>" if sentence_start else parts[i - 1][1].lower()
        next_word = "</s>" if sentence_end else parts[i + 1][1].lower()
        cands = list(suggest_word(checker, core, prev_word, next_word, top_k))
        if len(cands) < min_candidates:
            cands = _fill(lm, low, cands, min_candidates)
        start = m.start() + len(lead)
        errors.append({"index": i, "start": start, "end": start + len(core), "original": core,
                       "candidates": [(match_case(core, c), s) for c, s in cands]})
    return errors


def apply_choices(text, errors, choices):
    """choices: {vị trí trong `errors`: chuỗi thay thế, hoặc None nếu từ chối}."""
    text = nfc(text)
    for k in sorted(range(len(errors)), key=lambda k: errors[k]["start"], reverse=True):
        c = choices.get(k)
        if c:
            e = errors[k]
            text = text[:e["start"]] + c + text[e["end"]:]
    return text


def autocorrect(text, errors):
    """Chấp nhận phương án đầu tiên của mọi lỗi."""
    return apply_choices(text, errors,
                         {k: e["candidates"][0][0] for k, e in enumerate(errors) if e["candidates"]})


def confidence(candidates):
    """softmax trên điểm -> độ tin cậy 0..1 để hiển thị."""
    if not candidates:
        return []
    m = max(s for _, s in candidates)
    exps = [(c, math.exp(s - m)) for c, s in candidates]
    z = sum(e for _, e in exps)
    return [(c, e / z) for c, e in exps]