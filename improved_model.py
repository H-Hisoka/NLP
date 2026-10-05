import math
import os
import unicodedata
from collections import defaultdict

from config import CONFUSION_SET
from baseline import NGramLanguageModel
from candidate_generation import generate_candidates_fast
from error_detection import levenshtein
from ranking import rank_candidates, get_top_k_candidates

SPECIAL_TOKENS = {"<s>", "</s>"}
EDIT_PENALTY = 1.5          # hệ số phạt edit distance
CONFUSION_BONUS = 0.5       # thưởng cho ứng viên đến từ confusion set 
DECODE_COST = 0.3           # Telex/VNI giải mã ra từ hợp lệ: gần như chắc chắn là lỗi gõ
SPLIT_COST = 1.0            # thiếu một dấu cách
MERGE_COST = 1.0            # thừa một dấu cách
REAL_WORD_MARGIN = 2.0      # ứng viên real-word phải hơn bản gốc ít nhất ngần này (log-prob)


#  tiện ích tiếng Việt
def _nfd(s):
    return unicodedata.normalize("NFD", s)


def _nfc(s):
    return unicodedata.normalize("NFC", s)


VOWEL_LIST = list("aeiouy")
TONE_SET = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def get_tone(s):
    return next((c for c in _nfd(s) if c in TONE_SET), "")


def remove_tone(s):
    return _nfc("".join(c for c in _nfd(s) if c not in TONE_SET))


def strip_marks(s):
    d = "".join(c for c in _nfd(s) if not unicodedata.combining(c))
    return d.replace("đ", "d").replace("Đ", "D")


def _tone_index(s):
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


_TELEX_TONE = {"s": "\u0301", "f": "\u0300", "r": "\u0309", "x": "\u0303", "j": "\u0323"}
_VNI_TONE = {"1": "\u0301", "2": "\u0300", "3": "\u0309", "4": "\u0303", "5": "\u0323"}


def _has_vowel(out):
    return any(strip_marks(c) in VOWEL_LIST for c in out)


def decode_telex(raw):
    """Giải mã âm tiết gõ Telex nhưng bộ gõ chưa chuyển đổi ('vaof' -> 'vào'). None nếu không có phím Telex."""
    out, tone, changed = [], "", False
    for ch in _nfc(raw.lower()):
        if ch in _TELEX_TONE and _has_vowel(out):
            tone, changed = _TELEX_TONE[ch], True; continue
        if ch == "z" and _has_vowel(out):
            tone, changed = "", True; continue
        if out:
            last = out[-1]
            if ch == "d" and last == "d":
                out[-1] = "đ"; changed = True; continue
            if ch in "aeo" and last == ch:
                out[-1] = {"a": "â", "e": "ê", "o": "ô"}[ch]; changed = True; continue
            if ch == "w" and last in ("a", "o", "u"):
                if last == "o" and len(out) >= 2 and out[-2] == "u":
                    out[-2], out[-1] = "ư", "ơ"
                else:
                    out[-1] = {"a": "ă", "o": "ơ", "u": "ư"}[last]
                changed = True; continue
        out.append(ch)
    if not changed:
        return None
    res = _nfc("".join(out))
    return set_tone(res, tone) if tone else res


def decode_vni(raw):
    """Giải mã âm tiết gõ VNI chưa chuyển đổi ('la6p5' -> 'lập'). None nếu không có phím VNI."""
    out, tone, changed = [], "", False
    for ch in _nfc(raw.lower()):
        if ch in _VNI_TONE and _has_vowel(out):
            tone, changed = _VNI_TONE[ch], True; continue
        if ch == "0" and _has_vowel(out):
            tone, changed = "", True; continue
        if ch in "6789" and out:
            k = None
            if ch == "6":
                k = next((i for i in range(len(out) - 1, -1, -1) if out[i] in ("a", "e", "o")), None)
                if k is not None:
                    out[k] = {"a": "â", "e": "ê", "o": "ô"}[out[k]]
            elif ch == "7":
                k = next((i for i in range(len(out) - 1, -1, -1) if out[i] in ("o", "u")), None)
                if k is not None:
                    if out[k] == "o" and k > 0 and out[k - 1] == "u":
                        out[k - 1] = "ư"
                    out[k] = {"o": "ơ", "u": "ư"}[out[k]]
            elif ch == "8":
                k = next((i for i in range(len(out) - 1, -1, -1) if out[i] == "a"), None)
                if k is not None:
                    out[k] = "ă"
            elif ch == "9":
                k = next((i for i in range(len(out)) if out[i] == "d"), None)
                if k is not None:
                    out[k] = "đ"
            if k is not None:
                changed = True; continue
        out.append(ch)
    if not changed:
        return None
    res = _nfc("".join(out))
    return set_tone(res, tone) if tone else res


def _deletes1(s):
    return {s[:i] + s[i + 1:] for i in range(len(s))}


def _edit1(a, b):
    """Cách nhau đúng 1 thao tác chèn / xoá / thay / đảo hai ký tự liền kề."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


# confusion set
def load_confusion_dict(path):
    """Đọc confusion set: mỗi dòng 'a b' (hai chuỗi hay bị nhầm với nhau). Trả về dict hai chiều."""
    confusion_dict = {}
    if not path or not os.path.exists(path):
        print(f"[improved_model] Không tìm thấy confusion set: {path} -> bỏ qua confusion set")
        return confusion_dict
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pair = line.strip().split()
            if len(pair) == 2:
                a, b = pair
                confusion_dict.setdefault(a, set()).add(b)
                confusion_dict.setdefault(b, set()).add(a)
    return confusion_dict


#  hệ thống
class ImproveSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel, confusion_path=CONFUSION_SET,
                 real_word_margin=REAL_WORD_MARGIN):
        self.lm = lm_model
        self.confusion_dict = load_confusion_dict(confusion_path)   # đọc file một lần duy nhất
        self.real_word_margin = real_word_margin
        self._cand_cache = {}                                        # nhớ ứng viên theo từ (tính khá chậm)
        self._indexed_size = -1

    # chỉ mục dựng từ từ điển của mô hình ngôn ngữ (dựng lại nếu từ điển thay đổi)
    def _ensure_indexes(self):
        if self._indexed_size == len(self.lm.vocab):
            return
        self.words = {w for w in self.lm.vocab if w not in SPECIAL_TOKENS and w.isalpha()}
        self.by_key = defaultdict(list)            # dạng không dấu -> các âm tiết
        self.by_tone_base = {}                     # (bỏ thanh, thanh) -> âm tiết (chịu được khác vị trí dấu)
        self._del1 = defaultdict(set)              # chỉ mục xoá 1 ký tự để tìm từ hợp lệ gần giống
        for w in sorted(self.words, key=lambda x: -self.lm.unigram_counts[x]):
            self.by_key[strip_marks(w)].append(w)
            self.by_tone_base.setdefault((remove_tone(w), get_tone(w)), w)
            self._del1[w].add(w)
            for d in _deletes1(w):
                self._del1[d].add(w)
        self._cand_cache = {}
        self._indexed_size = len(self.lm.vocab)

    def _neighbors1(self, w):
        found = set()
        for d in _deletes1(w) | {w}:
            found |= self._del1.get(d, set())
        return {v for v in found if _edit1(w, v)}

    def _resolve(self, s):
        """Chuỗi giải mã Telex/VNI -> âm tiết có trong từ điển (nếu có)."""
        if s in self.words:
            return s
        return self.by_tone_base.get((remove_tone(s), get_tone(s)))

    # ---- phát hiện
    def is_suspect(self, core):
        """Từ nghi sai (kiểu không có trong từ điển): gồm chữ/số, có ít nhất một chữ cái (để bắt lỗi VNI như 'la6p5')."""
        return core.lower() not in self.lm.vocab and core.isalnum() and any(c.isalpha() for c in core)

    #  ứng viên
    def get_confusion_candidates(self, word, vocabulary):
        """Thay từng đoạn nhầm lẫn (vd 'ch' <-> 'tr') trong từ; giữ lại những từ có trong từ điển."""
        candidates = set()
        for wrong_part, replacements in self.confusion_dict.items():
            if wrong_part not in word:
                continue
            for correct_part in replacements:
                start = 0
                while True:
                    position = word.find(wrong_part, start)
                    if position == -1:
                        break
                    candidate = word[:position] + correct_part + word[position + len(wrong_part):]
                    if candidate in vocabulary and candidate != word:
                        candidates.add(candidate)
                    start = position + 1
        return list(candidates)

    def _context_logprob(self, prev_word, tokens, next_word):
        """log P của chuỗi bigram: prev -> tokens... -> next."""
        lp, left = 0.0, prev_word
        for t in tokens:
            lp += math.log(self.lm.get_bigram_prob(left, t))
            left = t
        return lp + math.log(self.lm.get_bigram_prob(left, next_word))

    def score(self, word, distance_candidates, confusion_candidates, prev_word, next_word, extra=None):
        dists = {}
        for candidate, dist, freq in distance_candidates:
            dists[candidate] = dist
        for candidate in confusion_candidates:
            dists.setdefault(candidate, levenshtein(word, candidate))
        for candidate, cost in (extra or {}).items():
            dists[candidate] = min(cost, dists.get(candidate, cost))

        scored_candidates = []
        for candidate, dist in dists.items():
            score = self._context_logprob(prev_word, candidate.split(" "), next_word) - EDIT_PENALTY * dist
            if candidate in confusion_candidates:
                score += CONFUSION_BONUS
            scored_candidates.append((candidate, score))
        return rank_candidates(scored_candidates)

    def suggest_word(self, word, prev_word, next_word, top_k=3):
        """Top-k phương án (ứng viên, điểm) cho một từ nghi sai (từ không có trong từ điển)."""
        self._ensure_indexes()
        word = word.lower()
        if word not in self._cand_cache:
            cands = generate_candidates_fast(word, self.lm.vocab, self.lm.unigram_counts,
                                             max_distance=2, top_n=22)
            self._cand_cache[word] = [c for c in cands if c[0] not in SPECIAL_TOKENS][:20]
        distance_candidates = self._cand_cache[word]
        confusion_candidates = [c for c in self.get_confusion_candidates(word, self.lm.vocab)
                                if c not in SPECIAL_TOKENS]

        extra = {}
        for fn in (decode_telex, decode_vni):                       # 1) Telex / VNI
            dec = fn(word)
            resolved = self._resolve(dec) if dec else None
            if resolved and resolved != word:
                extra[resolved] = DECODE_COST
        if word.isalpha():                                          # 2) thiếu dấu cách: "bữaăn" -> "bữa ăn"
            for p in range(1, len(word)):
                a, b = word[:p], word[p:]
                if a in self.words and b in self.words:
                    extra[a + " " + b] = SPLIT_COST

        ranked = self.score(word, distance_candidates, confusion_candidates, prev_word, next_word, extra)
        return get_top_k_candidates(ranked, top_k)

    def suggest_merge(self, first, second, prev_word, next_word, top_k=3):
        """Hai âm tiết liền nhau mà nối lại thành một âm tiết hợp lệ (thừa dấu cách): 'đa ng' -> 'đang'."""
        self._ensure_indexes()
        a, b = first.lower(), second.lower()
        merged = a + b
        if merged not in self.words or (a in self.lm.vocab and b in self.lm.vocab):
            return []
        score = self._context_logprob(prev_word, [merged], next_word) - EDIT_PENALTY * MERGE_COST
        return get_top_k_candidates([(merged, score)], top_k)

    def real_word_suggest(self, word, prev_word, next_word, top_k=3):
        """
        Real-word error: `word` là âm tiết hợp lệ nhưng không hợp ngữ cảnh. Chỉ đánh dấu khi:
          - cặp bigram (prev, word) hoặc (word, next) chưa từng xuất hiện trong dữ liệu huấn luyện, và
          - có ứng viên (cùng chữ không dấu / confusion set / cách 1 ký tự) mà cả hai cặp bigram đều đã từng
            xuất hiện, và điểm ngữ cảnh vượt bản gốc ít nhất `real_word_margin`.
        Trả về danh sách [(ứng viên, điểm)] hoặc [] nếu không nghi ngờ.
        """
        self._ensure_indexes()
        low = word.lower()
        if low not in self.words:
            return []
        seen = lambda a, b: self.lm.bigram_counts.get((a, b), 0) > 0
        if seen(prev_word, low) and seen(low, next_word):
            return []
        cands = set(self.by_key.get(strip_marks(low), [])) | self._neighbors1(low)
        cands |= {c for c in self.get_confusion_candidates(low, self.words)}
        cands.discard(low)
        orig_score = self._context_logprob(prev_word, [low], next_word)
        scored = []
        for c in cands:
            if not (seen(prev_word, c) and seen(c, next_word)):
                continue
            score = self._context_logprob(prev_word, [c], next_word) - EDIT_PENALTY * levenshtein(low, c)
            scored.append((c, score))
        ranked = rank_candidates(scored)
        if not ranked or ranked[0][1] - orig_score < self.real_word_margin:
            return []
        return get_top_k_candidates(ranked, top_k)

    def correct_sentence(self, sentence: str):
        """Sửa cả câu bằng phương án đầu tiên của mọi lỗi (dùng cùng bộ phát hiện với giao diện)."""
        import suggest                      # nhập khi cần để tránh vòng lặp import
        errors = suggest.suggest_sentence(self, sentence, top_k=1)
        return suggest.autocorrect(sentence, errors)
