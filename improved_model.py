import math
import os

from config import CONFUSION_SET
from baseline import NGramLanguageModel
from candidate_generation import generate_candidates_fast
from error_detection import levenshtein
from ranking import rank_candidates, get_top_k_candidates


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


class ImproveSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel, confusion_path=CONFUSION_SET):
        self.lm = lm_model
        self.confusion_dict = load_confusion_dict(confusion_path)   # đọc file một lần duy nhất
        self._cand_cache = {}                                        # nhớ ứng viên theo từ (tính khá chậm)

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

    def score(self, word, distance_candidates, confusion_candidates, prev_word, next_word):
        """Chấm điểm: N-gram ngữ cảnh - phạt edit distance (+0.5 nếu đến từ confusion set)."""
        dists = {}
        for candidate, dist, freq in distance_candidates:
            dists[candidate] = dist
        for candidate in confusion_candidates:
            dists.setdefault(candidate, levenshtein(word, candidate))

        scored_candidates = []
        for candidate, dist in dists.items():
            p1 = self.lm.get_bigram_prob(prev_word, candidate)
            p2 = self.lm.get_bigram_prob(candidate, next_word)
            score = math.log(p1) + math.log(p2) - (dist * 1.5)
            if candidate in confusion_candidates:
                score += 0.5
            scored_candidates.append((candidate, score))
        return rank_candidates(scored_candidates)

    def suggest_word(self, word, prev_word, next_word, top_k=3):
        """Top-k phương án (ứng viên, điểm) cho một từ nghi sai."""
        word = word.lower()
        if word not in self._cand_cache:
            cands = generate_candidates_fast(
                word,
                self.lm.vocab,
                self.lm.unigram_counts,
                max_distance=2,
                top_n=22
            )
            self._cand_cache[word] = [c for c in cands if c[0] not in ("<s>", "</s>")][:20]
        distance_candidates = self._cand_cache[word]
        confusion_candidates = [c for c in self.get_confusion_candidates(word, self.lm.vocab)
                                if c not in ("<s>", "</s>")]
        ranked = self.score(word, distance_candidates, confusion_candidates, prev_word, next_word)
        return get_top_k_candidates(ranked, top_k)

    def correct_sentence(self, sentence: str):
        words = sentence.split()
        corrected_words = []
        for i, word in enumerate(words):
            if word.lower() not in self.lm.vocab and word.isalpha():
                prev_word = words[i - 1].lower() if i > 0 else "<s>"
                next_word = words[i + 1].lower() if i < len(words) - 1 else "</s>"
                best = self.suggest_word(word, prev_word, next_word, top_k=1)
                corrected_words.append(best[0][0] if best else word)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)