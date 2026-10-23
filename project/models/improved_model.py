import ast
import hashlib
import os
import pickle
from data.preprocessing import telex_candidates, vni_candidates
from collections import defaultdict
from config import (CONFUSION_SET, CACHE_FILE, DECODE_ACCENT_COST,
                    DECODE_EDIT_COST, DECODE_OOV_COST, SCORE_EDIT_WEIGHT,
                    SCORE_CONFUSION_BONUS, SCORE_FREQ_WEIGHT, SCORE_LEFT_WEIGHT,
                    SCORE_RIGHT_WEIGHT, SCORE_TONE_AWARE)
from correction.ranking import weighted_edit_distance
from evaluation.error_analysis import remove_accents
from evaluation.baseline import NGramLanguageModel
import math
from correction.candidate_generation import generate_candidates_fast
from correction.error_detection import levenshtein

class  ImproveSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel):
        self.lm = lm_model
        self.confusion_dict = self.load_confusion_set(CONFUSION_SET)
        self.cache = {}
        self.weights = {          # trọng số chấm điểm, có thể chỉnh lúc chạy (xem config.py)
            "edit": SCORE_EDIT_WEIGHT, "confusion": SCORE_CONFUSION_BONUS,
            "freq": SCORE_FREQ_WEIGHT, "left": SCORE_LEFT_WEIGHT,
            "right": SCORE_RIGHT_WEIGHT, "tone_aware": SCORE_TONE_AWARE,
        }
        self.accent_index = defaultdict(list)   # từ bỏ dấu -> các từ có dấu trong từ vựng
        for w in self.lm.vocab:
            if not w.startswith("<"):
                self.accent_index[remove_accents(w)].append(w)
    def load_confusion_set(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        dict_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict))
        result = {}
        for k, v in zip(dict_node.keys, dict_node.values):
            result.setdefault(ast.literal_eval(k), set()).update(ast.literal_eval(v))
        return result
    @staticmethod
    def is_word(token):
        """Chữ, hoặc chữ lẫn số (kiểu gõ VNI như 'vie65t'); chuỗi toàn số thì không."""
        return token.isalnum() and any(c.isalpha() for c in token)

    def typing_candidates(self, word):
        """Các từ tiếng Việt mà `word` có thể là do gõ Telex/VNI khi bộ gõ bị tắt (xếp theo tần suất)."""
        word = word.lower()
        if word in self.lm.vocab:
            return []
        found = set(telex_candidates(word, self.lm.vocab)) | set(vni_candidates(word, self.lm.vocab))
        return sorted(found, key=lambda c: -self.lm.unigram_counts[c])
    def suggest_word_candidates(self, words, i, k=3, threshold=8.0):
        """Trả về tối đa k ứng viên cho words[i].
        Nếu từ không bị nghi sai thì trả về [từ gốc]."""
        word = words[i]
        word_lower = word.lower()
        if not self.is_word(word):
            return [word]
        typed = self.typing_candidates(word_lower)
        if typed and word_lower not in self.lm.vocab:
            # Gõ Telex/VNI: đặt các từ chuyển đổi lên đầu, rồi bổ sung bằng gợi ý thông thường
            rest = [c for c in self.suggest_without_typing(words, i, k, threshold) if c not in typed]
            return (typed + rest)[:k]
        if any(c.isdigit() for c in word):
            return [word]
        return self.suggest_without_typing(words, i, k, threshold)

    def suggest_without_typing(self, words, i, k=3, threshold=8.0):
        word = words[i]
        word_lower = word.lower()
        prev_word = words[i - 1].lower() if i > 0 else "<s>"
        next_word = words[i + 1].lower() if i < len(words) - 1 else "</s>"

        if word_lower not in self.lm.vocab:
            max_distance = 2
        elif threshold is None:
            return [word]
        else:
            max_distance = 1

        cache_key = (word_lower, max_distance)
        if cache_key not in self.cache:
            self.cache[cache_key] = (
                generate_candidates_fast(
                    word_lower, self.lm.vocab, self.lm.unigram_counts,
                    max_distance=max_distance, top_n=20
                ),
                self.get_confusion_candidates(word_lower, self.lm.vocab)
            )
        distance_candidates, confusion_candidates = self.cache[cache_key]
        if not distance_candidates and not confusion_candidates:
            return [word]

        scored = self.score(
            word_lower, distance_candidates, confusion_candidates, prev_word, next_word
        )

        if word_lower in self.lm.vocab:
            # Real-word error: chỉ đổi khi ứng viên tốt hơn từ gốc đủ nhiều
            original_score = dict(scored)[word_lower]
            if scored[0][0] == word_lower or scored[0][1] - original_score <= threshold:
                return [word]
        return [c for c, _ in scored[:k]]
    def correct_sentence_top_k(self, sentence: str, k=3, threshold=8.0):
        words = sentence.split()
        suggestions = [self.suggest_word_candidates(words, i, k, threshold) for i in range(len(words))]
        if self.is_unaccented(sentence):
            # Câu không dấu: chọn cả chuỗi từ hợp lý nhất, đặt lên đầu danh sách gợi ý
            decoded = self.decode_sentence(sentence)
            for i, best in enumerate(decoded):
                if best == words[i].lower():
                    best = words[i]                  # giữ nguyên kiểu viết hoa của từ gốc
                if best != words[i] or suggestions[i] != [words[i]]:
                    suggestions[i] = self.merge_suggestions(words[i], best, suggestions[i], k)
        return suggestions

    def accent_candidates(self, word):
        """Từ gõ Telex/VNI chuyển được, rồi các từ có dấu mà bỏ dấu đi thì giống `word` (theo tần suất)."""
        typed = self.typing_candidates(word)
        accented = [c for c in self.accent_index.get(word.lower(), []) if c != word.lower() and c not in typed]
        return typed + sorted(accented, key=lambda c: -self.lm.unigram_counts[c])

    def merge_suggestions(self, word, best, suggestions, k=3):
        """Gộp: từ giải mã được, các cách khôi phục dấu (theo tần suất), rồi gợi ý còn lại."""
        merged = []
        for candidate in [best] + self.accent_candidates(word) + list(suggestions):
            if candidate not in merged:
                merged.append(candidate)
        return merged[:k]
    def is_unaccented(self, sentence):
        """Câu có từ hai từ trở lên và không có chữ nào mang dấu."""
        return len(sentence.split()) >= 2 and remove_accents(sentence) == sentence

    def word_options(self, word):
        """Các từ có thể là từ đúng của `word`, kèm chi phí chuyển đổi."""
        in_vocab = word in self.lm.vocab
        max_distance = 1 if in_vocab else 2
        cache_key = (word, max_distance)
        if cache_key not in self.cache:
            self.cache[cache_key] = (
                generate_candidates_fast(
                    word, self.lm.vocab, self.lm.unigram_counts,
                    max_distance=max_distance, top_n=20
                ),
                self.get_confusion_candidates(word, self.lm.vocab)
            )
        options = {word: 0.0 if in_vocab else DECODE_OOV_COST}
        for candidate in self.typing_candidates(word):
            options[candidate] = DECODE_ACCENT_COST
        for candidate in self.accent_index.get(word, []):
            if candidate != word:
                options[candidate] = DECODE_ACCENT_COST
        for candidate, dist, freq in self.cache[cache_key][0][:8]:
            if candidate not in options and not candidate.startswith("<"):
                options[candidate] = DECODE_EDIT_COST * dist
        return options

    def decode_sentence(self, sentence):
        """Viterbi: chọn cả chuỗi từ có tổng điểm bigram trừ chi phí chuyển đổi cao nhất."""
        words = sentence.lower().split()
        lattice = [self.word_options(w) if self.is_word(w) else {w: 0.0} for w in words]
        previous = {"<s>": (0.0, None)}          # từ -> (điểm tốt nhất, từ đứng trước)
        history = []
        for options in lattice:
            current = {}
            for word, cost in options.items():
                best_score, best_prev = max(
                    (prev_score + math.log(self.lm.get_bigram_prob(prev, word)), prev)
                    for prev, (prev_score, _) in previous.items()
                )
                current[word] = (best_score - cost, best_prev)
            history.append(current)
            previous = current
        last_word = max(
            previous,
            key=lambda w: previous[w][0] + math.log(self.lm.get_bigram_prob(w, "</s>"))
        )
        result = [last_word]
        for current in reversed(history[1:]):
            result.append(current[result[-1]][1])
        return result[::-1]

    def correct_sentence(self, sentence: str, threshold=8.0):
        suggestions = self.correct_sentence_top_k(sentence, 1, threshold)
        return " ".join(g[0] for g in suggestions)
 
    def get_confusion_candidates(self, word, vocabulary):
        candidates = set()
        confusion_dict = self.confusion_dict
        for wrong_part, correct_part in confusion_dict.items():
            if wrong_part not in word:
                continue
            for correct_part in correct_part:
                candidate = word.replace(wrong_part, correct_part)
                start=0 
                while True:
                    position= word.find(wrong_part, start)
                    if position == -1:
                        break
                    candidate = word[:position] + correct_part + word[position + len(wrong_part):]
                    if candidate in vocabulary and candidate != word:
                        candidates.add(candidate)
                    start = position + 1
        return list(candidates)

    def score(self, word, distance_candidates, confusion_candidates, prev_word, next_word):
        scored_candidates=[]
        candidates = {}
        for candidate, dist, freq in distance_candidates:
            candidates[candidate] = dist
        for candidate in confusion_candidates:
            candidates.setdefault(candidate, levenshtein(word, candidate))
        w = self.weights
        for candidate, dist in candidates.items():
            if w["tone_aware"]:
                dist = weighted_edit_distance(word, candidate)
            p1 = self.lm.get_bigram_prob(prev_word, candidate)
            p2 = self.lm.get_bigram_prob(candidate, next_word)
            score = (
                w["left"] * math.log(p1)
                + w["right"] * math.log(p2)
                - w["edit"] * dist
                + w["freq"] * math.log(self.lm.unigram_counts[candidate] + 1)
            )
            if candidate in confusion_candidates:
                score += w["confusion"]
            scored_candidates.append((candidate, score))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates


#  Lưu / nạp cache
def generate_fingerprint(checker):
    """Cache chỉ dùng lại được khi từ vựng và confusion set không đổi."""
    content = "\n".join(sorted(checker.lm.vocab))
    content += repr(sorted((k, sorted(v)) for k, v in checker.confusion_dict.items()))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_cache(checker):
    if not os.path.exists(CACHE_FILE):
        return
    with open(CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)
    if cache_data["van_tay"] == generate_fingerprint(checker):
        checker.cache = cache_data["cache"]
        print(f"Đã nạp cache ứng viên ({len(checker.cache)} mục)")
    else:
        print("Cache cũ không khớp từ vựng / confusion set, bỏ qua.")


def save_cache(checker):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"van_tay": generate_fingerprint(checker), "cache": checker.cache}, f)
    print(f"\nĐã lưu cache ứng viên ({len(checker.cache)} mục) vào {CACHE_FILE}")


    