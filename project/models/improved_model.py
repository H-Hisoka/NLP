import ast
import hashlib
import os
import pickle
from config import CONFUSION_SET, CACHE_FILE
from evaluation.baseline import NGramLanguageModel
import math
from correction.candidate_generation import generate_candidates_fast
from correction.error_detection import levenshtein

class  ImproveSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel):
        self.lm = lm_model
        self.confusion_dict = self.load_confusion_set(CONFUSION_SET)
        self.cache = {}
    def load_confusion_set(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        dict_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict))
        result = {}
        for k, v in zip(dict_node.keys, dict_node.values):
            result.setdefault(ast.literal_eval(k), set()).update(ast.literal_eval(v))
        return result
    def suggest_word_candidates(self, words, i, k=3, threshold=8.0):
        """Trả về tối đa k ứng viên cho words[i].
        Nếu từ không bị nghi sai thì trả về [từ gốc]."""
        word = words[i]
        word_lower = word.lower()
        if not word.isalpha():
            return [word]
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
        return [self.suggest_word_candidates(words, i, k, threshold) for i in range(len(words))]

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
        for candidate, dist in candidates.items():
            p1 = self.lm.get_bigram_prob(prev_word, candidate)
            p2 = self.lm.get_bigram_prob(candidate, next_word)
            score = (
                math.log(p1)
                + math.log(p2)
                - (dist * 1.5)
            )
            if candidate in confusion_candidates:
                score += 0.5
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


    