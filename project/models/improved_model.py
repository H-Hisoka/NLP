import ast
from config import CONFUSION_SET
from evaluation.baseline import NGramLanguageModel
import math
from correction.candidate_generation import generate_candidates_fast
from correction.error_detection import levenshtein

class  ImproveSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel):
        self.lm = lm_model
        self.confusion_dict = self.load_confusion_set(CONFUSION_SET)
        self.cache = {}
    def load_confusion_set(self, duong_dan):
        with open(duong_dan, "r", encoding="utf-8") as f:
            cay = ast.parse(f.read())
        nut_dict = next(n for n in ast.walk(cay) if isinstance(n, ast.Dict))
        ket_qua = {}
        for k, v in zip(nut_dict.keys, nut_dict.values):
            ket_qua.setdefault(ast.literal_eval(k), set()).update(ast.literal_eval(v))
        return ket_qua
    def goi_y_cho_tu(self, words, i, k=3, nguong=8.0):
        """Trả về tối đa k ứng viên cho words[i].
        Nếu từ không bị nghi sai thì trả về [từ gốc]."""
        word = words[i]
        tu = word.lower()
        if not word.isalpha():
            return [word]
        prev_word = words[i - 1].lower() if i > 0 else "<s>"
        next_word = words[i + 1].lower() if i < len(words) - 1 else "</s>"

        if tu not in self.lm.vocab:
            khoang_cach = 2
        elif nguong is None:
            return [word]
        else:
            khoang_cach = 1

        khoa = (tu, khoang_cach)
        if khoa not in self.cache:
            self.cache[khoa] = (
                generate_candidates_fast(
                    tu, self.lm.vocab, self.lm.unigram_counts,
                    max_distance=khoang_cach, top_n=20
                ),
                self.get_confusion_candidates(tu, self.lm.vocab)
            )
        distance_candidates, confusion_candidates = self.cache[khoa]
        if not distance_candidates and not confusion_candidates:
            return [word]

        scored = self.score(
            tu, distance_candidates, confusion_candidates, prev_word, next_word
        )

        if tu in self.lm.vocab:
            # Real-word error: chỉ đổi khi ứng viên tốt hơn từ gốc đủ nhiều
            diem_goc = dict(scored)[tu]
            if scored[0][0] == tu or scored[0][1] - diem_goc <= nguong:
                return [word]
        return [c for c, _ in scored[:k]]
    def correct_sentence_top_k(self, sentence: str, k=3, nguong=8.0):
        words = sentence.split()
        return [self.goi_y_cho_tu(words, i, k, nguong) for i in range(len(words))]

    def correct_sentence(self, sentence: str, nguong=8.0):
        goi_y = self.correct_sentence_top_k(sentence, 1, nguong)
        return " ".join(g[0] for g in goi_y)
 
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




    