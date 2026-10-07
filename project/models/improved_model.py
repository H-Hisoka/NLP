
from config import CONFUSION_SET
from evaluation.baseline import NGramLanguageModel
import math
from correction.candidate_generation import generate_candidates_fast
from correction.error_detection import levenshtein

class  ImproveSpellChecker:
    def _init_(self, lm_model: NGramLanguageModel):
        self.lm = lm_model
    def correct_sentence(self, sentence: str):
        words = sentence.split()
        corrected_words = []
        for i, word in enumerate(words):
            if word.lower() not in self.lm.vocab and word.isalpha():
                candidates = generate_candidates_fast(
                    word.lower(),
                    self.lm.vocab,
                    self.lm.unigram_counts,
                    max_distance=2,
                    top_n=20
                )
                if not candidates:
                    corrected_words.append(word)
                    continue

                # Lấy từ trước và từ sau làm ngữ cảnh.
                prev_word = (
                    words[i - 1].lower()
                    if i > 0
                    else "<s>"
                )
                next_word = (
                    words[i + 1].lower()
                    if i < len(words) - 1
                    else "</s>"
                )
                best_cand = word
                best_score = -float("inf")
                for cand, dist, freq in candidates:
                    # P(candidate | previous word)
                    p1 = self.lm.get_bigram_prob(
                        prev_word,
                        cand
                    )
                    # P(next word | candidate)
                    p2 = self.lm.get_bigram_prob(
                        cand,
                        next_word
                    )
                    score = (
                        math.log(p1)
                        + math.log(p2)
                        - (dist * 1.5)
                    )
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                corrected_words.append(best_cand)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    def get_confusion_candidates(word, vocabulary, CONFUSION_SET):
        candidates = set()
        # Load confusion set từ file
        with open(CONFUSION_SET, "r", encoding="utf-8") as f:
            confusion_pairs = [line.strip().split() for line in f if line.strip()]
            confusion_dict = {}
            for pair in confusion_pairs:
                if len(pair) == 2:
                    a, b = pair
                    confusion_dict.setdefault(a, set()).add(b)
                    confusion_dict.setdefault(b, set()).add(a)
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
        candidates=set()
        for candidate, dist, freq in distance_candidates:
            candidates.add((candidate, dist, freq))
        for candidate in confusion_candidates:
            candidates.add((candidate, levenshtein(word, candidate)))
        for candidate, dist in candidates:
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




    