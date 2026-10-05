import math
from candidate_generation import generate_candidates_fast
from train_language_model import NGramLanguageModel

class BaselineSpellChecker:
    def __init__(self, lm_model: NGramLanguageModel):
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
                    # Score của baseline:
                    # N-gram context + phạt edit distance
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

