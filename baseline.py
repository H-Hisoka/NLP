import math
from collections import Counter
from candidate_generation import generate_candidates_fast

class NGramLanguageModel:
   def __init__(self):
       self.unigram_counts = Counter()
       self.bigram_counts = Counter()
       self.vocab = set()
       
   def train(self, corpus):
       for sentence in corpus:
           words = ["<s>"] + sentence.lower().split() + ["</s>"]
           self.vocab.update(words)
           for w in words:
               self.unigram_counts[w] += 1
           for i in range(len(words) - 1):
               self.bigram_counts[(words[i], words[i+1])] += 1
               
   def get_bigram_prob(self, w1, w2, alpha=0.001):
       """Tính xác suất P(w2 | w1) có Laplace smoothing"""
       count_w1_w2 = self.bigram_counts.get((w1, w2), 0)
       count_w1 = self.unigram_counts.get(w1, 0)
       return (count_w1_w2 + alpha) / (count_w1 + alpha * len(self.vocab))

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

