from correction.error_detection import levenshtein
import math
from data.preprocessing import tokenize
from correction.candidate_generation import generate_candidates

def error_logprob(observed, candidate, alpha=1.7):
    d = levenshtein(candidate, observed)
    return -alpha * d

def candidate_score(observed,
                    candidate,
                    left_context,
                    lm,
                    word_freq,
                    alpha=1.7,
                    lm_weight=1.0,
                    freq_weight=0.15):
    err = error_logprob(observed, candidate, alpha=alpha)
    lm_log = math.log(lm.prob(candidate, left_context) + 1e-15)
    freq_log = math.log(word_freq[candidate] + 1)

    total = err + lm_weight * lm_log + freq_weight * freq_log
    return total

def rank_word_candidates(sentence, error_index, lm, max_distance=2, top_k=3):
    tokens = tokenize(sentence)
    observed = tokens[error_index]

    cands = generate_candidates(
        observed, vocab, word_freq,
        max_distance=max_distance, top_n=50
    )

    rows = []
    for cand, dist, freq in cands:
        left_context = tokens[max(0, error_index-(lm.n-1)):error_index]
        score = candidate_score(
            observed, cand, left_context, lm, word_freq
        )
        rows.append((cand, dist, freq, score))

    rows.sort(key=lambda x: x[-1], reverse=True)
    return rows[:top_k]
