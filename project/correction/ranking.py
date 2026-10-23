from correction.error_detection import levenshtein
import math
from data.preprocessing import tokenize
from correction.candidate_generation import generate_candidates
from config import TONE_CONFUSIONS

SWAP_COST = 0.5        # đảo hai ký tự liền nhau
#kt đầu vào là ký tự
def checkToken(a,b):
    if a==b:
        return 0
    elif (a,b) in TONE_CONFUSIONS:
        return 0.5
    else:
        return 1

def weighted_edit_distance(a,b,return_matrix=False):
    n,m=len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = checkToken(a[i-1],b[j-1])
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # delete
                dp[i][j - 1] + 1,        # insert
                dp[i - 1][j - 1] + cost  # substitute / match
            )

    return (dp[n][m], dp) if return_matrix else dp[n][m]

def error_logprob(observed, candidate, alpha=1.7):
    d = weighted_edit_distance(candidate, observed)
    return -alpha * d

def candidate_score(observed,
                    candidate,
                    prev_word,
                    lm,
                    word_freq,
                    alpha=1.7,
                    lm_weight=1.0,
                    freq_weight=0.15):
    err = error_logprob(observed, candidate, alpha=alpha)
    lm_log = math.log(lm.get_bigram_prob(prev_word, candidate))
    freq_log = math.log(word_freq[candidate] + 1)

    total = err + lm_weight * lm_log + freq_weight * freq_log
    return total

def rank_word_candidates(sentence, error_index, lm, max_distance=2, top_k=3):
    tokens = tokenize(sentence)
    observed = tokens[error_index]
    vocab = lm.vocab
    word_freq = lm.unigram_counts

    cands = generate_candidates(
        observed, vocab, word_freq,
        max_distance=max_distance, top_n=50
    )

    prev_word = tokens[error_index - 1] if error_index > 0 else "<s>"
    rows = []
    for cand, dist, freq in cands:
        score = candidate_score(
            observed, cand, prev_word, lm, word_freq
        )
        rows.append((cand, dist, freq, score))

    rows.sort(key=lambda x: x[-1], reverse=True)
    return rows[:top_k]