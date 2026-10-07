def rank_candidates(score_candidates):
    ranked_candidates = sorted(score_candidates, key=lambda x: x[1], reverse=True)
    return ranked_candidates

def get_top_k_candidates(ranked_candidates, k):
    return ranked_candidates[:k]