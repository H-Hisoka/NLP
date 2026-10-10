from correction.error_detection import levenshtein

def generate_candidates(observed,
                        vocabulary,
                        freq,
                        max_distance=2,
                        top_n=20):
    scored = []
    for cand in vocabulary:
        d = levenshtein(observed, cand)
        if d <= max_distance:
            scored.append((cand, d, freq[cand]))

    scored.sort(key=lambda x: (x[1], -x[2], x[0]))
    return scored[:top_n]

def generate_candidates_fast(observed, vocabulary, freq,
                             max_distance=2, top_n=20):
    scored = []
    for cand in vocabulary:
        d = levenshtein(observed, cand)
        if d <= max_distance and abs(len(cand)-len(observed))<=max_distance:
            scored.append((cand, d, freq[cand]))    
    scored.sort(key=lambda x: (x[1], -x[2], x[0]))
    return scored[:top_n]

