def levenshtein(a: str, b: str, return_matrix=False):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # delete
                dp[i][j - 1] + 1,        # insert
                dp[i - 1][j - 1] + cost  # substitute / match
            )
    return (dp[n][m], dp) if return_matrix else dp[n][m]

def damerau_levenshtein(a: str, b: str, return_matrix=False):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
            if (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                dp[i][j] = min(
                    dp[i][j],
                    dp[i - 2][j - 2] + 1
                )
    if return_matrix:
        return dp[n][m], dp
    return dp[n][m]

def detect_errors(a: str, b: str):
    dist, dp = levenshtein(a, b, return_matrix=True)
    if dist == 0:
        return {
            "has_error": False,
            "distance": 0,
            "error_type": None
        }

    return {
        "has_error": True,
        "distance": dist,
        "error_type": "unknown"
    }
