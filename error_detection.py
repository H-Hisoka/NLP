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
    return (dp[n][m], dp) if return_matrix else dp[n][m]

def levenshtein_trace(a: str, b: str):
    dist, dp = levenshtein(a, b, return_matrix=True)
    i, j = len(a), len(b)
    ops = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1] and dp[i][j] == dp[i-1][j-1]:
            ops.append(("MATCH", a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("SUBSTITUTE", a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("DELETE", a[i-1], ""))
            i -= 1
        else:
            ops.append(("INSERT", "", b[j-1]))
            j -= 1

    ops.reverse()
    return dist, ops

def damerau_levenshtein_trace(a: str, b: str):
    dist, dp = damerau_levenshtein(a, b, return_matrix=True)
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        # MATCH
        if (
            i > 0 and j > 0
            and a[i - 1] == b[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            ops.append(("MATCH", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        # TRANSPOSE
        elif (
            i > 1 and j > 1
            and a[i - 1] == b[j - 2]
            and a[i - 2] == b[j - 1]
            and dp[i][j] == dp[i - 2][j - 2] + 1
        ):
            ops.append(("TRANSPOSE", a[i - 2:i], b[j - 2:j]))
            i -= 2
            j -= 2
        # SUBSTITUTE
        elif (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            ops.append(("SUBSTITUTE", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        # DELETE
        elif (
            i > 0
            and dp[i][j] == dp[i - 1][j] + 1
        ):
            ops.append(("DELETE", a[i - 1], ""))
            i -= 1
        # INSERT
        else:
            ops.append(("INSERT", "", b[j - 1]))
            j -= 1
    ops.reverse()
    return dist, ops

def detect_errors(a: str, b: str):
    dist, dp = damerau_levenshtein(a, b, return_matrix=True)
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
