from error_detection import levenshtein, damerau_levenshtein        

def pretty_trace(a: str, b: str):
    d, ops,I,D,S = improve_print_levenshtein_trace(a, b)
    print("Distance:", d)
    for op in ops:
        print(f"{op[0]}:{op[1]}->{op[2]}")

def count_edit_operations(opsa, a: str, b: str):
    d, ops,I,D,S = improve_print_levenshtein_trace(a, b)
    dic={
        "Insert:":I,
        "Delete:":D,
        "Substitute:":S,
    }
    return dic

def improve_print_levenshtein_trace(a: str, b: str):
    dist, dp = levenshtein(a, b, return_matrix=True)
    i, j = len(a), len(b)
    ops = []
    I,D,S=0,0,0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1] and dp[i][j] == dp[i-1][j-1]:
            ops.append(("MATCH:", a[i-1],"->", b[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("SUBSTITUTE:", a[i-1], b[j-1]))
            i -= 1; j -= 1
            S+=1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("DELETE", a[i-1], ""))
            i -= 1
            D+=1
        else:
            ops.append(("INSERT", "", b[j-1]))
            j -= 1
            I+=1

    ops.reverse()
    return dist, ops, I,D,S
    

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

def classify_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 0:
        return "No Error"
    error_types = set(op[0] for op in ops)
    if len(error_types) == 1:
        return error_types.pop()
    else:
        return "Mixed Errors"