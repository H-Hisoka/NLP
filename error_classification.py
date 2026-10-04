from error_detection import levenshtein, damerau_levenshtein        
import re
from config import TEENCODE_REPLACEMENTS, ABBREVIATION_REPLACEMENTS, PUNCTUATION, EMOIJI_LIST
import unicodedata

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

def missing_character_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 1 and any(op[0] == "INSERT" for op in ops):
        return True
    return False

def extra_character_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 1 and any(op[0] == "DELETE" for op in ops):
        return True
    return False

def duplicate_character_error(a: str, b: str):
    if len(a) != len(b) + 1:
        return False
    for i in range(len(a) - 1):
        if a[i] == a[i + 1]:
            test = a[:i] + a[i + 1:]
            if test == b:
                return True
    return False

def swap_character_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 1 and any(op[0] == "TRANSPOSE" for op in ops):
        return True
    return False

def wrong_capitalization_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return False
    for op in ops:
        if op[0] == "SUBSTITUTE":
            wrong_char = op[1]
            correct_char = op[2]
            if (
                wrong_char.lower() == correct_char.lower()
                and wrong_char.islower()
                and correct_char.isupper()
            ):
                return True
    return False

def wrong_lowercase_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return False
    for op in ops:
        if op[0] == "SUBSTITUTE":
            wrong_char = op[1]
            correct_char = op[2]

            if (
                wrong_char.lower() == correct_char.lower()
                and wrong_char.isupper()
                and correct_char.islower()
            ):
                return True
    return False

def missing_space_error(a: str, b: str):
    if len(b) != len(a) + 1:
        return False

    for i in range(len(b)):
        if b[:i] + b[i+1:] == a:
            if b[i] == " ":
                return True

    return False

def extra_space_error(a: str, b: str):
    if len(a) != len(b) + 1:
        return False

    for i in range(len(a)):
        if a[:i] + a[i+1:] == b:
            if a[i] == " ":
                return True

    return False

def duplicate_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b) + 1:
        return False
    for i in range(len(words_a)):
        test = words_a[:i] + words_a[i + 1:]
        if test == words_b:
            return True
    return False

def missing_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_b) != len(words_a) + 1:
        return False
    for i in range(len(words_b)):
        test = words_b[:i] + words_b[i + 1:]
        if test == words_a:
            return True
    return False


def swap_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b):
        return False
    differences = [
        i
        for i in range(len(words_a))
        if words_a[i] != words_b[i]
    ]
    if len(differences) != 2:
        return False
    i, j = differences
    return (
        words_a[i] == words_b[j]
        and words_a[j] == words_b[i]
    )



def wrong_punctuation_error(text_a, text_b):
    if text_a[:-1] == "." and  text_b[:-1] != ".":
        return True
    elif text_a[:-1] == "!" and  text_b[:-1] != "!":
        return True
    elif text_a[:-1] == "?" and  text_b[:-1] != "?":
        return True
    return False

def missing_punctuation_error(text_a, text_b):
    if not text_a or not text_b:
        return False
    if (
        text_b[-1] in PUNCTUATION
        and text_a[-1] not in PUNCTUATION
    ):
        return True
    return False

def extra_punctuation_error(text_a, text_b):
    if len(text_a) != len(text_b) + 1:
        return False
    for i in range(len(text_a)):
        if text_a[:i] + text_a[i + 1:] == text_b:
            if text_b[i] in PUNCTUATION:
                return True
    return False

def duplicate_punctuation_error(text_a, text_b):
    if len(text_a) != len(text_b) + 1:
        return False
    for i in range(len(text_a) - 1):
        if (
            text_a[i] == text_a[i + 1]
            and text_a[i] in PUNCTUATION
        ):
            test = text_a[:i] + text_a[i + 1:]
            if test == text_b:
                return True
    return False


def spelling_error(a, b):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return False
    return any(
        op[0] == "SUBSTITUTE"
        for op in ops
    )


def word_order_error(a, b):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b):
        return False
    if words_a == words_b:
        return False
    return sorted(words_a) == sorted(words_b)

def word_replacement_error(text_a, text_b):
    words_a = text_a.split()
    words_b = text_b.split()
    if len(words_a) != len(words_b):
        return False
    differences = sum(
        x != y
        for x, y in zip(words_a, words_b)
    )

    return differences == 1

def word_insertion_error(text_a, text_b):
    words_a = text_a.split()
    words_b = text_b.split()
    return len(words_b) == len(words_a) + 1


def invalid_character_error(text_a, text_b):
    invalid_chars = set("!@#$%^&*()_+=[]{}|;:'\",<>/?`~")
    for char in text_b:
        if char in invalid_chars:
            return True
    return False

def wrong_emoji_error(text_a, text_b):
    emoji_pattern = r'[\U0001F300-\U0001FAFF]'
    emojis_a = re.findall(emoji_pattern, text_a)
    emojis_b = re.findall(emoji_pattern, text_b)
    if not emojis_a or not emojis_b:
        return False
    if len(emojis_a) == len(emojis_b) and emojis_a != emojis_b:
        return True
    return False

def missing_emoji_error(text_a, text_b):
    emojis_a = re.findall(EMOIJI_LIST, text_a)
    emojis_b = re.findall(EMOIJI_LIST, text_b)

    return len(emojis_b) == len(emojis_a) + 1

def duplicate_emoji_error(text_a, text_b):
    emojis_a = re.findall(EMOIJI_LIST, text_a)
    emojis_b = re.findall(EMOIJI_LIST, text_b)

    if len(emojis_a) != len(emojis_b) + 1:
        return False

    for i in range(len(emojis_a)):

        test = emojis_a[:i] + emojis_a[i + 1:]

        if test == emojis_b:
            return True

    return False


def teencode_error(text_a, text_b):
    for teencode, replacement in TEENCODE_REPLACEMENTS.items():
        if teencode in text_a and replacement in text_b:
            return True
    return False

def abbreviation_error(text_a, text_b):
    for abbreviation, replacement in ABBREVIATION_REPLACEMENTS.items():
        if abbreviation in text_a and replacement in text_b:
            return True
    return False

def code_switching_error(text_a, text_b):
    pattern = r'\b[a-zA-Z]+\b'
    words_a = re.findall(pattern, text_a)
    words_b = re.findall(pattern, text_b)
    words_b_lower = [
        word.lower()
        for word in words_b
    ]
    extra_english = [
        word
        for word in words_a
        if word.lower() not in words_b_lower
    ]
    return len(extra_english) > 0


def unicode_error(a, b):
    normalized_a = unicodedata.normalize("NFC", a)

    return a != normalized_a

def leading_trailing_space_error(text_a, text_b):
    return text_a.strip() == text_b and text_a != text_b

def url_error(a: str, b: str):
    url_a = re.findall(r'https?//[^\s]+', a)
    url_b = re.findall(r'https?://[^\s]+', b)
    if url_a and url_b:
        return True
    return False

def email_error(a: str, b: str):
    email_a = re.findall(r'\b[\w.-]+@[\w.-]+\b', a)
    email_b = re.findall(r'\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b', b)
    if email_a and email_b:
        return True
    return False

#Thêm hàm xử lý tổng



    

        
