from error_detection import damerau_levenshtein_trace       
import re
from config import TEENCODE_REPLACEMENTS, ABBREVIATION_REPLACEMENTS, PUNCTUATION, EMOIJI_LIST
import unicodedata

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
    if dist == 1:
        for i, op in enumerate(ops):
            if op[0] == "INSERT":
                # Tính vị trí dựa trên các thao tác trước đó
                position = sum(
                    1 for x in ops[:i]
                    if x[0] != "INSERT"
                )
                return {
                    "error_type": "missing_character",
                    "position": position,
                    "wrong_char": "",
                    "correct_char": op[2]
                }
    return None


def extra_character_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 1:
        position = 0
        for op in ops:
            if op[0] == "DELETE":
                return {
                    "error_type": "extra_character",
                    "position": position,
                    "wrong_char": op[1],
                    "correct_char": ""
                }
            if op[0] != "INSERT":
                position += 1
    return None


def duplicate_character_error(a: str, b: str):
    if len(a) != len(b) + 1:
        return None
    for i in range(len(a) - 1):
        if a[i] == a[i + 1]:
            test = a[:i] + a[i + 1:]
            if test == b:
                return {
                    "error_type": "duplicate_character",
                    "position": i,
                    "wrong_char": a[i],
                    "correct_char": ""
                }
    return None


def swap_character_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist == 1:
        position = 0
        for op in ops:
            if op[0] == "TRANSPOSE":
                return {
                    "error_type": "character_transposition",
                    "positions": [position, position + 1],
                    "wrong_chars": op[1],
                    "correct_chars": op[2]
                }
            if op[0] != "INSERT":
                position += 1
    return None


def wrong_capitalization_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return None
    position = 0
    for op in ops:
        if op[0] == "SUBSTITUTE":
            wrong_char = op[1]
            correct_char = op[2]
            if (
                wrong_char.lower() == correct_char.lower()
                and wrong_char.islower()
                and correct_char.isupper()
            ):
                return {
                    "error_type": "wrong_capitalization",
                    "position": position,
                    "wrong_char": wrong_char,
                    "correct_char": correct_char
                }
            position += 1
        elif op[0] != "INSERT":
            position += 1
    return None


def wrong_lowercase_error(a: str, b: str):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return None
    position = 0
    for op in ops:
        if op[0] == "SUBSTITUTE":
            wrong_char = op[1]
            correct_char = op[2]
            if (
                wrong_char.lower() == correct_char.lower()
                and wrong_char.isupper()
                and correct_char.islower()
            ):
                return {
                    "error_type": "wrong_lowercase",
                    "position": position,
                    "wrong_char": wrong_char,
                    "correct_char": correct_char
                }
            position += 1
        elif op[0] != "INSERT":
            position += 1
    return None


def missing_space_error(a: str, b: str):
    if len(b) != len(a) + 1:
        return None
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            if b[i] == " ":
                return {
                    "error_type": "missing_space",
                    "position": i,
                    "wrong_char": "",
                    "correct_char": " "
                }
    return None


def extra_space_error(a: str, b: str):
    if len(a) != len(b) + 1:
        return None
    for i in range(len(a)):
        if a[:i] + a[i + 1:] == b:
            if a[i] == " ":
                return {
                    "error_type": "extra_space",
                    "position": i,
                    "wrong_char": " ",
                    "correct_char": ""
                }
    return None


def duplicate_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b) + 1:
        return None
    for i in range(len(words_a)):
        test = words_a[:i] + words_a[i + 1:]
        if test == words_b:
            return {
                "error_type": "word_repetition",
                "word_position": i,
                "wrong_word": words_a[i],
                "correct_word": ""
            }
    return None


def missing_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_b) != len(words_a) + 1:
        return None
    for i in range(len(words_b)):
        test = words_b[:i] + words_b[i + 1:]
        if test == words_a:
            return {
                "error_type": "missing_word",
                "word_position": i,
                "wrong_word": "",
                "correct_word": words_b[i]
            }
    return None


def swap_word_error(a: str, b: str):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b):
        return None
    differences = [
        i
        for i in range(len(words_a))
        if words_a[i] != words_b[i]
    ]
    if len(differences) != 2:
        return None
    i, j = differences
    if (
        words_a[i] == words_b[j]
        and words_a[j] == words_b[i]
    ):
        return {
            "error_type": "word_order",
            "word_positions": [i, j],
            "wrong_words": [words_a[i], words_a[j]],
            "correct_words": [words_b[i], words_b[j]]
        }
    return None


def wrong_punctuation_error(text_a, text_b):
    if not text_a or not text_b:
        return None
    punctuation = ".!?"
    if len(text_a) == len(text_b):
        if (
            text_a[:-1] == text_b[:-1]
            and text_a[-1] in punctuation
            and text_b[-1] in punctuation
            and text_a[-1] != text_b[-1]
        ):
            return {
                "error_type": "wrong_punctuation",
                "position": len(text_a) - 1,
                "wrong_char": text_a[-1],
                "correct_char": text_b[-1]
            }
    return None


def missing_punctuation_error(text_a, text_b):
    if not text_a or not text_b:
        return None
    if (
        text_b[-1] in PUNCTUATION
        and text_a[-1] not in PUNCTUATION
        and text_a == text_b[:-1]
    ):
        return {
            "error_type": "missing_punctuation",
            "position": len(text_a),
            "wrong_char": "",
            "correct_char": text_b[-1]
        }
    return None


def extra_punctuation_error(text_a, text_b):
    if len(text_a) != len(text_b) + 1:
        return None
    for i in range(len(text_a)):
        if text_a[:i] + text_a[i + 1:] == text_b:
            if text_a[i] in PUNCTUATION:
                return {
                    "error_type": "extra_punctuation",
                    "position": i,
                    "wrong_char": text_a[i],
                    "correct_char": ""
                }
    return None


def duplicate_punctuation_error(text_a, text_b):
    if len(text_a) != len(text_b) + 1:
        return None
    for i in range(len(text_a) - 1):
        if (
            text_a[i] == text_a[i + 1]
            and text_a[i] in PUNCTUATION
        ):
            test = text_a[:i] + text_a[i + 1:]
            if test == text_b:
                return {
                    "error_type": "duplicate_punctuation",
                    "position": i,
                    "wrong_char": text_a[i],
                    "correct_char": ""
                }
    return None


def spelling_error(a, b):
    dist, ops = damerau_levenshtein_trace(a, b)
    if dist != 1:
        return None
    position = 0
    for op in ops:
        if op[0] == "SUBSTITUTE":
            if op[1].lower() != op[2].lower():
                return {
                    "error_type": "spelling_error",
                    "position": position,
                    "wrong_char": op[1],
                    "correct_char": op[2]
                }
            position += 1
        elif op[0] != "INSERT":
            position += 1
    return None


def word_order_error(a, b):
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b):
        return None
    if words_a == words_b:
        return None
    if sorted(words_a) == sorted(words_b):
        positions = [
            i
            for i in range(len(words_a))
            if words_a[i] != words_b[i]
        ]
        return {
            "error_type": "word_order",
            "word_positions": positions,
            "scope": "sentence"
        }
    return None


def word_replacement_error(text_a, text_b):
    words_a = text_a.split()
    words_b = text_b.split()
    if len(words_a) != len(words_b):
        return None
    differences = [
        i
        for i, (x, y) in enumerate(zip(words_a, words_b))
        if x != y
    ]
    if len(differences) == 1:
        i = differences[0]

        return {
            "error_type": "wrong_word",
            "word_position": i,
            "wrong_word": words_a[i],
            "correct_word": words_b[i]
        }
    return None


def word_insertion_error(text_a, text_b):
    words_a = text_a.split()
    words_b = text_b.split()
    if len(words_b) != len(words_a) + 1:
        return None
    for i in range(len(words_b)):
        test = words_b[:i] + words_b[i + 1:]
        if test == words_a:
            return {
                "error_type": "missing_word",
                "word_position": i,
                "wrong_word": "",
                "correct_word": words_b[i]
            }
    return None


def invalid_character_error(text_a, text_b):
    invalid_chars = set(
        "!@#$%^&*()_+=[]{}|;:'\\\",<>/?`~"
    )
    positions = [
        i
        for i, char in enumerate(text_a)
        if char in invalid_chars
    ]
    if positions:
        return {
            "error_type": "invalid_character",
            "positions": positions,
            "wrong_chars": [text_a[i] for i in positions]
        }
    return None


def wrong_emoji_error(text_a, text_b):
    emoji_pattern = r'[\U0001F300-\U0001FAFF]'
    emojis_a = list(re.finditer(emoji_pattern, text_a))
    emojis_b = list(re.finditer(emoji_pattern, text_b))
    if not emojis_a or not emojis_b:
        return None
    chars_a = [m.group() for m in emojis_a]
    chars_b = [m.group() for m in emojis_b]
    if len(chars_a) == len(chars_b) and chars_a != chars_b:
        positions = [
            i
            for i, (x, y) in enumerate(zip(chars_a, chars_b))
            if x != y
        ]
        return {
            "error_type": "wrong_emoji",
            "emoji_positions": positions,
            "wrong_emoji": chars_a,
            "correct_emoji": chars_b
        }
    return None


def missing_emoji_error(text_a, text_b):
    emoji_pattern = EMOIJI_LIST
    emojis_a = list(re.finditer(emoji_pattern, text_a))
    emojis_b = list(re.finditer(emoji_pattern, text_b))
    if len(emojis_b) != len(emojis_a) + 1:
        return None
    chars_a = [m.group() for m in emojis_a]
    chars_b = [m.group() for m in emojis_b]
    for i in range(len(chars_b)):
        test = chars_b[:i] + chars_b[i + 1:]
        if test == chars_a:
            return {
                "error_type": "missing_emoji",
                "emoji_position": i,
                "correct_emoji": chars_b[i]
            }
    return None


def duplicate_emoji_error(text_a, text_b):
    emoji_pattern = EMOIJI_LIST
    emojis_a = list(re.finditer(emoji_pattern, text_a))
    emojis_b = list(re.finditer(emoji_pattern, text_b))
    if len(emojis_a) != len(emojis_b) + 1:
        return None
    chars_a = [m.group() for m in emojis_a]
    chars_b = [m.group() for m in emojis_b]
    for i in range(len(chars_a)):
        test = chars_a[:i] + chars_a[i + 1:]
        if test == chars_b:
            return {
                "error_type": "duplicate_emoji",
                "emoji_position": i,
                "wrong_emoji": chars_a[i]
            }
    return None


def teencode_error(text_a, text_b):
    for teencode, replacement in TEENCODE_REPLACEMENTS.items():
        if teencode in text_a and replacement in text_b:
            position = text_a.find(teencode)
            return {
                "error_type": "teencode",
                "position": position,
                "wrong_text": teencode,
                "correct_text": replacement
            }
    return None


def abbreviation_error(text_a, text_b):
    for abbreviation, replacement in ABBREVIATION_REPLACEMENTS.items():
        if abbreviation in text_a and replacement in text_b:
            position = text_a.find(abbreviation)
            return {
                "error_type": "abbreviation",
                "position": position,
                "wrong_text": abbreviation,
                "correct_text": replacement
            }
    return None


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
    if extra_english:
        positions = []
        for word in extra_english:
            match = re.search(
                rf'\b{re.escape(word)}\b',
                text_a
            )
            if match:
                positions.append({
                    "start": match.start(),
                    "end": match.end() - 1,
                    "text": word
                })
        return {
            "error_type": "code_switching",
            "positions": positions,
            "scope": "sentence"
        }
    return None


def unicode_error(a, b):
    normalized_a = unicodedata.normalize("NFC", a)
    if a != normalized_a:
        positions = [
            i
            for i, (x, y) in enumerate(zip(a, normalized_a))
            if x != y
        ]
        return {
            "error_type": "unicode_error",
            "positions": positions,
            "scope": "text"
        }
    return None


def leading_trailing_space_error(text_a, text_b):
    if text_a.strip() == text_b and text_a != text_b:
        positions = []
        if text_a.startswith(" "):
            positions.append(0)
        if text_a.endswith(" "):
            positions.append(len(text_a) - 1)
        return {
            "error_type": "leading_trailing_space",
            "positions": positions,
            "scope": "text"
        }
    return None


def url_error(a: str, b: str):
    invalid_pattern = r'https?//[^\s]+'
    valid_pattern = r'https?://[^\s]+'
    invalid_urls = list(re.finditer(invalid_pattern, a))
    valid_urls = re.findall(valid_pattern, b)
    if invalid_urls and valid_urls:
        match = invalid_urls[0]
        return {
            "error_type": "url_error",
            "position": match.start(),
            "end_position": match.end() - 1,
            "wrong_url": match.group(),
            "correct_url": valid_urls[0]
        }
    return None


def email_error(a: str, b: str):
    invalid_pattern = r'\b[\w.-]+@[\w.-]+\b'
    valid_pattern = r'\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b'
    invalid_emails = list(re.finditer(invalid_pattern, a))
    valid_emails = re.findall(valid_pattern, b)
    if invalid_emails and valid_emails:
        match = invalid_emails[0]
        return {
            "error_type": "email_error",
            "position": match.start(),
            "end_position": match.end() - 1,
            "wrong_email": match.group(),
            "correct_email": valid_emails[0]
        }
    return None

def classify_character_errors(wrong_text, correct_text):
    errors = []
    errors.append(missing_character_error(wrong_text, correct_text))
    errors.append(extra_character_error(wrong_text, correct_text))
    errors.append(duplicate_character_error(wrong_text, correct_text))
    errors.append(swap_character_error(wrong_text, correct_text))
    errors.append(wrong_capitalization_error(wrong_text, correct_text))
    errors.append(wrong_lowercase_error(wrong_text, correct_text))
    errors.append(spelling_error(wrong_text, correct_text))
    errors.append(invalid_character_error(wrong_text, correct_text))
    errors.append(unicode_error(wrong_text, correct_text))
    return [e for e in errors if e is not None]

def classify_word_errors(wrong_text, correct_text):
    errors = []
    errors.append(duplicate_word_error(wrong_text, correct_text))
    errors.append(missing_word_error(wrong_text, correct_text))
    errors.append(swap_word_error(wrong_text, correct_text))
    errors.append(word_order_error(wrong_text, correct_text))
    errors.append(word_replacement_error(wrong_text, correct_text))
    errors.append(word_insertion_error(wrong_text, correct_text))
    errors.append(teencode_error(wrong_text, correct_text))
    errors.append(abbreviation_error(wrong_text, correct_text))
    return [e for e in errors if e is not None]

def classify_text_errors(wrong_text, correct_text):
    errors = []
    errors.append(wrong_punctuation_error(wrong_text, correct_text))
    errors.append(missing_punctuation_error(wrong_text, correct_text))
    errors.append(extra_punctuation_error(wrong_text, correct_text))
    errors.append(duplicate_punctuation_error(wrong_text, correct_text))
    errors.append(wrong_emoji_error(wrong_text, correct_text))
    errors.append(missing_emoji_error(wrong_text, correct_text))
    errors.append(duplicate_emoji_error(wrong_text, correct_text))
    errors.append(code_switching_error(wrong_text, correct_text))
    errors.append(leading_trailing_space_error(wrong_text, correct_text))
    errors.append(url_error(wrong_text, correct_text))
    errors.append(email_error(wrong_text, correct_text))
    return [e for e in errors if e is not None]

    

        
