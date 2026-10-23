import os
ERROR_TYPES = [
    "missing_character",
    "extra_character",
    "duplicate_character",
    "swap_character",
    "wrong_capitalization",
    "wrong_lowercase",
    "missing_space",
    "extra_space",
    "duplicate_word",
    "missing_word",
    "swap_word",
    "wrong_punctuation",
    "missing_punctuation",
    "extra_punctuation",
    "duplicate_punctuation",
    "spelling_error",
    "word_order",
    "word_replacement",
    "word_insertion",
    "invalid_character",
    "wrong_emoji",
    "missing_emoji",
    "duplicate_emoji",
    "teencode",
    "abbreviation",
    "code_switching",
    "unicode_error",
    "leading_trailing_space",
    "url_error",
    "email_error"
]

EMOJI_LIST = [
    "😂", "😢", "😡", "❤️", "😱"
]

TEENCODE_REPLACEMENTS = {
    "không": "ko",
    "mình": "mk",
    "bạn": "bn",
    "được": "dc",
    "gì": "j",
    "vậy": "z",
    "rồi": "r",
    "biết": "bít"
}

ABBREVIATION_REPLACEMENTS = {
    "không": "ko",
    "và": "&",
    "được": "đc",
    "với": "vs"
}   

PUNCTUATION=[
    
]

VOWELS = {
    "a": "aàáảãạ", "ă": "ăằắẳẵặ", "â": "âầấẩẫậ",
    "e": "eèéẻẽẹ", "ê": "êềếểễệ", "i": "iìíỉĩị",
    "o": "oòóỏõọ", "ô": "ôồốổỗộ", "ơ": "ơờớởỡợ",
    "u": "uùúủũụ", "ư": "ưừứửữự", "y": "yỳýỷỹỵ",
}

# Các cặp ký tự chỉ khác nhau về dấu thanh (kể cả không dấu), tính cả hai chiều
TONE_CONFUSIONS = {
    (x, y)
    for group in VOWELS.values()
    for x in group
    for y in group
    if x != y
}
TONE_CONFUSIONS |= {("d", "đ"), ("đ", "d")}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFUSION_SET = os.path.join(BASE_DIR, "confusion_set.txt")

DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")      # cột: stt, wrong, right
CACHE_FILE = os.path.join(BASE_DIR, "candidate_cache.pkl")   # xóa file này nếu sửa code sinh ứng viên
TRAIN_RATIO = 0.8
SEED = 42
TEST_SAMPLE_SIZE = None        # đặt None để chạy toàn bộ tập test
REAL_WORD_THRESHOLD = 8.0      # từ đúng chỉ bị sửa khi ứng viên hơn ít nhất ngần này điểm
THRESHOLD_VALUES = [2.0, 5.0, 8.0, 12.0, 16.0]
# Giải mã cả câu cho câu không dấu (improved_model.py)
DECODE_ACCENT_COST = 1.0   # chi phí khôi phục dấu (bỏ dấu thì ra đúng từ đang xét)
DECODE_EDIT_COST = 4.0     # chi phí mỗi thao tác sửa ký tự khác
DECODE_OOV_COST = 4.0      # phạt khi giữ nguyên một từ không có trong từ vựng
# Các đoạn đặc biệt cần bỏ qua khi kiểm tra chính tả (tên -> regex).
SPECIAL_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "url": r"(?:https?://|www\.)[^\s]+",
    "phone": r"(?<![\w.])(?:(?:\+84|84|0)[\s.-]?\(?\d{1,3}\)?|\(0\d{1,3}\))(?:[\s.-]?\d{2,4}){2,3}(?!\w)",
    "date": r"(?<![\w/])(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2})(?![\w/])",
    "time": r"(?<![\w:])\d{1,2}(?:[:hH]\d{2}|[hH])(?![\w:])",
    "social": r"(?<!\w)[#@]\w+",
}