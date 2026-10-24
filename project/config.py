import os

VOWELS = {
    "a": "aàáảãạ", "ă": "ăằắẳẵặ", "â": "âầấẩẫậ",
    "e": "eèéẻẽẹ", "ê": "êềếểễệ", "i": "iìíỉĩị",
    "o": "oòóỏõọ", "ô": "ôồốổỗộ", "ơ": "ơờớởỡợ",
    "u": "uùúủũụ", "ư": "ưừứửữự", "y": "yỳýỷỹỵ",
}

# Cặp ký tự khác nhau về dấu thanh (kể cả không dấu), tính cả hai chiều
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
CACHE_FILE = os.path.join(BASE_DIR, "candidate_cache.pkl") 
TRAIN_RATIO = 0.8
SEED = 42
TEST_SAMPLE_SIZE = 1500      
REAL_WORD_THRESHOLD = 8.0      
THRESHOLD_VALUES = [2.0, 5.0, 8.0, 12.0, 16.0]

# Giải mã cả câu cho câu không dấu (improved_model.py)
DECODE_ACCENT_COST = 1.0   # chi phí khôi phục dấu
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