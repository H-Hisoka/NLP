
#Đường dẫn
DATA_DIR= "data"
RAW_DATA_DIR= "data/raw"
PROCESSED_DATA_DIR= "data/processed"
GENERATED_DATA_DIR= "data/generated"

#Tên file
RAW_DATA_FILE="data/raw/dataset.csv"
PROCESSED_DATA_FILE="data/processed/processed_data.csv"
GENERATED_DATA_FILE="data/generated/error_dataset.csv"

ERROR_TYPES = [
    # Lỗi ký tự
    "missing_character",
    "extra_character",
    "wrong_character",
    "character_transposition",
    "character_repetition",
    "character_deletion",

    # Lỗi phụ âm
    "wrong_initial_consonant",
    "missing_initial_consonant",
    "extra_initial_consonant",
    "confused_initial_consonant",

    # Lỗi vần
    "wrong_final_consonant",
    "missing_final_consonant",
    "extra_final_consonant",
    "wrong_vowel",
    "confused_vowel",

    # Lỗi dấu thanh
    "missing_tone",
    "wrong_tone",
    "extra_tone",

    # Lỗi từ
    "missing_word",
    "extra_word",
    "wrong_word",
    "word_repetition",
    "word_order",

    # Lỗi chính tả
    "phonetic_spelling",
    "abbreviation",
    "homophone_confusion",
    "compound_word_error",

    # Lỗi ngữ cảnh
    "contextually_wrong_word",
    "grammatically_wrong_word",
]

ERRORS_PER_SENTENCE = 3
RANDOM_SEED = 42

TOP_K = 3
NGRAM_ORDER = 3
MAX_CANDIDATES = 20

USE_LANGUAGE_MODEL = True
LM_WEIGHT = 0.5
EDIT_DISTANCE_WEIGHT = 0.5