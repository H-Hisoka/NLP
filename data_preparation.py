import pandas as pd
import os
import random

def tao_loi(text, loai_loi):

    if not isinstance(text, str):
        return text

    # 1. Thiếu ký tự
    if loai_loi == "missing_character":
        if len(text) > 1:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]

    # 2. Thừa ký tự
    elif loai_loi == "extra_character":
        if len(text) > 0:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos] + text[pos:]

    # 3. Lặp ký tự
    elif loai_loi == "duplicate_character":
        if len(text) > 0:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos] + text[pos + 1:]

    # 4. Đảo vị trí ký tự
    elif loai_loi == "swap_character":
        chars = list(text)

        if len(chars) >= 2:
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        return "".join(chars)

    # 5. Viết hoa sai
    elif loai_loi == "wrong_capitalization":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = words[pos].upper()

        return " ".join(words)

    # 6. Viết thường
    elif loai_loi == "wrong_lowercase":
        return text.lower()

    # 7. Thiếu khoảng trắng
    elif loai_loi == "missing_space":
        words = text.split()

        if len(words) >= 2:
            pos = random.randint(0, len(words) - 2)
            words[pos] = words[pos] + words[pos + 1]
            del words[pos + 1]

        return " ".join(words)

    # 8. Thừa khoảng trắng
    elif loai_loi == "extra_space":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = "  " + words[pos]

        return " ".join(words)

    # 9. Lặp từ
    elif loai_loi == "duplicate_word":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words.insert(pos, words[pos])

        return " ".join(words)

    # 10. Thiếu từ
    elif loai_loi == "missing_word":
        words = text.split()

        if len(words) > 1:
            pos = random.randint(0, len(words) - 1)
            del words[pos]

        return " ".join(words)

    # 11. Đảo vị trí từ
    elif loai_loi == "swap_word":
        words = text.split()

        if len(words) >= 2:
            pos = random.randint(0, len(words) - 2)
            words[pos], words[pos + 1] = words[pos + 1], words[pos]

        return " ".join(words)

    # 12. Sai dấu câu
    elif loai_loi == "wrong_punctuation":
        if text.endswith("."):
            return text[:-1] + "!"
        elif text.endswith("!"):
            return text[:-1] + "?"
        else:
            return text + "!"

    # 13. Thiếu dấu câu
    elif loai_loi == "missing_punctuation":
        return text.replace(".", "").replace(",", "")

    # 14. Thừa dấu câu
    elif loai_loi == "extra_punctuation":
        return text + "!"

    # 15. Lặp dấu câu
    elif loai_loi == "duplicate_punctuation":
        return text + "..."

    # 16. Sai chính tả
    elif loai_loi == "spelling_error":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            word = words[pos]

            if len(word) >= 2:
                chars = list(word)
                i = random.randint(0, len(chars) - 1)

                # Thay bằng một ký tự khác
                chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")

                words[pos] = "".join(chars)

        return " ".join(words)

    # 17. Đảo toàn bộ thứ tự từ
    elif loai_loi == "word_order":
        words = text.split()

        if len(words) >= 2:
            random.shuffle(words)

        return " ".join(words)

    # 18. Thay thế từ
    elif loai_loi == "word_replacement":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = "abc"

        return " ".join(words)

    # 19. Chèn từ
    elif loai_loi == "word_insertion":
        words = text.split()

        pos = random.randint(0, len(words))
        words.insert(pos, "abc")

        return " ".join(words)

    # 20. Ký tự không hợp lệ
    elif loai_loi == "invalid_character":
        return text + " @#$"

    # 21. Emoji sai
    elif loai_loi == "wrong_emoji":
        return text + " 😂"

    # 22. Thiếu emoji
    elif loai_loi == "missing_emoji":
        return (
            text.replace("😊", "")
                .replace("😂", "")
                .replace("👍", "")
                .replace("❤️", "")
        )

    # 23. Lặp emoji
    elif loai_loi == "duplicate_emoji":
        return text + " 😊😊"

    # 24. Teencode
    elif loai_loi == "teencode":
        replacements = {
            "không": "ko",
            "mình": "mk",
            "bạn": "bn",
            "được": "dc",
            "gì": "j",
            "vậy": "z",
            "rồi": "r",
            "biết": "bít"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # 25. Viết tắt
    elif loai_loi == "abbreviation":
        replacements = {
            "không": "ko",
            "và": "&",
            "được": "đc",
            "với": "vs"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # 26. Code switching
    elif loai_loi == "code_switching":
        return text + " very good"

    # 27. Lỗi Unicode
    elif loai_loi == "unicode_error":
        return (
            text.replace("đ", "d")
                .replace("Đ", "D")
        )

    # 28. Khoảng trắng đầu/cuối
    elif loai_loi == "leading_trailing_space":
        return "   " + text + "   "

    # 29. Lỗi URL
    elif loai_loi == "url_error":
        return text + " https//example.com"

    # 30. Lỗi Email
    elif loai_loi == "email_error":
        return text + " example@gmail"

    return text


DANH_SACH_LOI = [
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

# đọc dulieu.csv tạo file mới

def tao_file_loi():

    # Đường dẫn đến dulieu.csv
    duong_dan = os.path.join(
        os.path.dirname(__file__),
        "dulieu.csv"
    )

    # Đọc file CSV
    df = pd.read_csv(
        duong_dan,
        encoding="utf-8-sig"
    )

    # Kiểm tra cột id
    if "id" not in df.columns:
        raise ValueError(
            "File dulieu.csv phải có cột 'id'"
        )

    # Kiểm tra cột correct_text
    if "correct_text" not in df.columns:
        raise ValueError(
            "File dulieu.csv phải có cột 'correct_text'"
        )

    # Danh sách dữ liệu lỗi
    data = []

    # Duyệt từng câu
    for _, row in df.iterrows():

        sentence_id = row["id"]
        correct_text = row["correct_text"]

        # Tạo 30 loại lỗi
        for loai_loi in DANH_SACH_LOI:

            error_text = tao_loi(
                correct_text,
                loai_loi
            )

            data.append({
                "id": sentence_id,
                "correct_text": correct_text,
                "error_text": error_text,
                "error_type": loai_loi
            })

    # Tạo DataFrame mới
    error_df = pd.DataFrame(data)

    # Xuất file CSV
    error_df.to_csv(
        "dulieu_loi.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Đã tạo file: dulieu_loi.csv")
    print("Số câu gốc:", len(df))
    print("Số dòng lỗi:", len(error_df))
    print(
        "Số loại lỗi:",
        error_df["error_type"].nunique()
    )

tao_file_loi()