import pandas as pd
import os
import random
from config import ERROR_TYPES


def generate_error(text, error_type):

    if not isinstance(text, str):
        return text

    # 1. Thiếu ký tự
    if error_type == "missing_character":
        if len(text) > 1:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]

    # 2. Thừa ký tự
    elif error_type == "extra_character":
        if len(text) > 0:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos] + text[pos:]

    # 3. Lặp ký tự
    elif error_type == "duplicate_character":
        if len(text) > 0:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos] + text[pos + 1:]

    # 4. Đảo vị trí ký tự
    elif error_type == "swap_character":
        chars = list(text)

        if len(chars) >= 2:
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        return "".join(chars)

    # 5. Viết hoa sai
    elif error_type == "wrong_capitalization":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = words[pos].upper()

        return " ".join(words)

    # 6. Viết thường
    elif error_type == "wrong_lowercase":
        return text.lower()

    # 7. Thiếu khoảng trắng
    elif error_type == "missing_space":
        words = text.split()

        if len(words) >= 2:
            pos = random.randint(0, len(words) - 2)
            words[pos] = words[pos] + words[pos + 1]
            del words[pos + 1]

        return " ".join(words)

    # 8. Thừa khoảng trắng
    elif error_type == "extra_space":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = "  " + words[pos]

        return " ".join(words)

    # 9. Lặp từ
    elif error_type == "duplicate_word":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words.insert(pos, words[pos])

        return " ".join(words)

    # 10. Thiếu từ
    elif error_type == "missing_word":
        words = text.split()

        if len(words) > 1:
            pos = random.randint(0, len(words) - 1)
            del words[pos]

        return " ".join(words)

    # 11. Đảo vị trí từ
    elif error_type == "swap_word":
        words = text.split()

        if len(words) >= 2:
            pos = random.randint(0, len(words) - 2)
            words[pos], words[pos + 1] = words[pos + 1], words[pos]

        return " ".join(words)

    # 12. Sai dấu câu
    elif error_type == "wrong_punctuation":
        if text.endswith("."):
            return text[:-1] + "!"
        elif text.endswith("!"):
            return text[:-1] + "?"
        else:
            return text + "!"

    # 13. Thiếu dấu câu
    elif error_type == "missing_punctuation":
        return text.replace(".", "").replace(",", "")

    # 14. Thừa dấu câu
    elif error_type == "extra_punctuation":
        return text + "!"

    # 15. Lặp dấu câu
    elif error_type == "duplicate_punctuation":
        return text + "..."

    # 16. Sai chính tả
    elif error_type == "spelling_error":
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
    elif error_type == "word_order":
        words = text.split()

        if len(words) >= 2:
            random.shuffle(words)

        return " ".join(words)

    # 18. Thay thế từ
    elif error_type == "word_replacement":
        words = text.split()

        if len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            words[pos] = "abc"

        return " ".join(words)

    # 19. Chèn từ
    elif error_type == "word_insertion":
        words = text.split()

        pos = random.randint(0, len(words))
        words.insert(pos, "abc")

        return " ".join(words)

    # 20. Ký tự không hợp lệ
    elif error_type == "invalid_character":
        return text + " @#$"

    # 21. Emoji sai
    elif error_type == "wrong_emoji":
        emojis = ["😂", "😢", "😡", "❤️", "😱"]

        for emoji in emojis:
            if emoji in text:
                wrong_emoji = random.choice(
                    [e for e in emojis if e != emoji]
                )
                return text.replace(emoji, wrong_emoji, 1)

    # 22. Thiếu emoji
    elif error_type == "missing_emoji":
        return (
            text.replace("😊", "")
                .replace("😂", "")
                .replace("👍", "")
                .replace("❤️", "")
        )

    # 23. Lặp emoji
    elif error_type == "duplicate_emoji":
        return text + " 😊😊"

    # 24. Teencode
    elif error_type == "teencode":
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
    elif error_type == "abbreviation":
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
    elif error_type == "code_switching":
        words = text.split()
        if len(words) >= 2:
            pos = random.randint(1, len(words) - 1)
            words.insert(pos, random.choice([
                "very good",
                "awesome",
                "cool",
                "nice",
                "okay"
            ]))
    
            return " ".join(words)
    
        return text

    # 27. Lỗi Unicode
    elif error_type == "unicode_error":
        return (
            text.replace("đ", "d")
                .replace("Đ", "D")
        )

    # 28. Khoảng trắng đầu/cuối
    elif error_type == "leading_trailing_space":
        return "   " + text + "   "

    # 29. Lỗi URL
    elif error_type == "url_error":
        return text + " https//example.com"

    # 30. Lỗi Email
    elif error_type == "email_error":
        return text + " example@gmail"

    return text

# đọc dulieu.csv tạo file mới

def generate_error_file():

    # Đường dẫn đến dataset.csv
    file_path = os.path.join(
        os.path.dirname(__file__),
        "dataset.csv"
    )

    # Đọc file CSV
    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    # Kiểm tra cột id
    if "id" not in df.columns:
        raise ValueError(
            "File dataset.csv phải có cột 'id'"
        )

    # Kiểm tra cột correct_text
    if "correct_text" not in df.columns:
        raise ValueError(
            "File dataset.csv phải có cột 'correct_text'"
        )

    # Danh sách dữ liệu lỗi
    data = []

    # Duyệt từng câu
    for _, row in df.iterrows():

        sentence_id = row["id"]
        correct_text = row["correct_text"]

        # Tạo 30 loại lỗi
        for generate_error_file in ERROR_TYPES:

            error_text = generate_error(
                correct_text,
                generate_error_file
            )

            data.append({
                "id": sentence_id,
                "correct_text": correct_text,
                "error_text": error_text,
                "error_type": generate_error_file
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

generate_error_file()