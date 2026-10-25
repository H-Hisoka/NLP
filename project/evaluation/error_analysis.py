import unicodedata
from correction.error_detection import damerau_levenshtein_trace
from config import TONE_CONFUSIONS
from data.preprocessing import normalize_text
from correction.ranking import rank_word_candidates

error_categories = {
   "Sai dấu": [],
   "Sai ký tự": [],
   "Lỗi tách từ": [],
   "Từ riêng": [],
   "Từ chuyên ngành": [],
   "Real-word error": [],
   "Sửa nhầm từ vốn đã đúng": []
}
# Chia nhỏ nhóm "Sai ký tự" theo thao tác của damerau_levenshtein_trace
character_error_subcategories = {
   "Thiếu một ký tự": [],
   "Thiếu nhiều ký tự (viết tắt/teencode)": [],
   "Thừa ký tự": [],
   "Đảo ký tự": [],
   "Nhầm ký tự": [],
   "Nhiều lỗi cùng lúc": []
}
def remove_accents(text):
   """Bỏ hết dấu: 'hoàn' -> 'hoan', 'đúng' -> 'dung'"""
   text = text.replace("đ", "d").replace("Đ", "D")
   text = unicodedata.normalize("NFD", text)
   return "".join(c for c in text if unicodedata.category(c) != "Mn")

def get_edit_operations(original_word, correct_word):
      """Các thao tác (khác MATCH) để biến từ nhập vào thành đáp án."""
      _, ops = damerau_levenshtein_trace(original_word, correct_word)
      return [op for op in ops if op[0] != "MATCH"]

def log_error_analysis(original_word, predicted_word, correct_word, category):
   """Ghi nhận lỗi của mô hình phục vụ báo cáo đồ án"""
   if category in error_categories:
       error_categories[category].append({
           "Input": original_word,
           "Predict": predicted_word,
           "Target": correct_word
       })


def is_segmentation_error(original_word, correct_word):
    if len(original_word.split()) == len(correct_word.split()):
        return False
    a = remove_accents(original_word.replace(" ", "")).lower()
    b = remove_accents(correct_word.replace(" ", "")).lower()
    return a == b

def find_segmentation_errors(wrong_words, right_words, max_group=4):
      """Ghép từ câu sai với câu đúng theo chuỗi ký tự (bỏ dấu, bỏ cách).
      Trả về list (cụm_sai, cụm_đúng) tại các chỗ dính/tách từ."""
      results = []
      i = j = 0
      while i < len(wrong_words) and j < len(right_words):
          if wrong_words[i] == right_words[j]:
              i += 1; j += 1
              continue
          matched = False
          for total in range(3, 2 * max_group + 1):
              for di in range(1, max_group + 1):
                  dj = total - di
                  if dj < 1 or dj > max_group or (di == 1 and dj == 1):
                      continue
                  w_grp, r_grp = wrong_words[i:i + di], right_words[j:j + dj]
                  if len(w_grp) < di or len(r_grp) < dj:
                      continue
                  w_txt, r_txt = " ".join(w_grp), " ".join(r_grp)
                  if is_segmentation_error(w_txt, r_txt):
                      results.append((w_txt, r_txt))
                      i += di; j += dj
                      matched = True
                      break
              if matched:
                  break
          if not matched:
              i += 1; j += 1
      return results

def classify_word(original_word, predicted_word, correct_word, vocab):
      if original_word == correct_word:
          return "Sửa nhầm từ vốn đã đúng"
      if original_word in vocab:
          return "Real-word error"
      if is_segmentation_error(original_word, correct_word):
          return "Lỗi tách từ"
      edit_operations = get_edit_operations(original_word, correct_word)
      if edit_operations and all(
          operation_types == "SUBSTITUTE" and (a, b) in TONE_CONFUSIONS
          for operation_types, a, b in edit_operations
      ):
          return "Sai dấu"
      if correct_word not in vocab:
          return "Từ riêng"
      return "Sai ký tự"

def classify_character_error(original_word, correct_word):
   """Chia nhỏ nhóm 'Sai ký tự' dựa vào các thao tác của trace."""
   edit_operations = get_edit_operations(original_word, correct_word)
   operation_types = {t[0] for t in edit_operations}
   if operation_types == {"INSERT"}:
       if len(edit_operations) == 1:
           return "Thiếu một ký tự"
       return "Thiếu nhiều ký tự (viết tắt/teencode)"
   if operation_types == {"DELETE"}:
       return "Thừa ký tự"
   if operation_types == {"TRANSPOSE"}:
       return "Đảo ký tự"
   if operation_types == {"SUBSTITUTE"}:
       return "Nhầm ký tự"
   return "Nhiều lỗi cùng lúc"

def log_sub_error(original_word, predicted_word, correct_word, subcategory):
   if subcategory in character_error_subcategories:
       character_error_subcategories[subcategory].append({
           "Input": original_word,
           "Predict": predicted_word,
           "Target": correct_word
       })

def print_sub_error_statistics():
   print("Chia nhỏ nhóm (Sai ký tự)")
   for subcategory, logs in character_error_subcategories.items():
       print(f"  - {subcategory}: {len(logs)} mẫu")

def print_error_statistics():
   print("Bảng thống kê phân tích lỗi cho báo cáo")
   for cat, logs in error_categories.items():
       print(f"* Nhóm [{cat}]: {len(logs)} mẫu lỗi")

# Phân tích lỗi
def analyze_errors(lm, checker, test_df, threshold, num_samples_to_print=30):
    """Ghi lỗi thật của mô hình vào error_analysis, in thống kê và các mẫu."""
    errors_found = []
    for wrong, right in zip(test_df["wrong_norm"], test_df["right_norm"]):
        pred = checker.correct_sentence(wrong, threshold)
        w, g, p = wrong.split(), right.split(), pred.split()
        if not (len(w) == len(g) == len(p)):
            if len(w) != len(g):
                pred_text = " ".join(p)
                for wrong_chunk, right_chunk in find_segmentation_errors(w, g):
                    if right_chunk not in pred_text:   # model chưa sửa đúng
                        log_error_analysis(wrong_chunk, pred_text, right_chunk, "Lỗi tách từ")
                        errors_found.append((wrong_chunk, pred_text, right_chunk, "Lỗi tách từ"))
                continue
            if len(w) != len(p):
                continue
        for ww, gw, pw in zip(w, g, p):
            if pw != gw:
                category = classify_word(ww, pw, gw, lm.vocab)
                log_error_analysis(ww, pw, gw, category)
                if category == "Sai ký tự":
                    sub_category = classify_character_error(ww, gw)
                    log_sub_error(ww, pw, gw, sub_category)
                    category = f"{category} - {sub_category}"
                errors_found.append((ww, pw, gw, category))
    print("\n")
    print_error_statistics()
    print()
    print_sub_error_statistics()
    print(f"\n{num_samples_to_print} LỖI MẪU (input -> dự đoán | đáp án | nhóm)")
    for ww, pw, gw, category in errors_found[:num_samples_to_print]:
        print(f"{ww} -> {pw} | {gw} | {category}")


def show_sample_predictions(lm, checker, sentences, threshold):
    print("\nTHỬ CÂU MẪU ")
    for sentence in sentences:
        c = normalize_text(sentence)
        tokens = c.split()
        goi_y = checker.correct_sentence_top_k(c, 3, threshold)
        print(f"\nInput: {c}")
        print("Top-1: " + " ".join(g[0] for g in goi_y))
        for word, g in zip(tokens, goi_y):
            if g != [word]:
                print(f"  '{word}' nghi sai -> {g}")
        for idx, word in enumerate(tokens):
            if word not in lm.vocab:
                print("  ranking.py:", rank_word_candidates(c, idx, lm)[:3])
                break