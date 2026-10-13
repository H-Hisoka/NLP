import unicodedata
from correction.error_detection import damerau_levenshtein_trace
from config import TONE_CONFUSIONS
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
def classify_word(original_word, predicted_word, correct_word, vocab):
      if original_word == correct_word:
          return "Sửa nhầm từ vốn đã đúng"
      if original_word in vocab:
          return "Real-word error"
      if original_word.replace(" ", "") == correct_word.replace(" ", ""):
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
   print("--- CHIA NHỎ NHÓM [Sai ký tự] ---")
   for subcategory, logs in character_error_subcategories.items():
       print(f"  - {subcategory}: {len(logs)} mẫu")

def print_error_statistics():
   print("--- THỐNG KÊ BẢNG PHÂN TÍCH LỖI CHO BÁO CÁO ---")
   for cat, logs in error_categories.items():
       print(f"* Nhóm [{cat}]: {len(logs)} mẫu lỗi")