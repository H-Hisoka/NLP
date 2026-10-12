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
nhom_con_sai_ky_tu = {
   "Thiếu một ký tự": [],
   "Thiếu nhiều ký tự (viết tắt/teencode)": [],
   "Thừa ký tự": [],
   "Đảo ký tự": [],
   "Nhầm ký tự": [],
   "Nhiều lỗi cùng lúc": []
}
def bo_dau(text):
   """Bỏ hết dấu: 'hoàn' -> 'hoan', 'đúng' -> 'dung'"""
   text = text.replace("đ", "d").replace("Đ", "D")
   text = unicodedata.normalize("NFD", text)
   return "".join(c for c in text if unicodedata.category(c) != "Mn")
def lay_thao_tac(original_word, correct_word):
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
def tu_phan_loai(original_word, predicted_word, correct_word, vocab):
      if original_word == correct_word:
          return "Sửa nhầm từ vốn đã đúng"
      if original_word in vocab:
          return "Real-word error"
      if original_word.replace(" ", "") == correct_word.replace(" ", ""):
          return "Lỗi tách từ"
      thao_tac = lay_thao_tac(original_word, correct_word)
      if thao_tac and all(
          loai == "SUBSTITUTE" and (a, b) in TONE_CONFUSIONS
          for loai, a, b in thao_tac
      ):
          return "Sai dấu"
      if correct_word not in vocab:
          return "Từ riêng"
      return "Sai ký tự"
def phan_loai_sai_ky_tu(original_word, correct_word):
   """Chia nhỏ nhóm 'Sai ký tự' dựa vào các thao tác của trace."""
   thao_tac = lay_thao_tac(original_word, correct_word)
   loai = {t[0] for t in thao_tac}
   if loai == {"INSERT"}:
       if len(thao_tac) == 1:
           return "Thiếu một ký tự"
       return "Thiếu nhiều ký tự (viết tắt/teencode)"
   if loai == {"DELETE"}:
       return "Thừa ký tự"
   if loai == {"TRANSPOSE"}:
       return "Đảo ký tự"
   if loai == {"SUBSTITUTE"}:
       return "Nhầm ký tự"
   return "Nhiều lỗi cùng lúc"

def log_nhom_con(original_word, predicted_word, correct_word, nhom_con):
   if nhom_con in nhom_con_sai_ky_tu:
       nhom_con_sai_ky_tu[nhom_con].append({
           "Input": original_word,
           "Predict": predicted_word,
           "Target": correct_word
       })

def thong_ke_nhom_con():
   print("--- CHIA NHỎ NHÓM [Sai ký tự] ---")
   for nhom, logs in nhom_con_sai_ky_tu.items():
       print(f"  - {nhom}: {len(logs)} mẫu")

def thong_ke_loi():
   print("--- THỐNG KÊ BẢNG PHÂN TÍCH LỖI CHO BÁO CÁO ---")
   for cat, logs in error_categories.items():
       print(f"* Nhóm [{cat}]: {len(logs)} mẫu lỗi")