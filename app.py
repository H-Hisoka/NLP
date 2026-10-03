class SpellCheckerApp:
   def __init__(self, checker):
       self.checker = checker
       
   def process_text(self, text: str):
       print(f"\n[VĂN BẢN ĐẦU VÀO]: {text}")
       print("="*60)
       
       _, suggestions = self.checker.correct_pipeline(text)
       words = text.split()
       final_words = list(words)
       
       if not suggestions:
           print(" Không phát hiện lỗi chính tả nào!")
           return text
           
       print(f"Phát hiện {len(suggestions)} vị trí nghi ngờ có lỗi:\n")
       
       for idx, info in suggestions.items():
           orig_word = info["original"]
           top_3 = info["top_3"]
           
           print(f" Vị trí từ '{orig_word}' (Từ số {idx+1}):")
           print(f"   Top-3 gợi ý sửa: 1. [{top_3[0]}] | 2. [{top_3[1]}] | 3. [{top_3[2]}]")
           
           # Giả lập phản hồi của người dùng (User Accept/Reject)
           user_choice = input(f" Accept sửa '{orig_word}' -> '{top_3[0]}'? (y/n/1/2/3): ").strip().lower()
           
           if user_choice in ['y', 'yes', '1']:
               final_words[idx] = top_3[0]
               print(f"   -> Đã sửa thành: '{top_3[0]}'")
           elif user_choice == '2' and len(top_3) > 1:
               final_words[idx] = top_3[1]
               print(f"   -> Đã sửa thành: '{top_3[1]}'")
           elif user_choice == '3' and len(top_3) > 2:
               final_words[idx] = top_3[2]
               print(f"   -> Đã sửa thành: '{top_3[2]}'")
           else:
               print(f"   -> Giữ nguyên từ gốc: '{orig_word}'")
           print("-" * 40)
           
       final_sentence = ' '.join(final_words)
       print("="*60)
       print(f"[CÂU HOÀN CHỈNH SAU SỬA]: {final_sentence}\n")
       return final_sentence

# Khởi chạy App tương tác thử nghiệm
app = SpellCheckerApp(advanced_checker)

# Bạn có thể nhập câu tùy ý vào đây để test
sample_input = "Sinh vien can hoan thanh bai tap"
# Bỏ comment dòng dưới nếu muốn chạy tương tác gõ bằng bàn phím trong Terminal/Cell
# app.process_text(sample_input)