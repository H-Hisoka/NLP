import random

class VietnameseNoiseGenerator:
   def __init__(self):
       # Bảng quy đổi Telex/VNI & Ngữ âm đồng âm
       self.telex_map = {'â': 'aa', 'ê': 'ee', 'ô': 'oo', 'đ': 'dd', 'ư': 'uw', 'ơ': 'ow', 'ă': 'aw'}
       self.accent_map = {'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
                         'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
                         'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o'}
       self.confusion_consonants = {'ch': 'tr', 'tr': 'ch', 'd': 'gi', 'gi': 'd', 's': 'x', 'x': 's', 'c': 'k', 'k': 'c'}
       self.real_word_pairs = [("rang", "răng"), ("sinh", "xinh"), ("bài", "bai"), ("bái", "bài")]

   def add_noise(self, sentence: str, noise_prob=0.2) -> str:
       words = sentence.split()
       noisy_words = []
       
       for word in words:
           if random.random() < noise_prob:
               err_type = random.choice(["accent", "telex", "swap", "consonant", "delete_char"])
               
               if err_type == "accent": # Sai/thiếu dấu
                   word_clean = ''.join([self.accent_map.get(c, c) for c in word])
                   noisy_words.append(word_clean)
               elif err_type == "telex": # Lỗi gõ telex thừa
                   for char, telex in self.telex_map.items():
                       if char in word:
                           word = word.replace(char, telex)
                           break
                   noisy_words.append(word)
               elif err_type == "swap" and len(word) > 3: # Đảo vị trí ký tự
                   idx = random.randint(1, len(word) - 2)
                   word_list = list(word)
                   word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
                   noisy_words.append(''.join(word_list))
               elif err_type == "consonant": # Nhầm phụ âm đầu
                   for k, v in self.confusion_consonants.items():
                       if word.startswith(k):
                           word = v + word[len(k):]
                           break
                   noisy_words.append(word)
               else:
                   noisy_words.append(word)
           else:
               noisy_words.append(word)
               
       return ' '.join(noisy_words)

# Tạo dữ liệu thử nghiệm mẫu
generator = VietnameseNoiseGenerator()
raw_sentences = [
   "Sinh viên cần hoàn thành bài tập về nhà đúng thời hạn.",
   "Tôi muốn ăn cơm rang vào buổi trưa hôm nay.",
   "Hệ thống xủ lý ngôn ngữ tự nhiên hoạt động rất tốt.",
   "Trường đại học bách khoa hà nội là một trường đại học lớn.",
   "Chúng ta nên bảo vệ môi trường xung quanh chúng ta."
]

dataset = []
for sent in raw_sentences:
   noisy = generator.add_noise(sent, noise_prob=0.3)
   dataset.append({"correct": sent, "noisy": noisy})

print("--- MẪU DỮ LIỆU TẠO LỖI ---")
for d in dataset[:3]:
   print("Gốc  :", d["correct"])
   print("Lỗi  :", d["noisy"])
   print("-" * 50)