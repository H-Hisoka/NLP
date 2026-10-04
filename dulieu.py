import csv
import random

# 1. Danh sách từ vựng và thành phần cấu tạo câu
subjects = [
    "Tôi", "Bạn", "Anh ấy", "Cô ấy", "Chúng tôi", "Mẹ tôi", "Bố tôi", "Em trai tôi", "Chị gái tôi",
    "Người bạn thân của tôi", "Đồng nghiệp tôi", "Thầy giáo", "Cô giáo", "Con mèo nhà tôi", "Chú chó nhỏ",
    "Mọi người", "Học sinh", "Sinh viên", "Nhóm chúng tôi", "Cộng đồng"
]

verbs_actions = [
    "đang học lập trình Python", "thích đi du lịch Đà Lạt", "thường đi dạo công viên", "đang nấu một bữa ăn ngon",
    "thích uống cà phê sữa đá", "đang chuẩn bị cho kỳ thi", "vừa mua một cuốn sách mới", "đang nghe một bản nhạc hay",
    "muốn học thêm ngoại ngữ", "thích ngắm mưa qua cửa sổ", "đang tập thể dục buổi sáng", "rất thích ăn bánh mì",
    "đang lên kế hoạch cho chuyến đi", "vừa hoàn thành công việc", "thích xem phim vào cuối tuần",
    "rất tự hào về thành quả này", "đang dọn dẹp lại căn phòng", "thích chụp ảnh thiên nhiên",
    "vừa gặp lại một người bạn cũ", "đang học cách làm bánh"
]

adverbials = [
    "vào buổi sáng", "mỗi ngày", "khi trời đổ mưa", "vào cuối tuần", "sau giờ làm việc",
    "cùng với gia đình", "trong thời gian rảnh", "với tinh thần thoải mái", "ở một quán nhỏ",
    "dưới ánh nắng nhẹ", "vào một ngày đẹp trời", "trên con đường quen thuộc", "với sự kiên trì"
]

adjectives = ["rất vui vẻ", "vô cùng hạnh phúc", "hơi mệt mỏi", "rất hào hứng", "thấy bình yên", "đầy năng lượng"]
times = ["buổi sáng hôm nay", "chiều hôm nay", "tối qua", "cuối tuần trước", "mùa hè năm nay"]

# 2. Sinh 10,000 câu chuẩn không trùng lặp
sentences_pool = []
seen = set()

random.seed(42)
while len(sentences_pool) < 10000:
    sub = random.choice(subjects)
    act = random.choice(verbs_actions)
    adv = random.choice(adverbials)
    adj = random.choice(adjectives)
    tm = random.choice(times)
    
    t_type = random.randint(1, 5)
    if t_type == 1:
        s = f"{sub} {act} {adv}."
    elif t_type == 2:
        s = f"{sub} cảm thấy {adj} khi {act}."
    elif t_type == 3:
        s = f"Hôm nay {sub} {act}."
    elif t_type == 4:
        s = f"Vào {tm}, {sub} {act}."
    else:
        s = f"{sub} rất thích {act} {adv}."
        
    if s not in seen:
        seen.add(s)
        sentences_pool.append(s)

# 3. Tạo dữ liệu chỉ gồm id và correct_text
rows = [[i, correct] for i, correct in enumerate(sentences_pool, 1)]

# 4. Xuất ra file CSV
csv_filename = "dulieu.csv"
with open(csv_filename, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "correct_text"])
    writer.writerows(rows)

print(f"Đã xuất thành công file {csv_filename}!")