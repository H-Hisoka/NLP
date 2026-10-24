# NLP

# Đề tài 1: Hệ thống phát hiện và sửa lỗi chính tả tiếng Việt

## 1. Giới thiệu

Xây dựng hệ thống phát hiện và sửa các lỗi chính tả phổ biến trong văn bản tiếng Việt, bao gồm:

- Thiếu hoặc sai dấu thanh.
- Nhầm lẫn ký tự.
- Thừa hoặc thiếu ký tự.
- Đảo vị trí các ký tự liền nhau.
- Nhầm phụ âm đầu hoặc vần.
- Viết sai do sử dụng kiểu gõ Telex hoặc VNI.
- Sai định dạng email, ngày, tháng, năm, .... (dùng RE)

## 2. Tính năng

Ứng dụng cung cấp các chức năng sau:

- Nhập văn bản trực tiếp hoặc tải lên văn bản cần kiểm tra.
- Phát hiện và tô màu các vị trí nghi ngờ có lỗi chính tả.
- Đề xuất tối thiểu ba phương án sửa lỗi cho mỗi vị trí phù hợp.
- Cho phép người dùng chấp nhận hoặc từ chối từng phương án sửa lỗi.
- Hỗ trợ xử lý các định dạng đặc biệt như email, ngày, tháng và năm,.... cho phép người dùng chấp nhận gợi ý hoặc tự nhập nội dung chính xác.
- Hiển thị văn bản sau khi hoàn tất chỉnh sửa.

## 3. Cài đặt

### Yêu cầu

- Python 3.x
- pip

### Bước 1: Clone repository

```bash
git clone https://github.com/H-Hisoka/NLP.git
cd NLP
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

### Bước 3: Cài đặt thư viện

```bash
python -m pip install pandas
```

### Bước 4: Khởi chạy chương trình

Chạy file khởi chạy chính của dự án sau khi đã cài đặt đầy đủ các thư viện cần thiết.

*Lưu ý: Cần bổ sung tên file và lệnh chạy cụ thể khi xác định được file khởi chạy chính.*

## 4. Dataset

Bộ dữ liệu được xây dựng từ các nguồn:

- Dữ liệu lỗi chính tả thực tế thu thập từ mạng xã hội.
- Bộ dữ liệu VSEC (Vietnamese Spell Correction).
- Dữ liệu lỗi tổng hợp bằng AI nhằm mở rộng các dạng lỗi phục vụ huấn luyện và đánh giá.

Dữ liệu được sử dụng để phát triển, kiểm thử và đánh giá khả năng phát hiện, đề xuất sửa lỗi chính tả của hệ thống.

## 5. Đánh giá

Hệ thống được đánh giá thông qua các chỉ số:

- **Precision, Recall và F1-score:** Đánh giá hiệu quả phát hiện lỗi chính tả.
- **Top-1 Accuracy:** Tỷ lệ trường hợp phương án sửa đúng nằm ở vị trí đầu tiên.
- **Top-3 Accuracy:** Tỷ lệ trường hợp có ít nhất một phương án sửa đúng trong ba gợi ý đầu tiên.
- **Character Error Rate (CER):** Đo mức độ sai khác giữa văn bản dự đoán và văn bản chuẩn ở cấp ký tự.
- **Word Error Rate (WER):** Đo mức độ sai khác giữa văn bản dự đoán và văn bản chuẩn ở cấp từ.
- **Sentence Exact-Match Accuracy:** Tỷ lệ câu được sửa hoàn toàn trùng khớp với câu chuẩn.

## 6. Phân tích lỗi

Thực hiện phân tích tối thiểu 30 trường hợp lỗi của mô hình, phân loại theo các nhóm:

- Sai dấu thanh.
- Sai ký tự.
- Lỗi tách từ.
- Từ riêng.
- Từ chuyên ngành.
- Lỗi Real-word (từ viết đúng về mặt hình thức nhưng không phù hợp với ngữ cảnh).
- Mô hình sửa nhầm từ vốn đã đúng.
- Từ lỗi định dạng Vni/Telex
- Sửa các lỗi định dạng Regex (email, ngày, tháng, năm,....)

Mục tiêu là xác định nguyên nhân gây lỗi, đánh giá hạn chế của hệ thống và đề xuất hướng cải thiện.

## 7. Thành viên

| STT | Thành viên | Công việc phụ trách | Tiến độ |
|:---:|---|---|---|
| 1 | Khang | Lập trình, soạn thảo báo cáo Word | 100% |
| 2 | Hải | Lập trình, soạn thảo báo cáo Word | 100% |
| 3 | Lộc | Lập trình, soạn thảo báo cáo Word | 100% |
| 4 | An | Thiết kế slide thuyết trình | 100% |
| 5 | Lâm | Soạn thảo báo cáo Word | 100% |

## 8. Link video demo
Link: https://drive.google.com/file/d/1cTxadY1iMZKP_BFxJHlEOcucT-1lhPcM/view?usp=sharing

