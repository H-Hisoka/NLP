# NLP
#Đề tài 1: Hệ thống phát hiện và sửa lỗi chính tả tiếng Việt

1. Giới thiệu:

-Xây dựng hệ thống phát hiện và sửa các lỗi chính tả phổ biến trong văn bản tiếng Việt, bao gồm:

Thiếu hoặc sai dấu thanh.

Nhầm ký tự.

Thừa hoặc thiếu ký tự.

Đảo vị trí ký tự.

Nhầm phụ âm đầu hoặc vần.

Viết sai do kiểu gõ Telex hoặc VNI.

2. Sản phẩm ứng dụng 
-Ứng dụng cho phép:
-Nhập hoặc tải lên một đoạn văn.
-Đánh dấu các vị trí nghi ngờ sai.
-Đưa ra tối thiểu ba phương án sửa.
-Cho phép người dùng chấp nhận hoặc từ chối từng phương án.
-Hiển thị câu sau khi sửa.


3. Chỉ số đánh giá
-Precision, recall và F1 cho phát hiện lỗi.
-Top-1 và Top-3 accuracy cho đề xuất sửa.
-Character Error Rate.
-Word Error Rate.
-Sentence exact-match accuracy.


4. Phân tích bắt buộc

Phân loại ít nhất 30 lỗi của mô hình theo các nhóm:
-Sai dấu.
-Sai ký tự.
-Lỗi tách từ.
-Từ riêng.
-Từ chuyên ngành.
-Real-word error.
-Mô hình sửa nhầm từ vốn đã đúng.



