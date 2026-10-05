# NLP
Dự án nhóm (Đề tài 1)
cách chạy giao diện: 
làm theo các bước này, trên Windows PowerShell hoặc terminal trong VS Code.

1. Mở terminal đúng thư mục đồ án
ví dụ:
cd C:\Users\bhhbu\Downloads\doanNLP\NLP

Thư mục này phải có app.py, suggest.py, improved_model.py, config.py, baseline.py và dulieu.csv (hoặc data/processed/processed_data.csv).

2. Cài Streamlit (chỉ làm một lần)

pip install streamlit

3. Chạy giao diện

streamlit run app.py

Nếu báo không nhận lệnh streamlit, dùng:

python -m streamlit run app.py

4. Mở trang web
Trình duyệt thường tự mở. Nếu không, bạn mở http://localhost:8501. Lần đầu có thể mất vài giây để nạp mô hình ngôn ngữ.
