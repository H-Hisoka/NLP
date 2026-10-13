"""
app.py - Ứng dụng sửa lỗi chính tả tiếng Việt (giao diện Tkinter)

Chạy: mở terminal tại thư mục project/ rồi gõ  python app.py

Cách dùng: nhập hoặc tải lên đoạn văn -> "Kiểm tra chính tả" -> xem các từ nghi sai
được tô vàng -> với mỗi từ chọn 1 trong 3 phương án để chấp nhận (hoặc "Giữ nguyên"
để từ chối) -> đoạn văn sau khi sửa hiện ở ô cuối.
"""
import os
import re
import sys
import unicodedata
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

THU_MUC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THU_MUC)

from data.preprocessing import normalize_text
from models.train_language_model import NGramLanguageModel
from models.improved_model import ImproveSpellChecker
from correction.candidate_generation import generate_candidates_fast

FILE_DU_LIEU = os.path.join(THU_MUC, "dataset.csv")
NGUONG_REAL_WORD = 8.0
SO_PHUONG_AN = 3
DAU_NGAT_CAU = ".!?;\n"
FONT = ("Segoe UI", 11)


#  Phần xử lý (không liên quan giao diện)
def tai_mo_hinh():
    """Huấn luyện bigram trên cột 'right' của dataset.csv, trả về ImproveSpellChecker."""
    df = pd.read_csv(FILE_DU_LIEU, encoding="utf-8-sig").dropna(subset=["right"])
    cau_dung = [normalize_text(s) for s in df["right"]]
    lm = NGramLanguageModel()
    lm.train([c for c in cau_dung if c])
    return ImproveSpellChecker(lm)


def tach_cau(van_ban):
    """Tách đoạn văn thành các câu. Mỗi câu là danh sách (bắt đầu, kết thúc, từ)."""
    cac_cau, cau_hien_tai, vi_tri_cuoi = [], [], 0
    for m in re.finditer(r"[^\W\d_]+", van_ban):
        khoang = van_ban[vi_tri_cuoi:m.start()]
        if cau_hien_tai and any(c in DAU_NGAT_CAU for c in khoang):
            cac_cau.append(cau_hien_tai)
            cau_hien_tai = []
        cau_hien_tai.append((m.start(), m.end(), m.group()))
        vi_tri_cuoi = m.end()
    if cau_hien_tai:
        cac_cau.append(cau_hien_tai)
    return cac_cau


def khop_hoa_thuong(goc, moi):
    """Giữ kiểu viết hoa của từ gốc cho từ mới."""
    if len(goc) > 1 and goc.isupper():
        return moi.upper()
    if goc[0].isupper():
        return moi.capitalize()
    return moi


def bo_sung_phuong_an(checker, tu, truoc, sau, hien_co):
    """Nếu chưa đủ SO_PHUONG_AN thì lấy thêm ứng viên cách tối đa 2 ký tự."""
    lm = checker.lm
    ung_vien = generate_candidates_fast(
        tu, lm.vocab, lm.unigram_counts, max_distance=2, top_n=20
    )
    if not ung_vien:
        return hien_co
    for cand, _ in checker.score(tu, ung_vien, [], truoc, sau):
        if len(hien_co) >= SO_PHUONG_AN:
            break
        if cand != tu and not cand.startswith("<") and cand not in hien_co:
            hien_co.append(cand)
    return hien_co


def phan_tich(checker, van_ban):
    """Trả về danh sách các từ nghi sai (theo thứ tự xuất hiện), mỗi mục gồm:
    bat_dau, ket_thuc, tu, phuong_an (tối đa 3 từ gợi ý)."""
    ket_qua = []
    for cau in tach_cau(van_ban):
        cac_tu = [t[2] for t in cau]
        for i, (bat_dau, ket_thuc, tu) in enumerate(cau):
            goi_y = checker.goi_y_cho_tu(cac_tu, i, SO_PHUONG_AN, NGUONG_REAL_WORD)
            if goi_y == [tu]:
                continue                          # không nghi sai
            thuong = tu.lower()
            ds = []
            for c in goi_y:                       # bỏ từ gốc và token đặc biệt
                if c != thuong and not c.startswith("<") and c not in ds:
                    ds.append(c)
            if not ds:
                continue
            truoc = cac_tu[i - 1].lower() if i > 0 else "<s>"
            sau = cac_tu[i + 1].lower() if i < len(cac_tu) - 1 else "</s>"
            ds = bo_sung_phuong_an(checker, thuong, truoc, sau, ds)
            ket_qua.append({
                "bat_dau": bat_dau,
                "ket_thuc": ket_thuc,
                "tu": tu,
                "phuong_an": [khop_hoa_thuong(tu, c) for c in ds],
            })
    return ket_qua


def ap_dung(van_ban, ket_qua, lua_chon):
    """lua_chon[i] = 0: giữ nguyên (từ chối); 1..3: chấp nhận phương án thứ n."""
    phan, cuoi = [], 0
    for muc, chon in zip(ket_qua, lua_chon):
        phan.append(van_ban[cuoi:muc["bat_dau"]])
        phan.append(muc["phuong_an"][chon - 1] if chon else muc["tu"])
        cuoi = muc["ket_thuc"]
    phan.append(van_ban[cuoi:])
    return "".join(phan)


# Giao diện
class UngDung:
    def __init__(self, root, checker):
        self.root = root
        self.checker = checker
        self.van_ban = ""
        self.ket_qua = []
        self.bien_chon = []

        root.title("Sửa lỗi chính tả tiếng Việt")
        root.geometry("1150x720")
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        trai = ttk.Frame(root, padding=8)
        trai.grid(row=0, column=0, sticky="nsew")
        phai = ttk.Frame(root, padding=8)
        phai.grid(row=0, column=1, sticky="nsew")

        # Cột trái: nhập liệu và đoạn văn có đánh dấu
        ttk.Label(trai, text="1. Nhập hoặc tải lên đoạn văn").pack(anchor="w")
        self.o_nhap = tk.Text(trai, height=9, wrap="word", font=FONT)
        self.o_nhap.pack(fill="x")
        hang = ttk.Frame(trai)
        hang.pack(fill="x", pady=4)
        ttk.Button(hang, text="Tải file .txt", command=self.tai_file).pack(side="left")
        ttk.Button(hang, text="Kiểm tra chính tả", command=self.kiem_tra).pack(side="left", padx=4)
        ttk.Button(hang, text="Xóa", command=self.xoa).pack(side="left")

        ttk.Label(trai, text="2. Vị trí nghi sai (vàng = nghi sai, xanh = đã sửa)").pack(
            anchor="w", pady=(8, 0))
        self.o_danh_dau = tk.Text(trai, height=12, wrap="word", font=FONT, state="disabled")
        self.o_danh_dau.pack(fill="both", expand=True)
        self.o_danh_dau.tag_configure("nghi", background="#fff3a3")
        self.o_danh_dau.tag_configure("da_sua", background="#b7efc5")

        # Cột phải: phương án sửa và kết quả
        ttk.Label(phai, text="3. Chọn phương án để chấp nhận (hoặc 'Giữ nguyên' để từ chối)").pack(anchor="w")
        khung = ttk.Frame(phai)
        khung.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(khung, highlightthickness=0)
        thanh = ttk.Scrollbar(khung, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=thanh.set)
        thanh.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.trong = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.trong, anchor="nw")
        self.trong.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

        hang2 = ttk.Frame(phai)
        hang2.pack(fill="x", pady=4)
        ttk.Button(hang2, text="Chấp nhận gợi ý 1 cho tất cả",
                   command=self.chap_nhan_tat_ca).pack(side="left")
        ttk.Button(hang2, text="Từ chối tất cả",
                   command=self.tu_choi_tat_ca).pack(side="left", padx=4)

        ttk.Label(phai, text="4. Đoạn văn sau khi sửa").pack(anchor="w", pady=(8, 0))
        self.o_ket_qua = tk.Text(phai, height=10, wrap="word", font=FONT, state="disabled")
        self.o_ket_qua.pack(fill="both", expand=True)
        ttk.Button(phai, text="Lưu kết quả ra file...", command=self.luu_file).pack(
            anchor="e", pady=4)

        self.nhan_trang_thai = ttk.Label(root, text="Sẵn sàng", anchor="w")
        self.nhan_trang_thai.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6))

    # -- tiện ích
    def dat_text(self, o, noi_dung):
        o.config(state="normal")
        o.delete("1.0", "end")
        o.insert("1.0", noi_dung)
        o.config(state="disabled")

    def trang_thai(self, noi_dung):
        self.nhan_trang_thai.config(text=noi_dung)
        self.root.update_idletasks()

    # -- các nút
    def tai_file(self):
        duong_dan = filedialog.askopenfilename(
            title="Chọn file văn bản", filetypes=[("Văn bản", "*.txt"), ("Tất cả", "*.*")])
        if not duong_dan:
            return
        try:
            with open(duong_dan, "r", encoding="utf-8-sig") as f:
                noi_dung = f.read()
        except (OSError, UnicodeDecodeError) as loi:
            messagebox.showerror("Lỗi", f"Không đọc được file (cần mã hóa UTF-8):\n{loi}")
            return
        self.o_nhap.delete("1.0", "end")
        self.o_nhap.insert("1.0", noi_dung)

    def xoa(self):
        self.o_nhap.delete("1.0", "end")
        self.van_ban, self.ket_qua, self.bien_chon = "", [], []
        for w in self.trong.winfo_children():
            w.destroy()
        self.dat_text(self.o_danh_dau, "")
        self.dat_text(self.o_ket_qua, "")
        self.trang_thai("Sẵn sàng")

    def kiem_tra(self):
        van_ban = unicodedata.normalize("NFC", self.o_nhap.get("1.0", "end-1c"))
        if not van_ban.strip():
            messagebox.showinfo("Thông báo", "Hãy nhập hoặc tải lên một đoạn văn.")
            return
        self.trang_thai("Đang phân tích...")
        self.van_ban = van_ban
        self.ket_qua = phan_tich(self.checker, van_ban)
        self.dat_text(self.o_danh_dau, van_ban)
        self.xay_danh_sach_phuong_an()
        self.cap_nhat()

    def xay_danh_sach_phuong_an(self):
        for w in self.trong.winfo_children():
            w.destroy()
        self.bien_chon = []
        for muc in self.ket_qua:
            bien = tk.IntVar(value=0)
            self.bien_chon.append(bien)
            khung = ttk.LabelFrame(self.trong, text=f"“{muc['tu']}”")
            khung.pack(fill="x", padx=4, pady=3)
            ttk.Radiobutton(khung, text="Giữ nguyên (từ chối)", variable=bien,
                            value=0, command=self.cap_nhat).pack(anchor="w")
            for n, cand in enumerate(muc["phuong_an"], start=1):
                ttk.Radiobutton(khung, text=f"{n}. {cand}", variable=bien,
                                value=n, command=self.cap_nhat).pack(anchor="w")

    def cap_nhat(self):
        lua_chon = [b.get() for b in self.bien_chon]
        for muc, chon in zip(self.ket_qua, lua_chon):
            dau = f"1.0+{muc['bat_dau']}c"
            cuoi = f"1.0+{muc['ket_thuc']}c"
            self.o_danh_dau.tag_remove("nghi", dau, cuoi)
            self.o_danh_dau.tag_remove("da_sua", dau, cuoi)
            self.o_danh_dau.tag_add("da_sua" if chon else "nghi", dau, cuoi)
        self.dat_text(self.o_ket_qua, ap_dung(self.van_ban, self.ket_qua, lua_chon))
        if self.ket_qua:
            da_nhan = sum(1 for c in lua_chon if c)
            self.trang_thai(
                f"Tìm thấy {len(self.ket_qua)} từ nghi sai, đã chấp nhận {da_nhan}.")
        else:
            self.trang_thai("Không phát hiện từ nghi sai.")

    def chap_nhan_tat_ca(self):
        for b in self.bien_chon:
            b.set(1)
        self.cap_nhat()

    def tu_choi_tat_ca(self):
        for b in self.bien_chon:
            b.set(0)
        self.cap_nhat()

    def luu_file(self):
        if not self.van_ban:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu.")
            return
        duong_dan = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Văn bản", "*.txt")])
        if not duong_dan:
            return
        lua_chon = [b.get() for b in self.bien_chon]
        with open(duong_dan, "w", encoding="utf-8") as f:
            f.write(ap_dung(self.van_ban, self.ket_qua, lua_chon))
        self.trang_thai(f"Đã lưu: {duong_dan}")


def main():
    checker = tai_mo_hinh()
    root = tk.Tk()
    UngDung(root, checker)
    root.mainloop()


if __name__ == "__main__":
    main()