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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.preprocessing import normalize_text
from models.train_language_model import NGramLanguageModel
from models.improved_model import ImproveSpellChecker
from correction.candidate_generation import generate_candidates_fast

DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")
REAL_WORD_THRESHOLD = 8.0
NUM_CANDIDATES = 3
SENTENCE_DELIMITERS = ".!?;\n"
FONT = ("Segoe UI", 11)


#  Phần xử lý (không liên quan giao diện)
def load_model():
    """Huấn luyện bigram trên cột 'right' của dataset.csv, trả về ImproveSpellChecker."""
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").dropna(subset=["right"])
    correct_sentences = [normalize_text(s) for s in df["right"]]
    lm = NGramLanguageModel()
    lm.train([c for c in correct_sentences if c])
    return ImproveSpellChecker(lm)


def split_sentences(text):
    """Tách đoạn văn thành các câu. Mỗi câu là danh sách (bắt đầu, kết thúc, từ)."""
    sentences, current_sentence, last_position = [], [], 0
    for m in re.finditer(r"[^\W\d_]+", text):
        gap = text[last_position:m.start()]
        if current_sentence and any(c in SENTENCE_DELIMITERS for c in gap):
            sentences.append(current_sentence)
            current_sentence = []
        current_sentence.append((m.start(), m.end(), m.group()))
        last_position = m.end()
    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def match_case(original, new_word):
    """Giữ kiểu viết hoa của từ gốc cho từ mới."""
    if len(original) > 1 and original.isupper():
        return new_word.upper()
    if original[0].isupper():
        return new_word.capitalize()
    return new_word


def add_candidates(checker, word, previous_word, next_word, existing_candidates):
    """Nếu chưa đủ NUM_CANDIDATES thì lấy thêm ứng viên cách tối đa 2 ký tự."""
    lm = checker.lm
    candidates = generate_candidates_fast(
        word, lm.vocab, lm.unigram_counts, max_distance=2, top_n=20
    )
    if not candidates:
        return existing_candidates
    for cand, _ in checker.score(word, candidates, [], previous_word, next_word):
        if len(existing_candidates) >= NUM_CANDIDATES:
            break
        if cand != word and not cand.startswith("<") and cand not in existing_candidates:
            existing_candidates.append(cand)
    return existing_candidates


def analyze_text(checker, text):
    """Trả về danh sách các từ nghi sai (theo thứ tự xuất hiện), mỗi mục gồm:
    bat_dau, ket_thuc, tu, phuong_an (tối đa 3 từ gợi ý)."""
    results = []
    for sentence in split_sentences(text):
        words = [t[2] for t in sentence]
        for i, (start, end, word) in enumerate(sentence):
            suggestions = checker.suggest_word_candidates( words, i, NUM_CANDIDATES, REAL_WORD_THRESHOLD)
            if suggestions == [word]:
                continue                          # không nghi sai
            lowercase_word = word.lower()
            candidate_list = []
            for candidate in suggestions:                       # bỏ từ gốc và token đặc biệt
                if candidate != lowercase_word and not candidate.startswith("<") and candidate not in candidate_list:
                    candidate_list.append(candidate)
            if not candidate_list:
                continue
            previous_word = words[i - 1].lower() if i > 0 else "<s>"
            next_word = words[i + 1].lower() if i < len(words) - 1 else "</s>"
            candidate_list = add_candidates(checker, lowercase_word, previous_word, next_word, candidate_list)
            results.append({
                "bat_dau": start,
                "ket_thuc": end,
                "tu": word,
                "phuong_an": [match_case(word, c) for c in candidate_list],
            })
    return results


def apply_corrections(text, results, choices):
    """choices[i] = 0: giữ nguyên (từ chối); 1..3: chấp nhận phương án thứ n."""
    segments, end = [], 0
    for item, choice in zip(results, choices):
        segments.append(text[end:item["bat_dau"]])
        segments.append(item["phuong_an"][choice - 1] if choice else item["tu"])
        end = item["ket_thuc"]
    segments.append(text[end:])
    return "".join(segments)


# Giao diện
class SpellCheckerApp:
    def __init__(self, root, checker):
        self.root = root
        self.checker = checker
        self.text = ""
        self.results = []
        self.selection_vars = []

        root.title("Sửa lỗi chính tả tiếng Việt")
        root.geometry("1150x720")
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(root, padding=8)
        left_frame.grid(row=0, column=0, sticky="nsew")
        right_frame = ttk.Frame(root, padding=8)
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Cột trái: nhập liệu và đoạn văn có đánh dấu
        ttk.Label(left_frame, text="1. Nhập hoặc tải lên đoạn văn").pack(anchor="w")
        self.input_text = tk.Text(left_frame, height=9, wrap="word", font=FONT)
        self.input_text.pack(fill="x")
        button_row = ttk.Frame(left_frame)
        button_row.pack(fill="x", pady=4)
        ttk.Button(button_row, text="Tải file .txt", command=self.load_file).pack(side="left")
        ttk.Button(button_row, text="Kiểm tra chính tả", command=self.check_spelling).pack(side="left", padx=4)
        ttk.Button(button_row, text="Xóa", command=self.clear_text).pack(side="left")

        ttk.Label(left_frame, text="2. Vị trí nghi sai (vàng = nghi sai, xanh = đã sửa)").pack(
            anchor="w", pady=(8, 0))
        self.highlight_text = tk.Text(left_frame, height=12, wrap="word", font=FONT, state="disabled")
        self.highlight_text.pack(fill="both", expand=True)
        self.highlight_text.tag_configure("nghi", background="#fff3a3")
        self.highlight_text.tag_configure("da_sua", background="#b7efc5")

        # Cột phải: phương án sửa và kết quả
        ttk.Label(right_frame, text="3. Chọn phương án để chấp nhận (hoặc 'Giữ nguyên' để từ chối)").pack(anchor="w")
        frame = ttk.Frame(right_frame)
        frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.candidate_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.candidate_frame, anchor="nw")
        self.candidate_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

        button_row_2 = ttk.Frame(right_frame)
        button_row_2.pack(fill="x", pady=4)
        ttk.Button(button_row_2, text="Chấp nhận gợi ý 1 cho tất cả",
                   command=self.accept_all).pack(side="left")
        ttk.Button(button_row_2, text="Từ chối tất cả",
                   command=self.reject_all).pack(side="left", padx=4)

        ttk.Label(right_frame, text="4. Đoạn văn sau khi sửa").pack(anchor="w", pady=(8, 0))
        self.output_text = tk.Text(right_frame, height=10, wrap="word", font=FONT, state="disabled")
        self.output_text.pack(fill="both", expand=True)
        ttk.Button(right_frame, text="Lưu kết quả ra file...", command=self.save_file).pack(
            anchor="e", pady=4)

        self.status_label = ttk.Label(root, text="Sẵn sàng", anchor="w")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6))

    # -- tiện ích
    def set_text(self, o, content):
        o.config(state="normal")
        o.delete("1.0", "end")
        o.insert("1.0", content)
        o.config(state="disabled")

    def update_status(self, content):
        self.status_label.config(text=content)
        self.root.update_idletasks()

    # -- các nút
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file văn bản", filetypes=[("Văn bản", "*.txt"), ("Tất cả", "*.*")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as error:
            messagebox.showerror("Lỗi", f"Không đọc được file (cần mã hóa UTF-8):\n{error}")
            return
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", content)

    def clear_text(self):
        self.input_text.delete("1.0", "end")
        self.text, self.results, self.selection_vars = "", [], []
        for w in self.candidate_frame.winfo_children():
            w.destroy()
        self.set_text(self.highlight_text, "")
        self.set_text(self.output_text, "")
        self.update_status("Sẵn sàng")

    def check_spelling(self):
        text = unicodedata.normalize("NFC", self.input_text.get("1.0", "end-1c"))
        if not text.strip():
            messagebox.showinfo("Thông báo", "Hãy nhập hoặc tải lên một đoạn văn.")
            return
        self.update_status("Đang phân tích...")
        self.text = text
        self.results = analyze_text(self.checker, text)
        self.set_text(self.highlight_text, text)
        self.build_candidate_list()
        self.update_display()

    def build_candidate_list(self):
        for w in self.candidate_frame.winfo_children():
            w.destroy()
        self.selection_vars = []
        for item in self.results:
            variable = tk.IntVar(value=0)
            self.selection_vars.append(variable)
            frame = ttk.LabelFrame(self.candidate_frame, text=f"“{item['tu']}”")
            frame.pack(fill="x", padx=4, pady=3)
            ttk.Radiobutton(frame, text="Giữ nguyên (từ chối)", variable=variable,
                            value=0, command=self.update_display).pack(anchor="w")
            for n, cand in enumerate(item["phuong_an"], start=1):
                ttk.Radiobutton(frame, text=f"{n}. {cand}", variable=variable,
                                value=n, command=self.update_display).pack(anchor="w")

    def update_display(self):
        selections = [b.get() for b in self.selection_vars]
        for item, choice in zip(self.results, selections):
            first = f"1.0+{item['bat_dau']}c"
            last_end = f"1.0+{item['ket_thuc']}c"
            self.highlight_text.tag_remove("nghi", first, last_end)
            self.highlight_text.tag_remove("da_sua", first, last_end)
            self.highlight_text.tag_add("da_sua" if choice else "nghi", first, last_end)
        self.set_text(self.output_text, apply_corrections(self.text, self.results, selections))
        if self.results:
            accepted_count = sum(1 for c in selections if c)
            self.update_status(
                f"Tìm thấy {len(self.results)} từ nghi sai, đã chấp nhận {accepted_count}.")
        else:
            self.update_status("Không phát hiện từ nghi sai.")

    def accept_all(self):
        for b in self.selection_vars:
            b.set(1)
        self.update_display()

    def reject_all(self):
        for b in self.selection_vars:
            b.set(0)
        self.update_display()

    def save_file(self):
        if not self.text:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Văn bản", "*.txt")])
        if not file_path:
            return
        selections = [b.get() for b in self.selection_vars]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(apply_corrections(self.text, self.results, selections))
        self.update_status(f"Đã lưu: {file_path}")


def main():
    checker = load_model()
    root = tk.Tk()
    SpellCheckerApp(root, checker)
    root.mainloop()


if __name__ == "__main__":
    main()