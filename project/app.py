import os
import re
import sys
import unicodedata
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from data.preprocessing import normalize_text, mask_special_spans, find_invalid_format_spans
from models.train_language_model import NGramLanguageModel
from models.improved_model import ImproveSpellChecker
from correction.candidate_generation import generate_candidates_fast

DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")
REAL_WORD_THRESHOLD = 8.0
NUM_CANDIDATES = 3
SENTENCE_DELIMITERS = ".!?;\n"
FONT = ("Segoe UI", 11)
MANUAL = -1                     # giá trị lựa chọn "Nhập lại" (0 = giữ nguyên, 1..3 = phương án gợi ý)
FORMAT_LABELS = {"email": "email", "url": "liên kết", "date": "ngày", "time": "giờ"}


#  Load model
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
    masked = mask_special_spans(text)      # email, link, số điện thoại... không bị tách thành từ
    for m in re.finditer(r"[^\W_]+", masked):
        gap = masked[last_position:m.start()]
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
        decoded = None
        if checker.is_unaccented(" ".join(words)):      # câu không dấu: chọn cả chuỗi từ hợp lý nhất
            decoded = checker.decode_sentence(" ".join(words))
        for i, (start, end, word) in enumerate(sentence):
            suggestions = checker.suggest_word_candidates( words, i, NUM_CANDIDATES, REAL_WORD_THRESHOLD)
            if decoded is not None and (decoded[i] != word.lower() or suggestions != [word]):
                suggestions = checker.merge_suggestions(word, decoded[i], suggestions, NUM_CANDIDATES)
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
            if len(candidate_list) < NUM_CANDIDATES:      # từ gõ Telex/VNI không có từ nào gần: lấy từ gần với gợi ý đầu
                candidate_list = add_candidates(checker, candidate_list[0], previous_word, next_word, candidate_list)
            results.append({
                "bat_dau": start,
                "ket_thuc": end,
                "tu": word,
                "phuong_an": [match_case(word, c) for c in candidate_list],
            })
    results.extend(analyze_formats(text))          # email / link / ngày / giờ sai định dạng
    results.sort(key=lambda item: item["bat_dau"])
    return results


def analyze_formats(text):
    """Các đoạn email, link, ngày, giờ viết sai định dạng (cùng cấu trúc với kết quả của analyze_text).
    Có thêm khóa 'loai'; phuong_an có thể rỗng nếu không đoán được cách sửa (người dùng tự nhập lại)."""
    return [{
        "bat_dau": item["start"],
        "ket_thuc": item["end"],
        "tu": item["text"],
        "phuong_an": item["suggestions"],
        "loai": item["kind"],
    } for item in find_invalid_format_spans(text)]


def replacement_for(item, choice, manual_text=""):
    """Nội dung thay cho một vị trí nghi sai: 0 = giữ nguyên, 1..3 = phương án, MANUAL = người dùng tự nhập."""
    if choice == MANUAL:
        return manual_text.strip() or item["tu"]
    return item["phuong_an"][choice - 1] if choice else item["tu"]


def apply_corrections(text, results, choices, manual_texts=None):
    """choices[i] = 0: giữ nguyên (từ chối); 1..3: chấp nhận phương án thứ n; MANUAL: dùng chuỗi người dùng nhập."""
    manual_texts = manual_texts or [""] * len(results)
    segments, end = [], 0
    for item, choice, manual_text in zip(results, choices, manual_texts):
        segments.append(text[end:item["bat_dau"]])
        segments.append(replacement_for(item, choice, manual_text))
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
        self.manual_vars = []

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
        self.text, self.results, self.selection_vars, self.manual_vars = "", [], [], []
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
        self.selection_vars, self.manual_vars = [], []
        for item in self.results:
            variable = tk.IntVar(value=0)
            manual = tk.StringVar(value=item["tu"])
            self.selection_vars.append(variable)
            self.manual_vars.append(manual)
            title = f"“{item['tu']}”"
            if item.get("loai"):
                title += f" – {FORMAT_LABELS[item['loai']]} sai định dạng"
            frame = ttk.LabelFrame(self.candidate_frame, text=title)
            frame.pack(fill="x", padx=4, pady=3)
            ttk.Radiobutton(frame, text="Giữ nguyên (từ chối)", variable=variable,
                            value=0, command=self.update_display).pack(anchor="w")
            for n, cand in enumerate(item["phuong_an"], start=1):
                ttk.Radiobutton(frame, text=f"{n}. {cand}", variable=variable,
                                value=n, command=self.update_display).pack(anchor="w")
            if item.get("loai"):                      # lỗi định dạng: cho người dùng tự nhập lại
                row = ttk.Frame(frame)
                row.pack(fill="x")
                ttk.Radiobutton(row, text="Nhập lại:", variable=variable, value=MANUAL,
                                command=self.update_display).pack(side="left")
                ttk.Entry(row, textvariable=manual, width=34).pack(side="left", padx=4)
                manual.trace_add("write", lambda *_, v=variable: (v.set(MANUAL), self.update_display()))

    def update_display(self):
        selections = [b.get() for b in self.selection_vars]
        manual_texts = [m.get() for m in self.manual_vars]
        accepted_count = 0
        for item, choice, manual_text in zip(self.results, selections, manual_texts):
            first = f"1.0+{item['bat_dau']}c"
            last_end = f"1.0+{item['ket_thuc']}c"
            fixed = replacement_for(item, choice, manual_text) != item["tu"]
            accepted_count += fixed
            self.highlight_text.tag_remove("nghi", first, last_end)
            self.highlight_text.tag_remove("da_sua", first, last_end)
            self.highlight_text.tag_add("da_sua" if fixed else "nghi", first, last_end)
        self.set_text(self.output_text,
                      apply_corrections(self.text, self.results, selections, manual_texts))
        if self.results:
            format_count = sum(1 for item in self.results if item.get("loai"))
            self.update_status(
                f"Tìm thấy {len(self.results)} vị trí nghi sai (trong đó {format_count} lỗi định dạng), "
                f"đã sửa {accepted_count}.")
        else:
            self.update_status("Không phát hiện từ nghi sai.")

    def accept_all(self):
        for item, b in zip(self.results, self.selection_vars):
            b.set(1 if item["phuong_an"] else 0)      # lỗi định dạng không có gợi ý thì giữ nguyên
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
        manual_texts = [m.get() for m in self.manual_vars]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(apply_corrections(self.text, self.results, selections, manual_texts))
        self.update_status(f"Đã lưu: {file_path}")


def main():
    checker = load_model()
    root = tk.Tk()
    SpellCheckerApp(root, checker)
    root.mainloop()


if __name__ == "__main__":
    main()