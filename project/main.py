import hashlib
import os
import pickle
import sys

import pandas as pd

# Để in tiếng Việt được cả khi chạy bằng nút Run / Code Runner trên Windows
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.preprocessing import normalize_text
from models.train_language_model import NGramLanguageModel
from evaluation.baseline import BaselineSpellChecker
from models.improved_model import ImproveSpellChecker
from evaluation.evaluation import evaluate_system
from evaluation.error_analysis import (
    classify_word, log_error_analysis, print_error_statistics,
    classify_character_error, log_sub_error, print_sub_error_statistics,
)
from correction.ranking import rank_word_candidates

DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")      # cột: stt, wrong, right
CACHE_FILE = os.path.join(BASE_DIR, "cache_ung_vien.pkl")   # xóa file này nếu sửa code sinh ứng viên
TRAIN_RATIO = 0.8
SEED = 42
TEST_SAMPLE_SIZE = None        # đặt None để chạy toàn bộ tập test
REAL_WORD_THRESHOLD = 8.0      # từ đúng chỉ bị sửa khi ứng viên hơn ít nhất ngần này điểm
THRESHOLD_VALUES = [2.0, 5.0, 8.0, 12.0, 16.0]


# Dữ liệu 
def load_data(file_path=DATA_FILE):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df.dropna(subset=["wrong", "right"]).copy()
    df["wrong_norm"] = df["wrong"].apply(normalize_text)
    df["right_norm"] = df["right"].apply(normalize_text)
    return df[(df["wrong_norm"] != "") & (df["right_norm"] != "")]


def split_train_test(df, ratio=TRAIN_RATIO, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_index = int(len(df) * ratio)
    return df.iloc[:split_index], df.iloc[split_index:]


#  Lưu / nạp cache
def generate_fingerprint(checker):
    """Cache chỉ dùng lại được khi từ vựng và confusion set không đổi."""
    content = "\n".join(sorted(checker.lm.vocab))
    content += repr(sorted((k, sorted(v)) for k, v in checker.confusion_dict.items()))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_cache(checker):
    if not os.path.exists(CACHE_FILE):
        return
    with open(CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)
    if cache_data["van_tay"] == generate_fingerprint(checker):
        checker.cache = cache_data["cache"]
        print(f"Đã nạp cache ứng viên ({len(checker.cache)} mục)")
    else:
        print("Cache cũ không khớp từ vựng / confusion set, bỏ qua.")


def save_cache(checker):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"van_tay": generate_fingerprint(checker), "cache": checker.cache}, f)
    print(f"\nĐã lưu cache ứng viên ({len(checker.cache)} mục) vào {CACHE_FILE}")


#Đánh giá
def calculate_tp_fp_fn(golds, errors, preds):
    """Đếm ở mức từ, chỉ xét các câu có cùng số từ giữa gold / error / pred."""
    TP = FP = FN = 0
    for gold, err, pred in zip(golds, errors, preds):
        g, e, p = gold.split(), err.split(), pred.split()
        if not (len(g) == len(e) == len(p)):
            continue
        for gw, ew, pw in zip(g, e, p):
            if ew != gw:
                if pw == gw:
                    TP += 1
                else:
                    FN += 1
            elif pw != gw:
                FP += 1
    return TP, FP, FN


def print_results(name, results):
    print(f"\n=== {name} ===")
    for k, v in results.items():
        print(f"{k:>15}: {v:.4f}")


def evaluate_baseline(checker, test_df):
    """BaselineSpellChecker chỉ trả về top-1 nên top-3 = top-1."""
    errors = test_df["wrong_norm"].tolist()
    golds = test_df["right_norm"].tolist()
    preds = [checker.correct_sentence(s) for s in errors]
    TP, FP, FN = calculate_tp_fp_fn(golds, errors, preds)
    results = evaluate_system(preds, golds, preds, TP, FP, FN)
    print_results("BASELINE (edit distance + bigram)", results)
    return results


def evaluate_improved(name, checker, test_df, threshold, print_results_flag=True):
    errors = test_df["wrong_norm"].tolist()
    golds = test_df["right_norm"].tolist()
    suggestions = [checker.correct_sentence_top_k(s, 3, threshold) for s in errors]
    top1 = [" ".join(g[0] for g in suggestions) for suggestions in suggestions]
    top3 = [" ".join("|".join(g) for g in suggestions) for suggestions in suggestions]
    TP, FP, FN = calculate_tp_fp_fn(golds, errors, top1)
    results = evaluate_system(top1, golds, top3, TP, FP, FN)
    if print_results_flag:
        print_results(name, results)
    return results


# Phân tích lỗi
def analyze_errors(lm, checker, test_df, threshold, num_samples_to_print=30):
    """Ghi lỗi thật của mô hình vào error_analysis, in thống kê và các mẫu."""
    errors_found = []
    for wrong, right in zip(test_df["wrong_norm"], test_df["right_norm"]):
        pred = checker.correct_sentence(wrong, threshold)
        w, g, p = wrong.split(), right.split(), pred.split()
        if not (len(w) == len(g) == len(p)):
            continue
        for ww, gw, pw in zip(w, g, p):
            if pw != gw:
                category = classify_word(ww, pw, gw, lm.vocab)
                log_error_analysis(ww, pw, gw, category)
                if category == "Sai ký tự":
                    sub_category = classify_character_error(ww, gw)
                    log_sub_error(ww, pw, gw, sub_category)
                    category = f"{category} - {sub_category}"
                errors_found.append((ww, pw, gw, category))
    print("\n")
    print_error_statistics()
    print()
    print_sub_error_statistics()
    print(f"\n--- {num_samples_to_print} LỖI MẪU (input -> dự đoán | đáp án | nhóm) ---")
    for ww, pw, gw, category in errors_found[:num_samples_to_print]:
        print(f"{ww} -> {pw} | {gw} | {category}")


def test_sample_sentences(lm, checker, sentences, threshold):
    print("\n=== THỬ CÂU MẪU ===")
    for sentence in sentences:
        c = normalize_text(sentence)
        tokens = c.split()
        goi_y = checker.correct_sentence_top_k(c, 3, threshold)
        print(f"\nInput: {c}")
        print("Top-1: " + " ".join(g[0] for g in goi_y))
        for word, g in zip(tokens, goi_y):
            if g != [word]:
                print(f"  '{word}' nghi sai -> {g}")
        for idx, word in enumerate(tokens):
            if word not in lm.vocab:
                print("  ranking.py:", rank_word_candidates(c, idx, lm)[:3])
                break


def main():
    df = load_data()
    train_df, test_df = split_train_test(df)
    if TEST_SAMPLE_SIZE:
        test_df = test_df.head(TEST_SAMPLE_SIZE)
    print(f"Tổng: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")

    lm = NGramLanguageModel()
    lm.train(train_df["right_norm"].tolist())
    print(f"Kích thước từ vựng: {len(lm.vocab)}")

    baseline = BaselineSpellChecker(lm)
    improved = ImproveSpellChecker(lm)
    load_cache(improved)

    evaluate_baseline(baseline, test_df)
    evaluate_improved("IMPROVED (confusion set, chưa real-word)", improved, test_df, None)
    evaluate_improved(f"IMPROVED + REAL-WORD (ngưỡng {REAL_WORD_THRESHOLD})",
                      improved, test_df, REAL_WORD_THRESHOLD)

    print("\n=== THỬ NGƯỠNG REAL-WORD ===")
    print(f"{'ngưỡng':>7} {'precision':>10} {'recall':>8} {'f1':>8} {'top1':>8} {'top3':>8} {'exact':>8}")
    for ng in THRESHOLD_VALUES:
        results = evaluate_improved("", improved, test_df, ng, print_results_flag=False)
        print(f"{ng:>7} {results['precision']:>10.4f} {results['recall']:>8.4f} {results['f1']:>8.4f} "
              f"{results['top1_accuracy']:>8.4f} {results['top3_accuracy']:>8.4f} {results['exact_match']:>8.4f}")

    analyze_errors(lm, improved, test_df, REAL_WORD_THRESHOLD)

    sample_sentences = ["sinh viên cần hoàn thành bai tập", "tôi muốn ăn cơm răng"]
    sample_sentences += test_df["wrong"].head(2).tolist()
    test_sample_sentences(lm, improved, sample_sentences, REAL_WORD_THRESHOLD)
    save_cache(improved)


if __name__ == "__main__":
    main()