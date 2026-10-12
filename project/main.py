import hashlib
import os
import pickle
import sys

import pandas as pd

# Để in tiếng Việt được cả khi chạy bằng nút Run / Code Runner trên Windows
sys.stdout.reconfigure(encoding="utf-8")

THU_MUC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THU_MUC)

from data.preprocessing import normalize_text
from models.train_language_model import NGramLanguageModel
from evaluation.baseline import BaselineSpellChecker
from models.improved_model import ImproveSpellChecker
from evaluation.evaluation import evaluate_system
from evaluation.error_analysis import (
    tu_phan_loai, log_error_analysis, thong_ke_loi,
    phan_loai_sai_ky_tu, log_nhom_con, thong_ke_nhom_con,
)
from correction.ranking import rank_word_candidates

FILE_DU_LIEU = os.path.join(THU_MUC, "dataset.csv")      # cột: stt, wrong, right
FILE_CACHE = os.path.join(THU_MUC, "cache_ung_vien.pkl")   # xóa file này nếu sửa code sinh ứng viên
TI_LE_TRAIN = 0.8
SEED = 42
SO_MAU_TEST = 1000          # đặt None để chạy toàn bộ tập test
NGUONG_REAL_WORD = 8.0      # từ đúng chỉ bị sửa khi ứng viên hơn ít nhất ngần này điểm
CAC_NGUONG_THU = [2.0, 5.0, 8.0, 12.0, 16.0]


# Dữ liệu 
def doc_du_lieu(duong_dan=FILE_DU_LIEU):
    df = pd.read_csv(duong_dan, encoding="utf-8-sig")
    df = df.dropna(subset=["wrong", "right"]).copy()
    df["wrong_norm"] = df["wrong"].apply(normalize_text)
    df["right_norm"] = df["right"].apply(normalize_text)
    return df[(df["wrong_norm"] != "") & (df["right_norm"] != "")]


def chia_train_test(df, ti_le=TI_LE_TRAIN, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cat = int(len(df) * ti_le)
    return df.iloc[:cat], df.iloc[cat:]


#  Lưu / nạp cache
def dau_van_tay(checker):
    """Cache chỉ dùng lại được khi từ vựng và confusion set không đổi."""
    noi_dung = "\n".join(sorted(checker.lm.vocab))
    noi_dung += repr(sorted((k, sorted(v)) for k, v in checker.confusion_dict.items()))
    return hashlib.md5(noi_dung.encode("utf-8")).hexdigest()


def nap_cache(checker):
    if not os.path.exists(FILE_CACHE):
        return
    with open(FILE_CACHE, "rb") as f:
        du_lieu = pickle.load(f)
    if du_lieu["van_tay"] == dau_van_tay(checker):
        checker.cache = du_lieu["cache"]
        print(f"Đã nạp cache ứng viên ({len(checker.cache)} mục)")
    else:
        print("Cache cũ không khớp từ vựng / confusion set, bỏ qua.")


def luu_cache(checker):
    with open(FILE_CACHE, "wb") as f:
        pickle.dump({"van_tay": dau_van_tay(checker), "cache": checker.cache}, f)
    print(f"\nĐã lưu cache ứng viên ({len(checker.cache)} mục) vào {FILE_CACHE}")


#Đánh giá
def dem_tp_fp_fn(golds, errors, preds):
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


def in_ket_qua(ten, kq):
    print(f"\n=== {ten} ===")
    for k, v in kq.items():
        print(f"{k:>15}: {v:.4f}")


def danh_gia_baseline(checker, test_df):
    """BaselineSpellChecker chỉ trả về top-1 nên top-3 = top-1."""
    errors = test_df["wrong_norm"].tolist()
    golds = test_df["right_norm"].tolist()
    preds = [checker.correct_sentence(s) for s in errors]
    TP, FP, FN = dem_tp_fp_fn(golds, errors, preds)
    kq = evaluate_system(preds, golds, preds, TP, FP, FN)
    in_ket_qua("BASELINE (edit distance + bigram)", kq)
    return kq


def danh_gia_improved(ten, checker, test_df, nguong, in_kq=True):
    errors = test_df["wrong_norm"].tolist()
    golds = test_df["right_norm"].tolist()
    ds_goi_y = [checker.correct_sentence_top_k(s, 3, nguong) for s in errors]
    top1 = [" ".join(g[0] for g in goi_y) for goi_y in ds_goi_y]
    top3 = [" ".join("|".join(g) for g in goi_y) for goi_y in ds_goi_y]
    TP, FP, FN = dem_tp_fp_fn(golds, errors, top1)
    kq = evaluate_system(top1, golds, top3, TP, FP, FN)
    if in_kq:
        in_ket_qua(ten, kq)
    return kq


# Phân tích lỗi
def phan_tich_loi(lm, checker, test_df, nguong, so_mau_in=30):
    """Ghi lỗi thật của mô hình vào error_analysis, in thống kê và các mẫu."""
    cac_loi = []
    for wrong, right in zip(test_df["wrong_norm"], test_df["right_norm"]):
        pred = checker.correct_sentence(wrong, nguong)
        w, g, p = wrong.split(), right.split(), pred.split()
        if not (len(w) == len(g) == len(p)):
            continue
        for ww, gw, pw in zip(w, g, p):
            if pw != gw:
                nhom = tu_phan_loai(ww, pw, gw, lm.vocab)
                log_error_analysis(ww, pw, gw, nhom)
                if nhom == "Sai ký tự":
                    nhom_con = phan_loai_sai_ky_tu(ww, gw)
                    log_nhom_con(ww, pw, gw, nhom_con)
                    nhom = f"{nhom} - {nhom_con}"
                cac_loi.append((ww, pw, gw, nhom))
    print("\n")
    thong_ke_loi()
    print()
    thong_ke_nhom_con()
    print(f"\n--- {so_mau_in} LỖI MẪU (input -> dự đoán | đáp án | nhóm) ---")
    for ww, pw, gw, nhom in cac_loi[:so_mau_in]:
        print(f"{ww} -> {pw} | {gw} | {nhom}")


def thu_cau_mau(lm, checker, cac_cau, nguong):
    print("\n=== THỬ CÂU MẪU ===")
    for cau in cac_cau:
        c = normalize_text(cau)
        tokens = c.split()
        goi_y = checker.correct_sentence_top_k(c, 3, nguong)
        print(f"\nInput: {c}")
        print("Top-1: " + " ".join(g[0] for g in goi_y))
        for tu, g in zip(tokens, goi_y):
            if g != [tu]:
                print(f"  '{tu}' nghi sai -> {g}")
        for idx, tu in enumerate(tokens):
            if tu not in lm.vocab:
                print("  ranking.py:", rank_word_candidates(c, idx, lm)[:3])
                break


def main():
    df = doc_du_lieu()
    train_df, test_df = chia_train_test(df)
    if SO_MAU_TEST:
        test_df = test_df.head(SO_MAU_TEST)
    print(f"Tổng: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")

    lm = NGramLanguageModel()
    lm.train(train_df["right_norm"].tolist())
    print(f"Kích thước từ vựng: {len(lm.vocab)}")

    baseline = BaselineSpellChecker(lm)
    improved = ImproveSpellChecker(lm)
    nap_cache(improved)

    danh_gia_baseline(baseline, test_df)
    danh_gia_improved("IMPROVED (confusion set, chưa real-word)", improved, test_df, None)
    danh_gia_improved(f"IMPROVED + REAL-WORD (ngưỡng {NGUONG_REAL_WORD})",
                      improved, test_df, NGUONG_REAL_WORD)

    print("\n=== THỬ NGƯỠNG REAL-WORD ===")
    print(f"{'ngưỡng':>7} {'precision':>10} {'recall':>8} {'f1':>8} {'top1':>8} {'top3':>8} {'exact':>8}")
    for ng in CAC_NGUONG_THU:
        kq = danh_gia_improved("", improved, test_df, ng, in_kq=False)
        print(f"{ng:>7} {kq['precision']:>10.4f} {kq['recall']:>8.4f} {kq['f1']:>8.4f} "
              f"{kq['top1_accuracy']:>8.4f} {kq['top3_accuracy']:>8.4f} {kq['exact_match']:>8.4f}")

    phan_tich_loi(lm, improved, test_df, NGUONG_REAL_WORD)

    cau_mau = ["sinh viên cần hoàn thành bai tập", "tôi muốn ăn cơm răng"]
    cau_mau += test_df["wrong"].head(2).tolist()
    thu_cau_mau(lm, improved, cau_mau, NGUONG_REAL_WORD)
    luu_cache(improved)


if __name__ == "__main__":
    main()