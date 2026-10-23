"""
tune_weights.py - Tìm trọng số chấm điểm cho ImproveSpellChecker (mục SCORE_* và
REAL_WORD_THRESHOLD trong config.py).

Cách dùng (trong thư mục project/):  python tune_weights.py

Cách làm:
1. Tách một tập dev (10%) ra khỏi tập TRAIN. Tập test không bị đụng tới, nên số liệu
   test sau khi chỉnh vẫn trung thực.
2. Huấn luyện mô hình ngôn ngữ trên phần train còn lại.
3. Với từng trọng số, thử lần lượt các giá trị trong GRID, giữ giá trị nào làm F1 trên
   tập dev cao hơn (thử lại toàn bộ lưới ROUNDS vòng).
4. In ra các giá trị nên chép vào config.py.
"""
import sys

from config import (SCORE_EDIT_WEIGHT, SCORE_CONFUSION_BONUS, SCORE_FREQ_WEIGHT,
                    SCORE_LEFT_WEIGHT, SCORE_RIGHT_WEIGHT, SCORE_TONE_AWARE,
                    REAL_WORD_THRESHOLD, SEED)
from data.preprocessing import load_data, split_train_test
from models.train_language_model import NGramLanguageModel
from models.improved_model import ImproveSpellChecker
from evaluation.evaluation import calculate_tp_fp_fn

sys.stdout.reconfigure(encoding="utf-8")

DEV_FRACTION = 0.10     # tách 10% của tập train làm tập dev
DEV_SIZE = 600          # số câu dev dùng để chấm (lấy trong khối lỗi thật, stt < 10000)
ROUNDS = 2              # số vòng thử lại toàn bộ lưới

GRID = {
    "edit": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],     # không để về 0: bỏ qua hoàn toàn khoảng cách sửa
    "confusion": [0.0, 0.5, 1.5, 3.0, 4.0, 6.0, 8.0],
    "freq": [0.0, 0.15, 0.3, 0.6, 0.8, 1.0, 1.5],
    "left": [0.4, 0.55, 0.7, 0.85, 1.0, 1.4],
    "right": [0.7, 1.0, 1.3],
    "tone_aware": [False, True],
    "threshold": [4.0, 6.0, 8.0, 9.0, 10.0, 12.0],
}


def f1_on_dev(checker, params, errors, golds):
    """F1 sửa lỗi theo từ (cách tính giống evaluation.py) và exact-match trên tập dev."""
    checker.weights.update({k: v for k, v in params.items() if k != "threshold"})
    predictions = [
        " ".join(g[0] for g in checker.correct_sentence_top_k(s, 3, params["threshold"]))
        for s in errors
    ]
    TP, FP, FN = calculate_tp_fp_fn(golds, errors, predictions)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    exact = sum(p == g for p, g in zip(predictions, golds)) / len(golds)
    return f1, exact, precision, recall


def main():
    df = load_data()
    train_df, _ = split_train_test(df)                  # bỏ tập test, không dùng ở đây
    dev_df = train_df.sample(frac=DEV_FRACTION, random_state=SEED)
    small_train_df = train_df.drop(dev_df.index)
    dev_df = dev_df[dev_df["stt"] < 10000].head(DEV_SIZE)
    errors, golds = dev_df["wrong_norm"].tolist(), dev_df["right_norm"].tolist()
    print(f"Huấn luyện trên {len(small_train_df)} câu | tập dev: {len(dev_df)} câu")

    lm = NGramLanguageModel()
    lm.train(small_train_df["right_norm"].tolist())
    checker = ImproveSpellChecker(lm)

    best = {
        "edit": SCORE_EDIT_WEIGHT, "confusion": SCORE_CONFUSION_BONUS, "freq": SCORE_FREQ_WEIGHT,
        "left": SCORE_LEFT_WEIGHT, "right": SCORE_RIGHT_WEIGHT, "tone_aware": SCORE_TONE_AWARE,
        "threshold": REAL_WORD_THRESHOLD,
    }
    best_score = f1_on_dev(checker, best, errors, golds)
    print(f"Bắt đầu: F1={best_score[0]:.4f} exact={best_score[1]:.4f}")

    for round_number in range(1, ROUNDS + 1):
        for name, values in GRID.items():
            for value in values:
                if best[name] == value:
                    continue
                trial = dict(best, **{name: value})
                score = f1_on_dev(checker, trial, errors, golds)
                if (score[0], score[1]) > (best_score[0], best_score[1]):
                    best, best_score = trial, score
                    print(f"[vòng {round_number}] {name} = {value}: F1={score[0]:.4f} "
                          f"exact={score[1]:.4f} (precision {score[2]:.3f}, recall {score[3]:.3f})")

    print(f"\nKết quả tốt nhất trên tập dev: F1={best_score[0]:.4f}, exact={best_score[1]:.4f}")
    print("Chép vào config.py:")
    print(f"SCORE_EDIT_WEIGHT = {best['edit']}")
    print(f"SCORE_CONFUSION_BONUS = {best['confusion']}")
    print(f"SCORE_FREQ_WEIGHT = {best['freq']}")
    print(f"SCORE_LEFT_WEIGHT = {best['left']}")
    print(f"SCORE_RIGHT_WEIGHT = {best['right']}")
    print(f"SCORE_TONE_AWARE = {best['tone_aware']}")
    print(f"REAL_WORD_THRESHOLD = {best['threshold']}")


if __name__ == "__main__":
    main()