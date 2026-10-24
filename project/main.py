import sys

from config import TEST_SAMPLE_SIZE, REAL_WORD_THRESHOLD, THRESHOLD_VALUES
from data.preprocessing import load_data, split_train_test
from models.train_language_model import NGramLanguageModel
from models.improved_model import ImproveSpellChecker, load_cache, save_cache
from evaluation.baseline import BaselineSpellChecker
from evaluation.evaluation import evaluate_baseline, evaluate_improved
from evaluation.error_analysis import analyze_errors, show_sample_predictions

sys.stdout.reconfigure(encoding="utf-8")


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
    for threshold in THRESHOLD_VALUES:
        results = evaluate_improved("", improved, test_df, threshold, print_results_flag=False)
        print(f"{threshold:>7} {results['precision']:>10.4f} {results['recall']:>8.4f} {results['f1']:>8.4f} "
              f"{results['top1_accuracy']:>8.4f} {results['top3_accuracy']:>8.4f} {results['exact_match']:>8.4f}")

    analyze_errors(lm, improved, test_df, REAL_WORD_THRESHOLD)

    sample_sentences = ["sinh viên cần hoàn thành bai tập", "tôi muốn ăn cơm răng"]
    sample_sentences += test_df["wrong"].head(2).tolist()
    show_sample_predictions(lm, improved, sample_sentences, REAL_WORD_THRESHOLD)
    save_cache(improved)


if __name__ == "__main__":
    main()