import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from config import ERROR_TYPES  # list 30 nhóm lỗi
except ImportError:  # chạy độc lập để thử
    ERROR_TYPES = None


# ---------------------------------------------------------------- đọc dữ liệu
def load_results(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    # top3 lưu trong CSV thường là chuỗi JSON
    def _to_list(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return [x]
        return list(x) if x is not None else []

    df["top3"] = df["top3"].apply(_to_list)
    return df


# ------------------------------------------------------------------- độ đo
def _edit_distance(a, b):
    """Levenshtein trên chuỗi ký tự hoặc list từ."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(pred, ref):
    return _edit_distance(pred, ref) / max(len(ref), 1)


def wer(pred, ref):
    return _edit_distance(pred.split(), ref.split()) / max(len(ref.split()), 1)


# -------------------------------------------------------------- phân tích
def add_row_metrics(df):
    df = df.copy()
    df["pred1"] = df["top3"].apply(lambda c: c[0] if c else "")
    df["top1_ok"] = df["pred1"] == df["target"]
    df["top3_ok"] = [t in c[:3] for t, c in zip(df["target"], df["top3"])]
    df["cer"] = [cer(p, t) for p, t in zip(df["pred1"], df["target"])]
    df["wer"] = [wer(p, t) for p, t in zip(df["pred1"], df["target"])]
    return df


def analyze_by_type(df):
    """Bảng chỉ số theo từng nhóm lỗi."""
    g = df.groupby("error_type")
    out = pd.DataFrame({
        "n": g.size(),
        "top1": g["top1_ok"].mean(),
        "top3": g["top3_ok"].mean(),
        "exact_match": g["top1_ok"].mean(),  # exact match của ứng viên Top-1
        "cer": g["cer"].mean(),
        "wer": g["wer"].mean(),
    })
    if "detected" in df.columns:
        out["detection_rate"] = g["detected"].mean()
    if "pred_error_type" in df.columns:
        out["classification_acc"] = g.apply(
            lambda x: (x["pred_error_type"] == x["error_type"]).mean()
        )

    # đảm bảo đủ 30 nhóm, nhóm không có mẫu sẽ n = 0
    if ERROR_TYPES:
        out = out.reindex(ERROR_TYPES)
        out["n"] = out["n"].fillna(0).astype(int)
    out.index.name = "error_type"
    return out.sort_values("top1")


def confusion_matrix(df):
    """Ma trận nhầm lẫn giữa nhóm lỗi thật và nhóm lỗi dự đoán."""
    if "pred_error_type" not in df.columns:
        return None
    return pd.crosstab(df["error_type"], df["pred_error_type"])


def top_confusions(cm, k=10):
    """k cặp (thật -> dự đoán) bị nhầm nhiều nhất."""
    pairs = [
        (t, p, int(cm.loc[t, p]))
        for t in cm.index for p in cm.columns
        if t != p and cm.loc[t, p] > 0
    ]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def failure_examples(df, error_type, k=5):
    """Một vài mẫu sai Top-3 của một nhóm lỗi, để đưa vào báo cáo."""
    sub = df[(df["error_type"] == error_type) & (~df["top3_ok"])]
    return sub[["source", "target", "pred1"]].head(k)


def plot_top1_by_type(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.dropna(subset=["top1"]).sort_values("top1")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1))
    ax.barh(t.index, t["top1"])
    ax.set_xlabel("Top-1 accuracy")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="file kết quả .jsonl/.csv")
    ap.add_argument("--out", default="outputs/error_analysis", help="thư mục lưu kết quả")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_row_metrics(load_results(args.results))
    table = analyze_by_type(df)
    table.to_csv(out_dir / "metrics_by_error_type.csv", encoding="utf-8-sig")
    plot_top1_by_type(table, out_dir / "top1_by_error_type.png")

    print(f"Tổng số mẫu: {len(df)}")
    print(f"Top-1 toàn bộ: {df['top1_ok'].mean():.3f} | Top-3 toàn bộ: {df['top3_ok'].mean():.3f}")
    print("\n5 nhóm lỗi khó nhất (Top-1 thấp nhất):")
    print(table.dropna(subset=["top1"]).head(5).round(3).to_string())

    cm = confusion_matrix(df)
    if cm is not None:
        cm.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        print("\nCác cặp nhóm lỗi hay bị nhầm nhất:")
        for t, p, n in top_confusions(cm):
            print(f"  {t} -> {p}: {n}")

    # mẫu sai của 3 nhóm khó nhất, phục vụ viết phần phân tích lỗi trong báo cáo
    worst = table.dropna(subset=["top1"]).index[:3]
    with open(out_dir / "failure_examples.txt", "w", encoding="utf-8") as f:
        for et in worst:
            f.write(f"=== {et} ===\n")
            f.write(failure_examples(df, et).to_string(index=False) + "\n\n")

    print(f"\nĐã lưu kết quả vào: {out_dir}")


if __name__ == "__main__":
    main()