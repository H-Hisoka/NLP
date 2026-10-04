import argparse
import difflib
import re
from pathlib import Path
 
import pandas as pd
 
from config import GENERATED_DATA_FILE, PROCESSED_DATA_FILE, ERROR_TYPES, TOP_K, RANDOM_SEED
from evaluation import evaluate_system
import suggest as S
 
SRC_COLS = ["error_sentence", "noisy_sentence", "wrong_sentence", "incorrect_sentence", "erroneous_sentence",
            "source", "input", "error_text", "sentence_with_error"]
GOLD_COLS = ["correct_sentence", "original_sentence", "target", "clean_sentence", "ground_truth", "gold",
             "label", "original", "sentence"]
TYPE_COLS = ["error_type", "error_types", "type", "types", "errors", "error"]
 
GROUPS = ["Sai dấu", "Sai ký tự", "Lỗi tách từ", "Từ riêng", "Từ chuyên ngành", "Real-word error",
          "Sửa nhầm từ vốn đã đúng", "Khác (thiếu/thừa/đảo từ)"]
SYSTEM_NAMES = {"baseline": "Baseline", "improved": "Cải tiến"}
 
_TYPE_RE = re.compile(r"(?<![A-Za-z_])(" + "|".join(sorted(map(re.escape, ERROR_TYPES), key=len, reverse=True))
                      + r")(?![A-Za-z_])")
 
 
# đọc dữ liệu
def find_col(df, given, candidates, what, flag):
    lower = {c.lower(): c for c in df.columns}
    if given:
        if given not in df.columns:
            raise SystemExit(f"Không có cột '{given}'. Các cột hiện có: {list(df.columns)}")
        return given
    for c in candidates:
        if c in lower:
            return lower[c]
    raise SystemExit(f"Không đoán được cột {what}. Các cột hiện có: {list(df.columns)}\n"
                     f"Hãy chỉ định bằng {flag} <tên cột>.")
 
 
def parse_types(value):
    """Lấy các loại lỗi trong ERROR_TYPES xuất hiện trong ô (chịu mọi định dạng: 'a|b', 'a,b', "['a','b']"...)."""
    return list(dict.fromkeys(_TYPE_RE.findall(str(value))))
 
 
def read_terms(path):
    if not path:
        return set()
    return {l.strip().lower() for l in open(path, encoding="utf-8") if l.strip()}
 
 
# căn chỉnh lỗi chuẩn (gold)
def gold_units(src, gold):
    """
    So hai danh sách từ, trả về các 'đơn vị lỗi chuẩn':
      - thay từ 1-1: mỗi từ sai là một đơn vị `fixable` (hệ thống có thể sửa bằng cách đổi từ);
      - các trường hợp còn lại (thiếu/thừa từ, tách/ghép từ, đảo từ): một đơn vị không `fixable`.
    """
    units = []
    sm = difflib.SequenceMatcher(None, src, gold, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                units.append({"span": (i1 + k, i1 + k + 1), "src": src[i1 + k], "gold": gold[j1 + k],
                              "j": j1 + k, "fixable": True})
        else:
            units.append({"span": (i1, i2), "src": " ".join(src[i1:i2]), "gold": " ".join(gold[j1:j2]),
                          "j": j1, "fixable": False})
    return units
 
 
def classify_error(u, kind, vocab, tech, proper):
    """Nhóm của một lỗi: tách từ > từ riêng > chuyên ngành > real-word > sai dấu > sai ký tự."""
    if kind == "Sửa nhầm từ đúng":
        return "Sửa nhầm từ vốn đã đúng"
    src_w, gold_w = u["src"], u["gold"]
    if not u["fixable"]:
        if src_w.replace(" ", "").lower() == gold_w.replace(" ", "").lower():
            return "Lỗi tách từ"
        return "Khác (thiếu/thừa/đảo từ)"
    if (gold_w[:1].isupper() and u["j"] > 0) or gold_w.lower() in proper:
        return "Từ riêng"
    if gold_w.lower() in tech:
        return "Từ chuyên ngành"
    if src_w.lower() in vocab:
        return "Real-word error"
    if S.strip_marks(src_w.lower()) == S.strip_marks(gold_w.lower()):
        return "Sai dấu"
    return "Sai ký tự"
 
 
#  chạy một hệ thống
def run_system(name, checker, rows, top_k, tech, proper):
    vocab = checker.lm.vocab
    preds, top3s, golds = [], [], []
    TP = FP = FN = 0
    sent_rows, cases = [], []
    for r in rows:
        src_tokens, gold_tokens = r["source"].split(), r["gold"].split()
        errs = S.suggest_sentence(checker, r["source"], top_k)
        by_idx = {e["index"]: e for e in errs}
 
        pred = S.autocorrect(r["source"], errs)
        top3 = S.apply_choices(r["source"], errs, {k: "|".join(c for c, _ in e["candidates"][:3])
                                                   for k, e in enumerate(errs) if e["candidates"]})
        preds.append(pred); top3s.append(top3); golds.append(r["gold"])
 
        units = gold_units(src_tokens, gold_tokens)
        err_idx = {k for u in units for k in range(*u["span"])}
        n_det = n_top1 = n_top3 = 0
        for u in units:
            i1, i2 = u["span"]
            detected = any(k in by_idx for k in range(i1, i2))
            cl = [c.lower() for c, _ in by_idx[i1]["candidates"]] if u["fixable"] and detected else []
            top1 = cl[:1] == [u["gold"].lower()]
            top3_hit = u["gold"].lower() in cl[:3]
            TP += detected
            FN += not detected
            n_det += detected; n_top1 += top1; n_top3 += top3_hit
            if not top1:
                kind = "Bỏ sót" if not detected else "Sửa sai"
                shown = cl[0] if cl else "—"
                cases.append({"Hệ thống": name, "Kiểu lỗi mô hình": kind,
                              "Nhóm lỗi": classify_error(u, kind, vocab, tech, proper),
                              "Loại lỗi (config)": "|".join(r["types"]), "Câu nguồn": r["source"],
                              "Từ lỗi": u["src"], "Đáp án": u["gold"], "Mô hình đề xuất": shown})
        for k, e in by_idx.items():                       # hệ thống đánh dấu một từ vốn đúng
            if k not in err_idx:
                FP += 1
                cases.append({"Hệ thống": name, "Kiểu lỗi mô hình": "Sửa nhầm từ đúng",
                              "Nhóm lỗi": "Sửa nhầm từ vốn đã đúng", "Loại lỗi (config)": "|".join(r["types"]),
                              "Câu nguồn": r["source"], "Từ lỗi": e["original"], "Đáp án": e["original"],
                              "Mô hình đề xuất": e["candidates"][0][0] if e["candidates"] else "—"})
        sent_rows.append({"types": r["types"], "n_units": len(units), "det": n_det, "top1": n_top1,
                          "top3": n_top3, "exact": pred == r["gold"]})
    overall = evaluate_system(preds, golds, top3s, TP, FP, FN)
    return overall, pd.DataFrame(sent_rows), pd.DataFrame(cases)
 
 
def table_by_type(sent_df):
    rows = []
    for t in ERROR_TYPES:
        sub = sent_df[sent_df["types"].apply(lambda ts: t in ts)]
        clean = sub[(sub["types"].apply(len) == 1) & (sub["n_units"] == 1)]
        nu = sub["n_units"].sum()
        rows.append({
            "error_type": t,
            "số_câu": len(sub),
            "số_từ_lỗi": int(nu),
            "phát_hiện": sub["det"].sum() / nu if nu else float("nan"),
            "top1": sub["top1"].sum() / nu if nu else float("nan"),
            "top3": sub["top3"].sum() / nu if nu else float("nan"),
            "exact_match_câu": sub["exact"].mean() if len(sub) else float("nan"),
            "số_câu_1_lỗi": len(clean),
            "top1_câu_1_lỗi": clean["top1"].mean() if len(clean) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("error_type")
 
 
# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=GENERATED_DATA_FILE, help="file lỗi sinh ra (CSV)")
    ap.add_argument("--train-file", default=PROCESSED_DATA_FILE, help="câu đúng để huấn luyện mô hình ngôn ngữ")
    ap.add_argument("--train-col", default=None)
    ap.add_argument("--src-col", default=None, help="cột câu có lỗi")
    ap.add_argument("--gold-col", default=None, help="cột câu đúng")
    ap.add_argument("--type-col", default=None, help="cột loại lỗi")
    ap.add_argument("--limit", type=int, default=0, help="chỉ đánh giá ngẫu nhiên N câu (0 = tất cả)")
    ap.add_argument("--no-holdout", action="store_true",
                    help="KHÔNG loại câu đánh giá khỏi dữ liệu huấn luyện LM (mặc định có loại để tránh rò rỉ)")
    ap.add_argument("--systems", nargs="+", default=["baseline", "improved"], choices=list(SYSTEM_NAMES))
    ap.add_argument("--tech-file", default=None, help="(tuỳ chọn) file thuật ngữ chuyên ngành, mỗi dòng một từ")
    ap.add_argument("--proper-file", default=None, help="(tuỳ chọn) file tên riêng, mỗi dòng một từ")
    ap.add_argument("--out", default="data/analysis")
    args = ap.parse_args()
 
    df = pd.read_csv(args.data)
    sc = find_col(df, args.src_col, SRC_COLS, "câu có lỗi", "--src-col")
    gc = find_col(df, args.gold_col, GOLD_COLS, "câu đúng", "--gold-col")
    tc = find_col(df, args.type_col, TYPE_COLS, "loại lỗi", "--type-col")
    df = df.dropna(subset=[sc, gc])
    if args.limit and len(df) > args.limit:
        df = df.sample(args.limit, random_state=RANDOM_SEED)
    rows = [{"source": S.nfc(str(a)), "gold": S.nfc(str(b)), "types": parse_types(t)}
            for a, b, t in zip(df[sc], df[gc], df[tc])]
    print(f"Đánh giá {len(rows)} câu | cột: nguồn='{sc}', đúng='{gc}', loại lỗi='{tc}'")
    untyped = sum(1 for r in rows if not r["types"])
    if untyped:
        print(f"  Cảnh báo: {untyped} câu không có loại lỗi nào khớp ERROR_TYPES (vẫn tính vào chỉ số tổng thể).")
 
    exclude = None if args.no_holdout else {r["gold"] for r in rows}
    lm, n_train = S.load_language_model(args.train_file, args.train_col, exclude)
    print(f"Mô hình ngôn ngữ: {n_train} câu huấn luyện, {len(lm.vocab)} từ trong từ điển"
          + ("" if args.no_holdout else " (đã loại các câu đánh giá)"))
    checkers = S.build_checkers(lm)
    tech, proper = read_terms(args.tech_file), read_terms(args.proper_file)
 
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    overall, per_type, all_cases = {}, {}, []
    for key in args.systems:
        name = SYSTEM_NAMES[key]
        print(f"\nĐang chạy {name} ...")
        overall[name], sent_df, cases = run_system(name, checkers[key], rows, TOP_K, tech, proper)
        per_type[name] = table_by_type(sent_df)
        all_cases.append(cases)
 
    overall_df = pd.DataFrame(overall).T
    print("\n=== Chỉ số tổng thể (evaluation.evaluate_system) ===")
    print(overall_df.round(4).to_string())
    overall_df.to_csv(out / "overall_metrics.csv", encoding="utf-8-sig")
 
    by_type = pd.concat(per_type, axis=1)
    by_type.to_csv(out / "metrics_by_error_type.csv", encoding="utf-8-sig")
    print("\n=== Theo 30 loại lỗi (ERROR_TYPES) ===")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(by_type.round(3).to_string())
 
    cases = pd.concat(all_cases, ignore_index=True)
    cases.to_csv(out / "error_cases.csv", index=False, encoding="utf-8-sig")
    summary = (cases.pivot_table(index="Nhóm lỗi", columns="Hệ thống", values="Câu nguồn",
                                 aggfunc="count", fill_value=0).reindex(GROUPS).fillna(0).astype(int))
    summary["Tổng"] = summary.sum(axis=1)
    summary.to_csv(out / "summary_by_group.csv", encoding="utf-8-sig")
    print(f"\n=== Phân loại lỗi của mô hình theo nhóm (tổng {len(cases)} lỗi; đề yêu cầu >= 30) ===")
    print(summary.to_string())
    print("ĐẠT yêu cầu >= 30 lỗi" if len(cases) >= 30 else "CHƯA ĐỦ 30 lỗi: hãy tăng --limit hoặc dùng thêm dữ liệu")
 
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top1 = pd.DataFrame({n: t["top1"] for n, t in per_type.items()})
        ax = top1.plot(kind="barh", figsize=(9, 9))
        ax.set_xlabel("Top-1 (trên từ lỗi)"); ax.set_xlim(0, 1); ax.invert_yaxis()
        plt.tight_layout(); plt.savefig(out / "top1_by_error_type.png", dpi=150); plt.close()
    except ImportError:
        pass
    print(f"\nĐã lưu kết quả vào: {out}")
 
 
if __name__ == "__main__":
    main()