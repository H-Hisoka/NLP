import json
from datetime import datetime
from pathlib import Path

import streamlit as st

FEEDBACK_PATH = Path("outputs/feedback.jsonl")
REJECT = "Từ chối (giữ nguyên)"


# ------------------------------------------------- nối với pipeline của bạn
def get_suggestions(text, system):
    """
    TODO: sửa lại cho khớp code của bạn. Ví dụ:

        from improved_model import ImprovedModel
        model = ImprovedModel()
        return model.correct(text, top_k=3)

    Bản dưới đây chỉ là dữ liệu giả để chạy thử giao diện.
    """
    try:
        if system == "Cải tiến":
            from improved_model import correct
        else:
            from baseline import correct
        return correct(text, top_k=3)
    except ImportError:
        pass

    # ---- dữ liệu giả để thử giao diện ----
    words = text.split()
    demo, pos = [], 0
    for w in words:
        start = text.index(w, pos)
        end = start + len(w)
        pos = end
        if len(w) > 3:  # giả vờ coi các từ dài là có lỗi
            demo.append({
                "start": start, "end": end, "original": w,
                "error_type": "demo_error",
                "candidates": [(w.lower(), 0.9), (w.upper(), 0.5), (w.title(), 0.3)],
            })
            break
    return demo


# ------------------------------------------------------------- tiện ích
def apply_choices(text, errors, choices):
    """Thay các đoạn lỗi từ cuối về đầu để không lệch vị trí."""
    for i in sorted(range(len(errors)), key=lambda k: errors[k]["start"], reverse=True):
        choice = choices.get(i, REJECT)
        if choice != REJECT:
            e = errors[i]
            text = text[: e["start"]] + choice + text[e["end"]:]
    return text


def log_feedback(text, errors, choices, system):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        for i, e in enumerate(errors):
            choice = choices.get(i, REJECT)
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "system": system,
                "text": text,
                "original": e["original"],
                "error_type": e["error_type"],
                "candidates": [c for c, _ in e["candidates"]],
                "accepted": choice != REJECT,
                "chosen": None if choice == REJECT else choice,
            }, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- giao diện
st.set_page_config(page_title="Sửa lỗi văn bản", layout="centered")
st.title("Hệ thống phát hiện và sửa lỗi văn bản")

system = st.sidebar.radio("Hệ thống", ["Cải tiến", "Baseline"])
st.sidebar.caption("Baseline: Edit Distance + N-gram\n\nCải tiến: Edit Distance + Confusion Set + LM")

text = st.text_area("Nhập văn bản cần kiểm tra", height=150, key="input_text")

if st.button("Kiểm tra", type="primary"):
    st.session_state["checked_text"] = text
    st.session_state["errors"] = get_suggestions(text, system) if text.strip() else []
    st.session_state["system"] = system
    st.session_state.pop("result", None)

errors = st.session_state.get("errors")
checked_text = st.session_state.get("checked_text", "")

if errors is not None:
    if not errors:
        st.success("Không phát hiện lỗi nào.")
    else:
        st.subheader(f"Phát hiện {len(errors)} lỗi")
        choices = {}
        for i, e in enumerate(errors):
            with st.container(border=True):
                st.markdown(f"**Đoạn lỗi:** `{e['original']}`  \n**Nhóm lỗi:** {e['error_type']}")
                options = [c for c, _ in e["candidates"][:3]] + [REJECT]
                scores = {c: s for c, s in e["candidates"][:3]}
                choices[i] = st.radio(
                    "Chọn phương án",
                    options,
                    index=len(options) - 1,  # mặc định: từ chối
                    format_func=lambda c, s=scores: c if c == REJECT else f"{c}   (điểm {s[c]:.2f})",
                    key=f"choice_{i}",
                )

        if st.button("Áp dụng"):
            st.session_state["result"] = apply_choices(checked_text, errors, choices)
            log_feedback(checked_text, errors, choices, st.session_state["system"])

        if "result" in st.session_state:
            st.subheader("Văn bản sau khi sửa")
            st.text_area("Kết quả", st.session_state["result"], height=150)