import html
import json
from datetime import datetime
from pathlib import Path
 
import streamlit as st
 
import suggest as S
from config import TOP_K
 
REJECT = "✗ Từ chối (giữ nguyên)"
FEEDBACK_FILE = Path("data/feedback.jsonl")
DEFAULT_TEXT = "Sinh viên cần hoàn thành bai tập trước khi nộp."
 
st.set_page_config(page_title="Sửa lỗi chính tả tiếng Việt", layout="centered")
st.title("Phát hiện và sửa lỗi chính tả tiếng Việt")
 
 
@st.cache_resource(show_spinner="Đang nạp mô hình ngôn ngữ...")
def load_all():
    lm, n_sentences = S.load_language_model()
    return lm, S.build_checkers(lm), n_sentences
 
 
lm, checkers, n_sentences = load_all()
st.sidebar.caption(f"Mô hình ngôn ngữ huấn luyện trên {n_sentences} câu, {len(lm.vocab)} từ.")
system = st.sidebar.radio("Hệ thống", ["Cải tiến", "Baseline"])
checker = checkers["improved" if system == "Cải tiến" else "baseline"]
 
# ---- nhập văn bản hoặc tải file .txt
uploaded = st.file_uploader("Tải lên file .txt (không bắt buộc)", type=["txt"])
if uploaded is not None:
    sig = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:          # chỉ nạp một lần cho mỗi file
        st.session_state["upload_sig"] = sig
        st.session_state["text_input"] = uploaded.read().decode("utf-8", errors="ignore")
st.session_state.setdefault("text_input", DEFAULT_TEXT)
text = st.text_area("Nhập hoặc dán đoạn văn", key="text_input", height=160)
 
if st.button("Kiểm tra", type="primary"):
    clean = S.nfc(text)
    st.session_state["checked_text"] = clean
    st.session_state["errors"] = S.suggest_sentence(checker, clean, top_k=max(TOP_K, 3), min_candidates=3)
    st.session_state["system"] = system
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
    st.session_state.pop("result", None)
 
errors = st.session_state.get("errors")
checked = st.session_state.get("checked_text", "")
 
if errors is not None:
    if not errors:
        st.success("Không phát hiện lỗi nào (mọi từ đều có trong từ điển).")
    else:
        # ---- đánh dấu vị trí nghi sai
        parts, pos = [], 0
        for k, e in enumerate(errors, 1):
            parts.append(html.escape(checked[pos:e["start"]]))
            tip = " / ".join(c for c, _ in e["candidates"][:3])
            parts.append(f"<mark title='{html.escape(tip)}'>{html.escape(e['original'])}<sup>{k}</sup></mark>")
            pos = e["end"]
        parts.append(html.escape(checked[pos:]))
        st.markdown("**Vị trí nghi sai:**")
        st.markdown("<div style='line-height:1.9;font-size:16px'>" + "".join(parts).replace("\n", "<br>")
                    + "</div>", unsafe_allow_html=True)
 
        # ---- chọn phương án cho từng lỗi
        st.subheader(f"{len(errors)} lỗi — chọn phương án hoặc từ chối")
        choices = {}
        for k, e in enumerate(errors):
            cands = S.confidence(e["candidates"][:3])
            options = [c for c, _ in cands] + [REJECT]
            conf = dict(cands)
            with st.container(border=True):
                st.markdown(f"**Lỗi {k + 1}:** `{e['original']}`")
                if not cands:
                    st.caption("Không tìm được phương án nào cho từ này.")
                pick = st.radio(
                    "Phương án", options, index=len(options) - 1, key=f"choice_{st.session_state['run_id']}_{k}",
                    format_func=lambda c, conf=conf: c if c == REJECT else f"{c}   ({conf[c]:.0%})")
                choices[k] = None if pick == REJECT else pick
 
        if st.button("Áp dụng lựa chọn", type="primary"):
            st.session_state["result"] = S.apply_choices(checked, errors, choices)
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:       # lưu phản hồi để phân tích thêm
                for k, e in enumerate(errors):
                    f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                        "system": st.session_state["system"], "original": e["original"],
                                        "options": [c for c, _ in e["candidates"][:3]],
                                        "accepted": choices.get(k)}, ensure_ascii=False) + "\n")
 
    if "result" in st.session_state:
        st.subheader("Văn bản sau khi sửa")
        st.text_area("Kết quả", st.session_state["result"], height=160)