import hmac
import os
import shutil
import tempfile

import streamlit as st

st.set_page_config(page_title="Class Quality Assessor", page_icon="🎓", layout="centered")


def _password():
    try:
        return os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD")
    except Exception:
        return os.getenv("APP_PASSWORD")


pw = _password()
if pw and not st.session_state.get("authed"):
    st.title("🎓 Class Quality Assessor")
    entered = st.text_input("Access password", type="password")
    if entered and hmac.compare_digest(entered, str(pw)):
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()

from pipeline import llm
from pipeline.score import assess, fmt

st.title("🎓 Class Quality Assessor")
st.caption("Upload a class recording or paste a link. The file is processed on this server and "
           "deleted straight after. Video and audio are never sent to outside AI services.")

_models = ["base", "small", "medium"]
_default = os.getenv("DEFAULT_MODEL", "small")
model_size = st.sidebar.selectbox("Speech model", _models,
                                  index=_models.index(_default) if _default in _models else 1,
                                  help="Bigger = more accurate but slower. Use 'base' on free hosting.")
LANGS = {"Auto-detect": None, "English": "en", "Malayalam": "ml", "Hindi": "hi", "Tamil": "ta"}
lang_choice = st.sidebar.selectbox("Class language", list(LANGS),
                                   help="If the class is in Malayalam choose Malayalam and the 'medium' model.")
use_llm = False
if llm.enabled():
    use_llm = st.sidebar.checkbox("Add AI teaching review",
                                  help="Sends the TEXT transcript (no audio/video) to an AI provider.")

tab_up, tab_link = st.tabs(["Upload file", "Paste link"])
with tab_up:
    upload = st.file_uploader("Class recording", type=["mp4", "mov", "mkv", "webm", "m4v", "avi"])
with tab_link:
    url = st.text_input("Link to ONE video (direct link or public Google Drive file)",
                        placeholder="https://drive.google.com/file/d/.../view")
    st.caption("Folders are not supported. Drive files must be shared as 'anyone with the link'.")

if st.button("Assess this class", type="primary"):
    if not upload and not url.strip():
        st.warning("Please upload a file or paste a link first.")
        st.stop()
    from pipeline.extract import analyze

    workdir = tempfile.mkdtemp()
    llm_res = None
    try:
        with st.status("Working...", expanded=True) as status:
            if upload:
                path = os.path.join(workdir, "video" + (os.path.splitext(upload.name)[1] or ".mp4"))
                with open(path, "wb") as out:
                    for chunk in iter(lambda: upload.read(8 * 1024 * 1024), b""):
                        out.write(chunk)
            else:
                from pipeline.fetch import download
                st.write("Downloading the video from the link...")
                path = download(url.strip(), workdir)
            feats = analyze(path, model_size, progress=st.write, language=LANGS[lang_choice])
            if use_llm:
                st.write("Running AI teaching review on the transcript...")
                llm_res = llm.review(feats["segments"])
            result = assess(feats, llm_res)
            status.update(label="Done", state="complete")
    except Exception as e:
        st.error(f"Could not process this recording: {e}")
        st.stop()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    icon = {"Excellent": "🟢", "Good": "🟢", "Average": "🟡", "Poor": "🔴"}[result["label"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality", f"{icon} {result['label']}")
    c2.metric("Score", f"{result['total']:.0f} / 100")
    c3.metric("Confidence", result["confidence"])
    st.write(result["summary"])
    lang = feats["text"].get("language")
    if lang and lang != "en":
        st.caption(f"Speech language: {lang}. Check the transcript below; Malayalam works best with the 'medium' model.")
    if use_llm and not llm_res:
        st.warning("The AI review was not available, so this result uses the rules only.")
    if llm_res and llm_res["summary"]:
        st.info("AI teaching review: " + llm_res["summary"])
    if feats["sampled_only"]:
        st.info("This is a long recording, so the start, middle and end were analysed "
                "(about %.0f of %.0f minutes)." % (feats["covered"] / 60, feats["duration"] / 60))
    for flag in result["flags"]:
        st.warning(f"Quality issue: {flag}")

    st.subheader("Scores by area")
    for dim in result["dimensions"].values():
        st.write(f"**{dim['name']}** \u2014 {dim['score']}/100 (weight {dim['weight']})")
        st.progress(dim["score"] / 100)

    st.subheader("What we observed")
    for o in result["observations"]:
        st.markdown(("\u2705 " if o["kind"] == "+" else "\u26a0\ufe0f ") + o["text"])

    if result["suggestions"]:
        st.subheader("How to improve")
        for s in result["suggestions"]:
            st.markdown(f"- {s}")

    with st.expander("Transcript with timestamps"):
        st.text("\n".join(f"[{fmt(s['start'])}] {s['text']}" for s in feats["segments"]))
    with st.expander("Raw measurements"):
        st.json({"audio": feats["audio"], "video": feats["video"],
                 "text": {k: v for k, v in feats["text"].items() if not isinstance(v, (list, dict))}})