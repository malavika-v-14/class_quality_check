"""Feature extraction. Everything runs on your own computer (no API calls),
so children's recordings never leave the machine."""
import os
import re

import cv2
import librosa
import numpy as np
from faster_whisper import WhisperModel

SR = 16000
LANGUAGE = None          # None = auto-detect. Or "en", "ml" (Malayalam), "hi", "ta"
LONG_VIDEO_MIN = 25      # longer videos: transcribe only start, middle, end
SAMPLE_EVERY_SEC = 5     # video: look at one frame every N seconds
MAX_FRAMES = 400
BG_MIN_MATCHES = 25      # how many feature matches mean "background present"

_models = {}


def get_model(size):
    if size not in _models:
        _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _models[size]


def _frames(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def load_audio(path):
    """Decode audio to 16 kHz mono float32 with PyAV (works with old and new PyAV versions)."""
    import av
    container = av.open(path)
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("This file has no audio track.")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        parts = []
        for frame in container.decode(stream):
            for f in _frames(resampler.resample(frame)):
                parts.append(f.to_ndarray().reshape(-1))
        for f in _frames(resampler.resample(None)):
            parts.append(f.to_ndarray().reshape(-1))
    finally:
        container.close()
    if not parts:
        raise ValueError("Could not read any audio from this file.")
    return np.concatenate(parts).astype(np.float32) / 32768.0


def plan_chunks(duration):
    """Short class: whole audio. Long class: opening, middle, ending."""
    if duration <= LONG_VIDEO_MIN * 60:
        return [(0.0, duration)]
    mid = duration / 2
    return [(0.0, 480.0), (mid - 180, mid + 180), (duration - 420, duration)]


def transcribe(audio, chunks, model_size, language=None):
    model = get_model(model_size)
    segs, lang = [], language
    for a, b in chunks:
        part = audio[int(a * SR):int(b * SR)]
        it, info = model.transcribe(part, language=language, vad_filter=True, beam_size=1)
        lang = lang or info.language
        for s in it:
            segs.append({"start": a + s.start, "end": a + s.end,
                         "text": s.text.strip(), "logprob": s.avg_logprob})
    return segs, lang


def audio_features(audio):
    hop = 1024
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-9)
    silent = db < -40
    out = {"silence_ratio": float(silent.mean()), "pitch_std_st": None, "level_db": None}
    if (~silent).sum() > 50:
        out["level_db"] = float(20 * np.log10(np.median(rms[~silent]) + 1e-9))
        # pitch in ~4-minute pieces: one long call can need several GB of memory
        step, parts = hop * 4096, []
        for a in range(0, len(audio), step):
            piece = audio[a:a + step]
            if len(piece) < 4096:
                break
            g0 = a // hop
            f0 = librosa.yin(piece, fmin=70, fmax=450, sr=SR, frame_length=2048, hop_length=hop)
            m = min(len(f0), len(silent) - g0)
            if m > 0:
                parts.append(f0[:m][~silent[g0:g0 + m]])
        f = np.concatenate(parts) if parts else np.array([])
        f = f[(f > 75) & (f < 440)]
        if len(f) > 50:
            st = 12 * np.log2(f / np.median(f))
            out["pitch_std_st"] = float(np.std(np.clip(st, -12, 12)))
    return out


# ---------- transcript signals (simple, explainable keyword rules) ----------
P = {
    "greeting": r"\b(hello|hi|hey|good (morning|afternoon|evening)|welcome|namaste)\b",
    "rapport": r"\b(how are you|how is your|how was your|how's your|are you all (fine|good|ready))\b",
    "intro": r"\b(introduce yourself|introduction|new (friend|student|joiner|member)|tell (us|me) your name|what is your name|your name)\b",
    "homework": r"\b(home ?work|assignment|worksheet|practice (at|in) home|hw)\b",
    "closing": r"\b(see you|bye|good ?bye|thank you (all|everyone)|next class|take care|that's all for today|that is all for today|today we (learned|learnt|did)|recap|revise)\b",
    "praise": r"\b(very good|good job|well done|excellent|great|super|wonderful|brilliant|perfect|nice|awesome|shabash|clap)\b",
    "extras": r"\b(video|puzzle|game|activity|quiz|story|riddle|song|rhyme|experiment)\b",
}
STOP = {"So", "Now", "Okay", "Ok", "And", "But", "Next", "Let", "What", "Today", "Please",
        "Yes", "No", "Good", "Very", "Great", "Hello", "Class", "Everyone", "Can", "Do",
        "Did", "How", "Why", "Who", "Which", "Where", "When", "Thank", "The", "This",
        "That", "It", "You", "We", "Alright", "Right", "Well", "Then", "Look", "See",
        "Listen", "Tell", "Come", "Read", "Wait", "Show", "Open", "Students", "Children"}
NAME_RX = re.compile(r"\b([A-Z][a-z]{2,}),\s+(can|could|would|will|what|how|why|tell|please|"
                     r"now|your|come|do|are|you|read|try|answer|say|show|did|let)\b")


# Malayalam (script) versions of the same rules. No \\b here: it does not work with Malayalam letters.
PM = {
    "greeting": r"നമസ്കാരം|നമസ്തേ|ഹലോ|ഹായ്|ഗുഡ് ?(മോർണിംഗ്|ആഫ്റ്റർനൂൺ|ഈവനിംഗ്)|സ്വാഗതം",
    "rapport": r"സുഖമാണോ|സുഖമല്ലേ|എങ്ങനെയുണ്ട്|വിശേഷം",
    "intro": r"പരിചയപ്പെടു|സ്വയം പരിചയ|പേര്|പുതിയ (കുട്ടി|കൂട്ടുകാർ|സ്റ്റുഡന്റ്)",
    "homework": r"ഹോം ?വർക്ക്|ഗൃഹപാഠം|അസൈൻമെന്റ്|വർക്ക് ?ഷീറ്റ്",
    "closing": r"ബൈ|ടാറ്റാ|അടുത്ത ക്ലാസ്|നാളെ കാണാം|കാണാം|നന്ദി|റിവിഷൻ|ഇന്ന് പഠിച്ച",
    "praise": r"വെരി ഗുഡ്|ഗുഡ് ജോബ്|ഗുഡ്(?! ?(മോർണിംഗ്|ആഫ്റ്റർനൂൺ|ഈവനിംഗ്))|കൊള്ളാം|നന്നായി|സൂപ്പർ|എക്സലന്റ്|പൊളി|മിടുക്ക|അടിപൊളി|കയ്യടി|ക്ലാപ്",
    "extras": r"വീഡിയോ|ഗെയിം|പസിൽ|കഥ|പാട്ട്|ആക്ടിവിറ്റി|ക്വിസ്|കടംകഥ|റൈം",
}
_RX = {k: (re.compile(P[k], re.I), re.compile(PM[k])) for k in P}
ML_Q = re.compile(r"(ോ|ഓ)\s*[?.!]?\s*$|എന്ത്|എന്താണ്|ആരാണ്|എവിടെ|എപ്പോൾ|എത്ര|ഏത്|എങ്ങനെ|എന്തിന്")
NAME_ML = re.compile(r"([\u0D00-\u0D7F]{2,}),\s+(പറയൂ|പറയാമോ|ഒന്ന്|ഇനി|വായിക്ക|നോക്ക|ഉത്തരം|വരൂ|എഴുതൂ|ചെയ്യൂ)")


def is_question(text):
    t = text.rstrip()
    return t.endswith("?") or bool(ML_Q.search(t))


def _hits(key, segs, t0=None, t1=None):
    en, ml = _RX[key]
    return [s for s in segs
            if (en.search(s["text"]) or ml.search(s["text"]))
            and (t0 is None or s["start"] >= t0) and (t1 is None or s["start"] <= t1)]


def _ref(s):
    return {"t": s["start"], "text": s["text"][:140]}


def transcript_features(segs, duration, covered_sec):
    mins = max(covered_sec / 60, 0.5)
    words = sum(len(s["text"].split()) for s in segs)
    speech = sum(s["end"] - s["start"] for s in segs)
    wpm = words / (speech / 60) if speech > 5 else None

    questions = [s for s in segs if is_question(s["text"])]
    waits, no_wait = [], []
    for i, s in enumerate(segs[:-1]):
        if is_question(s["text"]):
            gap = segs[i + 1]["start"] - s["end"]
            if 0 <= gap < 15:
                waits.append(gap)
                if gap < 0.8:
                    no_wait.append({**_ref(s), "gap": gap})

    calls = []
    for s in segs:
        hit = any(m.group(1) not in STOP for m in NAME_RX.finditer(s["text"]))
        if hit or NAME_ML.search(s["text"]):
            calls.append(_ref(s))

    best = {"start": 0.0, "end": 0.0, "len": 0.0}
    start = None
    for i, s in enumerate(segs):
        if start is None:
            start = s["start"]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if nxt is None or nxt["start"] - s["end"] >= 3.0:
            if s["end"] - start > best["len"]:
                best = {"start": start, "end": s["end"], "len": s["end"] - start}
            start = None

    g = _hits("greeting", segs, 0, 180)
    r = _hits("rapport", segs, 0, 180)
    it = _hits("intro", segs, 0, 300)
    hw = _hits("homework", segs)
    cl = _hits("closing", segs, max(duration - 420, 0))
    pr = _hits("praise", segs)
    ex = _hits("extras", segs)
    lp = [s["logprob"] for s in segs if s.get("logprob") is not None]
    return {
        "n_words": words, "wpm": wpm, "minutes_covered": mins,
        "n_questions": len(questions), "avg_wait": (sum(waits) / len(waits)) if waits else None,
        "answered_ratio": (sum(1 for w in waits if w >= 1.0) / len(waits)) if waits else None,
        "no_wait": no_wait[:5], "calls": calls, "longest_block": best,
        "greeting": _ref(g[0]) if g else None, "rapport": _ref(r[0]) if r else None,
        "intro_prompt": _ref(it[0]) if it else None, "homework": _ref(hw[0]) if hw else None,
        "closing": _ref(cl[-1]) if cl else None,
        "n_praise": len(pr), "praise_examples": [_ref(s) for s in pr[:2]],
        "n_extras": len(ex), "extras_examples": [_ref(s) for s in ex[:2]],
        "mean_logprob": (sum(lp) / len(lp)) if lp else None,
    }


def speaker_split(audio, segs):
    """Rough teacher/student split: children's voices are usually higher than an adult's.
    Returns None when the two voice groups are not clearly separated."""
    vals, durs = [], []
    for sg in segs:
        a, b = int(sg["start"] * SR), int(sg["end"] * SR)
        if b - a < SR * 0.6:
            continue
        f0 = librosa.yin(audio[a:b], fmin=70, fmax=500, sr=SR, frame_length=1024, hop_length=512)
        f0 = f0[(f0 > 80) & (f0 < 480)]
        if len(f0) >= 5:
            vals.append(float(np.median(f0)))
            durs.append(sg["end"] - sg["start"])
    if len(vals) < 20:
        return None
    x, d = np.array(vals), np.array(durs)
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    for _ in range(20):
        high = np.abs(x - hi) < np.abs(x - lo)
        if high.all() or (~high).all():
            return None
        lo, hi = x[~high].mean(), x[high].mean()
    if hi - lo < 40 or min(high.sum(), (~high).sum()) < 5:
        return None
    t_lo, t_hi = d[~high].sum(), d[high].sum()
    student = t_hi if t_lo >= t_hi else t_lo      # the group that talks most is the teacher
    return {"share": float(student / (t_lo + t_hi)),
            "segments": int(high.sum() if t_lo >= t_hi else (~high).sum())}


# ---------- video signals ----------
def video_features(path, bg_ref=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frames / fps if frames > 0 else 0.0
    if duration <= 0:
        cap.release()
        return None
    face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ref_des = None
    if bg_ref and os.path.exists(bg_ref):
        img = cv2.imread(bg_ref, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, ref_des = orb.detectAndCompute(img, None)

    step = max(SAMPLE_EVERY_SEC, duration / MAX_FRAMES)
    samples, prev, t = [], None, 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_det.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        small = cv2.resize(gray, (64, 36))
        change = prev is not None and float(np.mean(cv2.absdiff(small, prev))) > 6
        prev = small
        bg = None
        if ref_des is not None:
            _, des = orb.detectAndCompute(gray, None)
            good = 0
            if des is not None and len(des) >= 2:
                for m in matcher.knnMatch(ref_des, des, k=2):
                    if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                        good += 1
            bg = good >= BG_MIN_MATCHES
        samples.append({"t": t, "face": len(faces) > 0, "bright": float(gray.mean()),
                        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                        "change": bool(change), "bg": bg})
        t += step
    cap.release()
    if not samples:
        return None

    n = len(samples)
    run_start, longest = None, (0.0, 0.0)
    for s in samples + [{"t": duration, "face": True}]:
        if not s["face"] and run_start is None:
            run_start = s["t"]
        if s["face"] and run_start is not None:
            if s["t"] - run_start > longest[1] - longest[0]:
                longest = (run_start, s["t"])
            run_start = None
    bgs = [s["bg"] for s in samples if s["bg"] is not None]
    return {
        "duration": duration, "n_samples": n,
        "face_ratio": sum(s["face"] for s in samples) / n,
        "bg_ratio": (sum(bgs) / len(bgs)) if bgs else None,
        "dark_ratio": sum(s["bright"] < 20 for s in samples) / n,
        "brightness": float(np.median([s["bright"] for s in samples])),
        "sharpness": float(np.median([s["sharp"] for s in samples])),
        "changes_per_min": sum(s["change"] for s in samples) / max(duration / 60, 0.5),
        "longest_no_face": {"start": longest[0], "end": longest[1]},
    }


def analyze(video_path, model_size="small", bg_ref="assets/background_ref.png", progress=None, language=None):
    say = progress or (lambda m: None)
    say("Reading audio...")
    audio = load_audio(video_path)
    duration = len(audio) / SR
    if duration < 30:
        raise ValueError("Recording is shorter than 30 seconds or has no audio.")
    say("Checking audio quality and voice...")
    af = audio_features(audio)
    say("Looking at video frames (face, background, light)...")
    vf = video_features(video_path, bg_ref)
    chunks = plan_chunks(duration)
    covered = sum(b - a for a, b in chunks)
    say(f"Transcribing speech ({covered / 60:.0f} min, model '{model_size}'). This is the slow step...")
    segs, lang = transcribe(audio, chunks, model_size, language or LANGUAGE)
    tf = transcript_features(segs, duration, covered)
    tf["language"] = lang
    say("Estimating who is speaking (teacher or students)...")
    sp = speaker_split(audio, segs)
    tf["student_share"] = sp["share"] if sp else None
    tf["student_segments"] = sp["segments"] if sp else None
    return {"duration": duration, "covered": covered, "sampled_only": len(chunks) > 1,
            "segments": segs, "audio": af, "text": tf, "video": vf}