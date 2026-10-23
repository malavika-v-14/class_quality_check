"""Rubric scoring. Pure Python: turns features into a label and an explanation.
The rubric follows the evaluators' own feedback notes (intro, rapport, individual
questions, homework, energy, 8x8 background, face visible, clear video)."""

WEIGHTS = {"opening": 15, "interaction": 20, "tone": 15, "homework_closing": 15,
           "setup": 20, "extras": 5, "clarity": 10, "teaching": 15}
NAMES = {"opening": "Opening & rapport", "interaction": "Interaction with students",
         "tone": "Energy & encouragement", "homework_closing": "Homework & closing",
         "setup": "Setup (face, background, light)", "extras": "Activities & extras",
         "clarity": "Audio / video clarity",
         "teaching": "Teaching quality (AI review)"}
# Tune these with calibrate.py on your sample recordings.
THRESHOLDS = [(80, "Excellent"), (60, "Good"), (40, "Average"), (0, "Poor")]
ORDER = ["Poor", "Average", "Good", "Excellent"]

SUGGEST = {
    "opening": "Start with a warm greeting, ask the kids how they are, and give new joiners a moment to introduce themselves.",
    "interaction": "Ask more questions, call students by name, and wait 2-3 seconds before moving on.",
    "tone": "Speak with more energy and a happier tone; praise answers more often (\"very good\", \"well done\").",
    "homework_closing": "Finish with a short recap and give homework before saying goodbye.",
    "setup": "Keep the 8x8 background on, keep your face visible and the camera on, and sit where there is good light.",
    "extras": "Add a short video, puzzle or activity related to the topic to keep kids engaged.",
    "clarity": "Check microphone and camera quality before class; speak close to the mic.",
    "teaching": "Explain ideas with simple examples, check understanding, and respond kindly to mistakes.",
}


def fmt(t):
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def ramp(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _q(text):
    return "\u201c" + text.strip()[:110] + "\u201d"


def dimension_scores(f, llm=None):
    t, a, v = f["text"], f["audio"], f["video"]
    per10 = 10.0 / t["minutes_covered"]
    d = {}
    d["opening"] = (0.3 * bool(t["greeting"]) + 0.4 * bool(t["rapport"])
                    + 0.3 * bool(t["intro_prompt"]))
    wait = ramp(t["avg_wait"], 0.5, 2.0) if t["avg_wait"] is not None else 0.0
    mono = 1 - ramp(t["longest_block"]["len"], 120, 420)
    base = (0.3 * ramp(t["n_questions"] * per10, 0, 6)
            + 0.3 * ramp(len(t["calls"]) * per10, 0, 3)
            + 0.25 * wait + 0.15 * mono)
    share = t.get("student_share")   # voice-based estimate, may be missing
    d["interaction"] = base if share is None else 0.9 * base + 0.1 * ramp(share, 0.05, 0.25)
    pitch = ramp(a["pitch_std_st"], 1.5, 4.0) if a["pitch_std_st"] is not None else 0.5
    english = t.get("language", "en") == "en"   # word-per-minute limits only make sense for English
    pace = 1 - ramp(abs(t["wpm"] - 135), 35, 80) if (t["wpm"] and english) else 0.5
    d["tone"] = 0.4 * pitch + 0.4 * ramp(t["n_praise"] * per10, 0, 4) + 0.2 * pace
    d["homework_closing"] = 0.5 * bool(t["homework"]) + 0.5 * bool(t["closing"])
    if v:
        parts = [(0.5, v["face_ratio"] * (1 - v["dark_ratio"]))]
        if v["bg_ratio"] is not None:
            parts.append((0.3, v["bg_ratio"]))
        b = v["brightness"]
        bright = 1.0 if 80 <= b <= 210 else (ramp(b, 30, 80) if b < 80 else 1 - ramp(b, 210, 250))
        parts.append((0.2, 0.5 * bright + 0.5 * ramp(v["sharpness"], 30, 120)))
        d["setup"] = sum(w * s for w, s in parts) / sum(w for w, _ in parts)
    else:
        d["setup"] = None
    words = ramp(t["n_extras"], 0, 3)
    d["extras"] = words if not v else 0.6 * words + 0.4 * ramp(v["changes_per_min"], 0.3, 2.0)
    lvl = ramp(a["level_db"], -45, -30) if a["level_db"] is not None else 0.5
    lp = ramp(t["mean_logprob"], -1.0, -0.3) if t["mean_logprob"] is not None else 0.5
    vid = ramp(v["sharpness"], 30, 120) if v else lvl
    d["clarity"] = 0.4 * lvl + 0.4 * lp + 0.2 * vid
    d["teaching"] = (llm["score"] / 5.0) if llm else None   # only when AI review was used
    return d


def label_for(total):
    for cutoff, name in THRESHOLDS:
        if total >= cutoff:
            return name


def observations(f, d, llm=None):
    t, a, v = f["text"], f["audio"], f["video"]
    per10 = 10.0 / t["minutes_covered"]
    o = []
    add = lambda kind, text: o.append({"kind": kind, "text": text})
    sampled = " (start, middle and end of the class were analysed)" if f["sampled_only"] else ""

    if t["greeting"]:
        add("+", f"Opening: greeted the class at {fmt(t['greeting']['t'])} \u2013 {_q(t['greeting']['text'])}")
    else:
        add("-", "No clear greeting found in the first 3 minutes.")
    if t["rapport"]:
        add("+", f"Built rapport at {fmt(t['rapport']['t'])} \u2013 {_q(t['rapport']['text'])}")
    else:
        add("-", "Went into the lesson without a rapport question (e.g. \"how are you?\").")
    if t["intro_prompt"]:
        add("+", f"Gave new students a chance to introduce themselves at {fmt(t['intro_prompt']['t'])}.")
    else:
        add("-", "No time was given for new joiners to introduce themselves.")

    add("+" if t["n_questions"] * per10 >= 3 else "-",
        f"Asked {t['n_questions']} questions in {t['minutes_covered']:.0f} min "
        f"(about {t['n_questions'] * per10:.1f} per 10 min){sampled}.")
    if t["calls"]:
        ex = "; ".join(f"{fmt(c['t'])} {_q(c['text'])}" for c in t["calls"][:2])
        add("+", f"Called students by name {len(t['calls'])} times, e.g. {ex}")
    else:
        add("-", "Did not call students by name to answer individually.")
    for n in t["no_wait"][:2]:
        add("-", f"At {fmt(n['t'])} asked {_q(n['text'])} but carried on after only {n['gap']:.1f}s \u2013 too little thinking time.")
    if t["avg_wait"] is not None and t["avg_wait"] >= 1.5:
        add("+", f"Waited about {t['avg_wait']:.1f}s on average after questions.")
    lb = t["longest_block"]
    if lb["len"] >= 180:
        add("-", f"Spoke for {lb['len'] / 60:.1f} min with no long pause ({fmt(lb['start'])}\u2013{fmt(lb['end'])}); little room for students.")

    if a["pitch_std_st"] is not None:
        if a["pitch_std_st"] < 2.0:
            add("-", "Voice sounded flat (little change in pitch), which can feel low-energy to children.")
        elif a["pitch_std_st"] >= 3.5:
            add("+", "Expressive, lively voice (good variation in pitch).")
    if t["n_praise"]:
        ex = t["praise_examples"][0]
        add("+", f"Praised students {t['n_praise']} times, e.g. at {fmt(ex['t'])} {_q(ex['text'])}")
    else:
        add("-", "Hardly any praise or encouragement found.")
    if t["wpm"] and t.get("language", "en") == "en" and (t["wpm"] < 90 or t["wpm"] > 180):
        add("-", f"Speaking pace was {t['wpm']:.0f} words per minute \u2013 {'slow' if t['wpm'] < 90 else 'fast'} for children.")

    if t["homework"]:
        add("+", f"Homework mentioned at {fmt(t['homework']['t'])} \u2013 {_q(t['homework']['text'])}")
    else:
        add("-", f"No homework was given or mentioned{sampled}.")
    if t["closing"]:
        add("+", f"Closed the class at {fmt(t['closing']['t'])} \u2013 {_q(t['closing']['text'])}")
    else:
        add("-", "No recap or goodbye found near the end of the class.")

    if v:
        if v["face_ratio"] < 0.6:
            add("-", f"Teacher's face was visible in only {v['face_ratio'] * 100:.0f}% of sampled frames.")
        nf = v["longest_no_face"]
        if nf["end"] - nf["start"] >= 60:
            add("-", f"No face visible from {fmt(nf['start'])} to {fmt(nf['end'])}.")
        if v["dark_ratio"] > 0.05:
            add("-", f"Camera looked off or black in {v['dark_ratio'] * 100:.0f}% of the class.")
        if v["bg_ratio"] is not None:
            add("+" if v["bg_ratio"] >= 0.7 else "-",
                f"8x8 background detected in {v['bg_ratio'] * 100:.0f}% of frames.")
        if v["brightness"] < 70:
            add("-", "Video is dark \u2013 sit where there is more light.")
        if v["sharpness"] < 30:
            add("-", "Video is blurry or low quality.")
    else:
        add("-", "Video could not be read, so setup checks were skipped.")
    if t["n_extras"]:
        ex = t["extras_examples"][0]
        add("+", f"Used extras such as videos, puzzles or activities, e.g. at {fmt(ex['t'])} {_q(ex['text'])}")
    share = t.get("student_share")
    if share is not None:
        add("+" if share >= 0.12 else "-",
            f"Students spoke about {share * 100:.0f}% of the time (estimated from voice pitch; approximate).")
    if v:
        cpm = v["changes_per_min"]
        if cpm >= 1:
            add("+", f"Visible content changed about {cpm:.1f} times per minute (slides, shared screen or activities).")
        elif cpm < 0.2 and f["duration"] > 600:
            add("-", "The screen hardly changed during the class; little visual material was used.")
    if d["clarity"] < 0.4:
        add("-", "Audio was quiet or hard to transcribe, which lowers confidence in this assessment.")
    for item in (llm or {}).get("observations", []):
        add(item["kind"], f"AI review at {item['time']}: {item['text']}")
    return o


def assess(f, llm=None):
    d = dimension_scores(f, llm)
    avail = {k: v for k, v in d.items() if v is not None}
    total = 100 * sum(WEIGHTS[k] * s for k, s in avail.items()) / sum(WEIGHTS[k] for k in avail)
    label = label_for(total)

    v = f["video"]
    flags = []
    if v:
        if v["face_ratio"] < 0.5:
            flags.append("Teacher's face not visible in most of the class")
        if v["bg_ratio"] is not None and v["bg_ratio"] < 0.5:
            flags.append("8x8 background missing")
        if v["dark_ratio"] > 0.1:
            flags.append("Camera off or black for part of the class")
    if d["clarity"] < 0.35:
        flags.append("Poor audio/video clarity")
    capped = False
    if flags and ORDER.index(label) > ORDER.index("Average"):
        label, capped = "Average", True

    gap = min(abs(total - c) for c, _ in THRESHOLDS if c > 0)
    conf = "High" if gap >= 8 else "Medium" if gap >= 4 else "Low"
    if flags and conf == "High":
        conf = "Medium"

    ranked = sorted(avail, key=lambda k: avail[k])
    summary = (f"Score {total:.0f}/100 \u2192 {label}. Strongest area: {NAMES[ranked[-1]]}. "
               f"Weakest area: {NAMES[ranked[0]]}.")
    if capped:
        summary += " The label was capped at Average because of recording/setup problems."
    return {
        "label": label, "total": total, "confidence": conf, "summary": summary,
        "flags": flags, "dimensions": {k: {"name": NAMES[k], "score": round(100 * s), "weight": WEIGHTS[k]}
                                       for k, s in avail.items()},
        "observations": observations(f, d, llm),
        "suggestions": (list((llm or {}).get("suggestions", []))
                        + [SUGGEST[k] for k in ranked if avail[k] < 0.55 and k != "teaching"])[:3],
    }