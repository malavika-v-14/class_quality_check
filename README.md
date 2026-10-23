# Class Quality Assessor

An AI-powered app that assesses the quality of an online class recording. It gives:

- a label: **Poor / Average / Good / Excellent**
- a score out of 100 and a confidence level
- scores for 7 areas of the class
- **specific observations with timestamps and short quotes**
- up to 3 improvement tips

Built for the Humaniti AI "Can You Teach AI to Recognise a Great Class?" challenge.

**Hosted app:** https://classqualitycheck-jecoa8c4ag6slva7dnqdjr.streamlit.app/
Password : App@123

---

## 1. Use the hosted app (no installation)

1. Open the link above. If the page says the app is asleep, click the button to wake it
   and wait about one minute (free hosting puts unused apps to sleep).
2. Enter the access password.
3. In the left sidebar:
   - **Class language:** choose the language of the class (Auto-detect, English, Malayalam, Hindi, Tamil).
   - **Speech model:** `small` is a good default. `base` is faster, `medium` is more accurate but
     slow and may not fit on free hosting.
4. Provide the recording in one of two ways:
   - **Upload file** tab: upload an mp4, mov, mkv, webm, m4v or avi file.
   - **Paste link** tab: paste a link to ONE video (a direct video link, or a Google Drive file
     shared as "anyone with the link"). Folder links are not supported.
5. Click **Assess this class** and wait. Transcription is the slow step; a 20-minute class can
   take several minutes on free hosting. Please do not refresh the page while it works.
6. Read the result: label, score, confidence, scores by area, observations, and tips.
   The full transcript and raw measurements are in expandable sections at the bottom.

**Tips**
- Recordings should be at least 30 seconds long and have an audio track.
- For classes longer than 25 minutes, the app analyses the start (8 min), the middle (6 min)
  and the end (7 min) to stay fast. The result tells you when this happens.
- Free hosting has limited memory. For very large files, a shorter recording or the `base`
  model is safer.

## 2. Run it on your own computer

You need Python 3.10, 3.11 or 3.12.

```
git clone <this-repository-url>
cd <repository-folder>
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The first run downloads the Whisper speech model
once, so it needs internet and is slower.

## 3. How it works

```
Recording -> audio + video frames -> transcript and measurements -> rubric score -> label + explanation
```

1. **Audio** is decoded and transcribed locally with `faster-whisper`.
2. **Voice** is measured with `librosa`: pitch variation (energy), loudness, silence,
   and a rough estimate of how much the students spoke (children's voices are usually higher).
3. **Video** is sampled (one frame every few seconds) with OpenCV: is the teacher's face visible,
   is the 8x8 background present, is the camera on, is the picture bright and sharp,
   how often does the screen change.
4. **Transcript signals** use simple, explainable rules (English and Malayalam keyword lists):
   greeting and rapport, time for new joiners to introduce themselves, number of questions,
   wait time after questions, calling students by name, praise, homework, recap or goodbye,
   videos, puzzles and activities, and the longest stretch without a pause.
5. **Rubric.** The areas come from the evaluators' own feedback notes on the sample classes:

   | Area | Weight |
   |---|---|
   | Opening and rapport | 15 |
   | Interaction with students | 20 |
   | Energy and encouragement | 15 |
   | Homework and closing | 15 |
   | Setup (face, background, light) | 20 |
   | Activities and extras | 5 |
   | Audio / video clarity | 10 |

   The weighted total gives the label (below 40 Poor, 40-60 Average, 60-80 Good, 80+ Excellent).
   If there is a recording or setup problem (face not visible, background missing, camera off,
   poor audio), the label is capped at **Average**, as in the "known quality issues" examples.
6. **Explanation.** Each finding becomes a sentence with a time and a quote, for example:
   *"At 12:40 asked 'What is a noun?' but carried on after only 0.3s - too little thinking time."*

**Why rules and not a trained model?** Only about 10 labelled recordings were available. A trained
model would memorise them. A rubric built from the evaluators' notes generalises better and every
result can be explained.

**Which signals can mislead?** Long talking time, loud audio, and many questions do not by
themselves mean a good class. That is why no single signal decides the label and why setup
problems are reported separately instead of simply lowering the teaching score.

## 4. Privacy

- Video and audio are processed on the server that runs the app and are **deleted straight after**
  processing. Nothing is stored.
- Video and audio are **never sent to outside AI services**.
- Recordings are not part of this repository (see `.gitignore`).
- An **optional** AI teaching review exists but is **off by default**. If turned on, it sends only the
  text transcript to an AI provider. It is not enabled in the hosted app.

## 5. Project structure

```
app.py                     Streamlit interface (upload / link, results)
pipeline/extract.py        audio, transcript and video measurements
pipeline/score.py          rubric, label, observations, tips
pipeline/fetch.py          download a video from a link
pipeline/llm.py            optional AI teaching review (off by default)
calibrate.py               run on sample recordings and compare with true labels
make_background_ref.py     build the 8x8 background reference image
tests/test_score.py        unit tests for scoring
```

## 6. Tune it on your own recordings (optional)

Put videos in `data/` named `excellent_1.mp4`, `good_1.mp4`, `average_1.mp4`, `weak_1.mp4`,
`quality_1.mp4` and so on, then run:

```
python make_background_ref.py data/excellent_1.mp4 data/excellent_2.mp4   # builds assets/background_ref.png
python calibrate.py --model small --language ml
```

`calibrate.py` prints the score of every video, how many labels matched, suggested label
cut-offs, and which areas separate the quality levels. Edit `THRESHOLDS` and the weights in
`pipeline/score.py`, then run it again (results are cached; use `--fresh` after changing
`pipeline/extract.py`). Never commit the recordings.

Run the tests with:

```
python -m unittest discover
```

## 7. Settings (environment variables or Streamlit secrets)

| Name | Meaning |
|---|---|
| `APP_PASSWORD` | Ask for a password before the app opens |
| `DEFAULT_MODEL` | `base`, `small` or `medium` (default `small`) |
| `LLM_ENABLED`, `ANTHROPIC_API_KEY`, `LLM_MODEL` | Turn on the optional AI teaching review. Use only if the data rules allow it. Never commit keys. |

## 8. Limitations

- The sample set is very small, so the weights and cut-offs are starting values checked on a
  handful of recordings. They are not a validated model.
- Keyword rules cover English and basic Malayalam. Other languages, and mixed-language speech,
  may be missed. Speech recognition for Malayalam is weaker with the small models.
- There is no true speaker separation. The teacher/student split is a rough guess from voice pitch
  and is skipped when the voices are not clearly different.
- Question detection in Malayalam is approximate.
- The screen-change count also reacts to teacher movement, so it is a weak signal.
- Face detection can miss small faces in grid layouts.
- Long recordings (over 25 minutes) are sampled, so events in the unsampled parts are missed.
- Free hosting is slow, has limited memory, and goes to sleep when unused.
