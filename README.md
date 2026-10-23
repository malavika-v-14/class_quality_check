# Class Quality Assessor

Upload a class recording (or paste a link) and get a label (Poor / Average / Good / Excellent),
a confidence level, scores per area, timestamped observations and improvement tips.

## How it works
1. **Audio** is decoded and transcribed locally with `faster-whisper`.
2. **Signals** are measured: greeting and rapport, questions and wait time, calling students by
   name, praise, voice energy, estimated student speaking share, homework and closing, and from
   sampled video frames: face visible, 8x8 background, lighting, sharpness and screen changes.
3. **Rubric** (areas and weights in `pipeline/score.py`) follows the evaluators' feedback notes.
   The weighted total gives the label. A recording/setup problem caps the label at Average.
4. **Explanation** lists what was seen, with timestamps and short quotes, plus improvement tips.
5. **Optional AI teaching review** (off by default) rates explanation clarity, checking of
   understanding, handling of mistakes and lesson structure from the transcript text.

By default everything runs locally. No recording or transcript is sent to an outside API.

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the Whisper model once. Input: upload a file, or paste a link to ONE
video (direct link or public Google Drive file; folders are not supported).

## Set up the 8x8 background check
```
python make_background_ref.py data/excellent_1.mp4 data/excellent_2.mp4 data/good_1.mp4
```
This builds `assets/background_ref.png` (the median of many frames removes the moving teachers).
Open the image and check it shows the 8x8 background. Without it the background check is skipped.

## Tune on the sample recordings (keep them private, never commit them)
Put videos in `data/` named `excellent_1.mp4`, `good_1.mp4`, `average_1.mp4`, `weak_1.mp4`,
`quality_1.mp4`, then run `python calibrate.py`. It prints per-video scores, accuracy, suggested
cut-offs and which areas really separate the levels. Edit `THRESHOLDS` and weights in
`pipeline/score.py`, run again (results are cached; use `--fresh` after changing extract.py).

## Free hosting (Streamlit Community Cloud)
Push the repo to GitHub (without recordings), then at share.streamlit.io choose "New app", pick
the repo and `app.py`, and under Advanced settings > Secrets add:
```
APP_PASSWORD = "choose-a-password"
DEFAULT_MODEL = "base"
```
The free tier has about 2 CPU cores and 2.7 GB RAM and apps go to sleep when unused, so use the
`base` model and expect long recordings to be slow. Keep recordings out of GitHub.

## Optional settings (environment variables)
- `DEFAULT_MODEL` : base, small or medium (default small).
- `APP_PASSWORD` : asks for a password before the app opens (recommended for the hosted app).
- `LLM_ENABLED=1`, `ANTHROPIC_API_KEY`, optional `LLM_MODEL` : enables the AI teaching review
  (`pip install -r requirements-optional.txt`). It sends the transcript TEXT to the AI provider.
  Use only if the organiser permits that service. Never commit keys.

## Tests
```
python -m unittest discover
```

## Limits
- Very small sample set (about 10 labelled recordings), so thresholds and weights are starting
  values checked with calibrate.py, not a trained model. Update this line with your real result.
- Keyword rules assume mostly English speech; other languages are missed.
- Teacher/student split is a rough guess from voice pitch (children vs adult); it is skipped when
  the voices are not clearly different.
- Screen-change count also reacts to teacher movement, so it is a weak signal.
- Face detection can miss small faces in grid layouts.
- Long recordings (over 25 min) are sampled: start, middle and end only.
