"""Build assets/background_ref.png (the 8x8 background) automatically.
Teachers move, the background stays: the MEDIAN of many frames from several recordings
keeps the background and removes the teachers.
Usage: python make_background_ref.py data/excellent_1.mp4 data/excellent_2.mp4 data/good_1.mp4
Then OPEN the image and check that it shows the 8x8 background."""
import sys

import cv2
import numpy as np

frames, size = [], None
for path in sys.argv[1:]:
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for k in range(1, 13):
        if n <= 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * k / 13))
        ok, fr = cap.read()
        if not ok:
            continue
        if size is None:
            size = (640, int(fr.shape[0] * 640 / fr.shape[1]))
        frames.append(cv2.resize(fr, size))
    cap.release()
if len(frames) < 8:
    sys.exit("Not enough frames read. Pass 2-4 video paths.")
bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
cv2.imwrite("assets/background_ref.png", bg)
print(f"Saved assets/background_ref.png from {len(frames)} frames. Open it and check it.")
