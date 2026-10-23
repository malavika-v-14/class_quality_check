"""Run the pipeline on every recording in data/ and compare with the true label.
Name files like:  excellent_1.mp4  good_2.mp4  average_1.mp4  weak_1.mp4  quality_1.mp4
Usage:  python calibrate.py [--model small]
Results are cached in cache/ so re-running after tuning score.py is fast."""
import argparse
import csv
import glob
import json
import os

from pipeline.extract import analyze
from pipeline.score import ORDER, WEIGHTS, assess

EXPECTED = {"excellent": "Excellent", "good": "Good", "average": "Average", "weak": "Poor"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small")
ap.add_argument("--fresh", action="store_true", help="ignore cache")
ap.add_argument("--language", default=None, help="en, ml (Malayalam), hi, ta. Default: auto-detect")
args = ap.parse_args()

os.makedirs("cache", exist_ok=True)
rows = []
for path in sorted(glob.glob("data/*.*")):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith(".") or path.endswith(".gitkeep"):
        continue
    cache = f"cache/{name}.json"
    if os.path.exists(cache) and not args.fresh:
        feats = json.load(open(cache))
    else:
        print("Analysing", name, "...")
        feats = analyze(path, args.model, progress=print, language=args.language)
        json.dump(feats, open(cache, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    res = assess(feats)
    true = EXPECTED.get(name.split("_")[0].lower(), "-")
    row = {"file": name, "true": true, "predicted": res["label"], "total": round(res["total"]),
           "flags": "; ".join(res["flags"])}
    row.update({k: v["score"] for k, v in res["dimensions"].items()})
    rows.append(row)

if rows:
    keys = list(rows[0].keys())
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n" + "\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))
    graded = [r for r in rows if r["true"] != "-"]
    if graded:
        ok = sum(r["true"] == r["predicted"] for r in graded)
        near = sum(abs(ORDER.index(r["true"]) - ORDER.index(r["predicted"])) <= 1 for r in graded)
        print(f"\nExact match: {ok}/{len(graded)}   Within one level: {near}/{len(graded)}")
        means = {lab: [r["total"] for r in graded if r["true"] == lab] for lab in ORDER}
        means = {k: sum(v) / len(v) for k, v in means.items() if v}
        print("Average total per true label:", {k: round(v) for k, v in means.items()})
        labs = [l for l in ORDER if l in means]
        if len(labs) >= 2:
            cuts = [round((means[a] + means[b]) / 2) for a, b in zip(labs, labs[1:])]
            print("Suggested cut-offs between", " | ".join(labs), "->", cuts)
            if any(means[a] >= means[b] for a, b in zip(labs, labs[1:])):
                print("WARNING: scores do not rise with quality. Check the area columns below.")
        print("\nAverage score per area and true label (an area that does not rise is a weak signal):")
        for k in WEIGHTS:
            cells = []
            for lab in labs:
                v = [r[k] for r in graded if r["true"] == lab and k in r]
                cells.append(f"{lab}={sum(v) / len(v):.0f}" if v else f"{lab}=-")
            print(f"  {k:18s}", "  ".join(cells))
else:
    print("No recordings found in data/")