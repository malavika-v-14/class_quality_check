import copy
import unittest

from pipeline.llm import parse
from pipeline.score import assess

REF = lambda t, x: {"t": t, "text": x}
GOOD = {
    "duration": 2400, "covered": 2400, "sampled_only": False, "segments": [],
    "audio": {"silence_ratio": .3, "pitch_std_st": 3.8, "level_db": -24},
    "text": {"minutes_covered": 40, "wpm": 130, "n_questions": 50, "avg_wait": 2.1, "answered_ratio": .7,
             "no_wait": [{"t": 300, "text": "What is this?", "gap": .3}],
             "calls": [REF(60, "Aarav, can you tell me?")] * 30,
             "longest_block": {"start": 10, "end": 100, "len": 90},
             "greeting": REF(5, "Hello everyone"), "rapport": REF(12, "How are you all?"),
             "intro_prompt": REF(30, "introduce yourself"), "homework": REF(2300, "Your homework is"),
             "closing": REF(2380, "See you next class"), "n_praise": 30,
             "praise_examples": [REF(90, "Very good")], "n_extras": 4,
             "extras_examples": [REF(500, "Let's play a game")], "mean_logprob": -.3,
             "student_share": 0.3},
    "video": {"face_ratio": .95, "bg_ratio": .9, "dark_ratio": 0, "brightness": 130, "sharpness": 90,
              "changes_per_min": 2, "longest_no_face": {"start": 0, "end": 0}},
}


class ScoreTests(unittest.TestCase):
    def test_good_class_is_excellent(self):
        self.assertEqual(assess(GOOD)["label"], "Excellent")

    def test_missing_basics_and_setup_issue_is_capped(self):
        bad = copy.deepcopy(GOOD)
        bad["text"].update(greeting=None, rapport=None, intro_prompt=None, homework=None, closing=None,
                           n_praise=0, n_questions=2, calls=[], n_extras=0, avg_wait=.2, student_share=None)
        bad["video"].update(face_ratio=.3, bg_ratio=.2)
        r = assess(bad)
        self.assertIn(r["label"], ("Poor", "Average"))
        self.assertTrue(r["flags"])

    def test_video_unreadable_does_not_crash(self):
        f = copy.deepcopy(GOOD)
        f["video"] = None
        self.assertNotIn("setup", assess(f)["dimensions"])

    def test_old_cache_without_student_share(self):
        f = copy.deepcopy(GOOD)
        del f["text"]["student_share"]
        self.assertIn(assess(f)["label"], ("Excellent", "Good"))

    def test_ai_review_adds_dimension(self):
        llm = parse('{"scores":{"explanation_clarity":4,"checks_understanding":5,"handles_mistakes_kindly":3,'
                    '"lesson_structure":4},"observations":[{"time":"3:10","kind":"+","text":"Clear example"}],'
                    '"suggestions":["Wait longer"],"summary":"Good."}')
        r = assess(GOOD, llm)
        self.assertIn("teaching", r["dimensions"])
        self.assertTrue(any("AI review" in o["text"] for o in r["observations"]))

    def test_llm_parse_clips_scores(self):
        out = parse('x {"scores":{"explanation_clarity":9,"checks_understanding":-2,"handles_mistakes_kindly":3,'
                    '"lesson_structure":4}} y')
        self.assertEqual(out["scores"]["explanation_clarity"], 5.0)
        self.assertEqual(out["scores"]["checks_understanding"], 0.0)


if __name__ == "__main__":
    unittest.main()
