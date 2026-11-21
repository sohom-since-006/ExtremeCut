"""Tests for live stream real-time clipping, AI judging, fallback, and framing."""
import unittest
import numpy as np
from pathlib import Path
import tempfile
import time

from studio import live, brain, creators, transcribe

class TestLiveClipping(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "clips").mkdir(parents=True, exist_ok=True)
        (self.dir / "buffer").mkdir(parents=True, exist_ok=True)
        self.job = {
            "id": "live-test-01",
            "creator": "ishowspeed",
            "title": "🔴 GTA 6 FINALE LAST MISSION",
            "uploader": "IShowSpeed",
            "source": {"name": "IShowSpeed Live", "url": "https://youtube.com/@IShowSpeed/live"},
            "clips": [],
            "dir": str(self.dir),
            "user": "u_local"
        }
        self.settings = dict(creators.DEFAULTS)
        self.settings["ai_every_min"] = 1
        self.settings["decide_mode"] = "either"
        self.settings["min_score"] = 60

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_ai_check_fallback_on_energy_peak(self):
        """When AI returns no candidates or is unavailable, loudness peak creates a clip."""
        sess = live.Session(self.job, self.settings)
        sess.dir = self.dir
        sess.buf = self.dir / "buffer"
        sess.t_done = 40.0
        sess.words = [
            {"w": "hello", "s": 2.0, "e": 2.5},
            {"w": "guys", "s": 2.6, "e": 3.0},
            {"w": "welcome", "s": 3.1, "e": 3.5},
            {"w": "to", "s": 3.6, "e": 3.8},
            {"w": "the", "s": 3.9, "e": 4.0},
            {"w": "stream", "s": 4.1, "e": 4.5},
            {"w": "look", "s": 20.0, "e": 20.5},
            {"w": "at", "s": 20.6, "e": 20.8},
            {"w": "this", "s": 20.9, "e": 21.2},
            {"w": "insane", "s": 21.3, "e": 21.8},
            {"w": "play", "s": 21.9, "e": 22.4},
            {"w": "oh", "s": 22.5, "e": 22.8},
            {"w": "my", "s": 22.9, "e": 23.2},
            {"w": "god", "s": 23.3, "e": 23.8},
        ]
        # 40 seconds of loudness, with a huge spike around second 22
        db_levels = [-25.0] * 40
        for i in range(20, 26):
            db_levels[i] = -5.0
        sess.db = db_levels

        sess._ai_check()

        # A clip should have been created via AI or fallback!
        self.assertGreater(len(self.job["clips"]), 0, "Expected at least 1 clip to be generated from energy peak")
        clip = self.job["clips"][0]
        self.assertGreater(len(clip["title"]), 5)
        self.assertNotEqual(clip["title"], "Loud moment")
        self.assertGreater(len(clip["hook"]), 2)
        self.assertNotEqual(clip["hook"], "WAIT FOR IT")
        self.assertGreater(len(clip["description"]), 20)
        self.assertIn("hashtags", clip)
        self.assertTrue(clip["status"] in ("queued", "rendering"))

    def test_live_align_does_not_block_with_whisper(self):
        """_align immediately returns guess when WHISPER is enabled, avoiding lockup."""
        sess = live.Session(self.job, self.settings)
        transcribe.WHISPER = True
        fake_src = self.dir / "test.ts"
        aligned = sess._align(fake_src, 120.5, 30.0)
        self.assertEqual(aligned, 120.5)

if __name__ == "__main__":
    unittest.main()
