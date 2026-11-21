"""Tests for live recording options and framing modal workflow."""
import unittest
from studio import brain, live, jobs


class TestLiveOptions(unittest.TestCase):
    def test_unique_seo_generation(self):
        """Verify that every clip gets a unique, high-CTR YouTube title, hook, and description."""
        video_title = "GTA 6 FINALE LAST MISSION 90% RIGHT NOW GET IN HERE!"
        uploader = "IShowSpeed"
        words = [
            {"w": "Oh", "s": 0.0, "e": 0.3},
            {"w": "my", "s": 0.4, "e": 0.7},
            {"w": "god", "s": 0.8, "e": 1.1},
            {"w": "look", "s": 1.2, "e": 1.5},
            {"w": "at", "s": 1.6, "e": 1.8},
            {"w": "this", "s": 1.9, "e": 2.2},
            {"w": "play", "s": 2.3, "e": 2.6},
        ]
        
        # Test 5 clips with different timestamps and indices
        clips_seo = []
        for i in range(1, 6):
            clip = {"start": float(i * 30), "end": float(i * 30 + 25), "score": 75}
            seo = brain.generate_clip_seo(clip, words, video_title, uploader, clip_index=i, ai_enhance=False)
            clips_seo.append(seo)
            self.assertNotIn(seo["title"], ("Loud moment", "Clip", ""))
            self.assertNotEqual(seo["hook"], "WAIT FOR IT")
            self.assertTrue(len(seo["title"]) >= 15)
            self.assertTrue(len(seo["hook"]) >= 3)
            self.assertTrue(len(seo["description"]) >= 50)
            self.assertTrue(len(seo["hashtags"]) >= 4)

        # Titles and hooks must be strictly distinct
        titles = [s["title"] for s in clips_seo]
        hooks = [s["hook"] for s in clips_seo]
        self.assertEqual(len(titles), len(set(titles)), "All generated titles must be strictly unique")
        self.assertEqual(len(hooks), len(set(hooks)), "All generated hooks must be strictly unique")

    def test_session_options_storage(self):
        """Verify that options passed to start_session and update_session_options are preserved."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dummy_creator = {
                "id": "test_creator",
                "name": "Test Streamer",
                "channel": "https://www.twitch.tv/test_streamer",
                "style": "bounce",
                "layout": "blur",
                "channels": {"twitch": "https://www.twitch.tv/test_streamer"}
            }
            custom_opts = {
                "layout": "bottom",
                "style": "neon",
                "top": 0.40,
                "cam": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3},
                "game": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                "min_score": 75,
                "min_len": 20,
                "max_len": 50,
                "ai_every_min": 2,
                "trigger_words": "omg, clutch, insane"
            }
            
            # Simulate Session object with settings
            dummy_job = {
                "id": "job_test_live",
                "creator": dummy_creator["id"],
                "user": "u_local",
                "dir": td,
                "clips": [],
                "source": {"url": dummy_creator["channel"]}
            }
            
            merged = dict(dummy_creator)
            merged.update(custom_opts)
            
            sess = live.Session(dummy_job, merged)
            self.assertEqual(sess.settings["layout"], "bottom")
            self.assertEqual(sess.settings["style"], "neon")
            self.assertEqual(sess.settings["top"], 0.40)
            self.assertEqual(sess.settings["min_score"], 75)
            
            # Simulate on-the-fly option update
            live.SESSIONS["job_test_live"] = sess
            try:
                from studio import ctx
                with ctx.as_user("u_local"):
                    updated = live.update_session_options("job_test_live", {"style": "mrbeast", "layout": "stack", "av_offset": 4.0})
                    self.assertEqual(sess.settings["style"], "mrbeast")
                    self.assertEqual(sess.settings["layout"], "stack")
                    self.assertEqual(sess.settings["av_offset"], 4.0)
                    self.assertEqual(sess.job["options"]["style"], "mrbeast")
                    self.assertEqual(sess.job["options"]["av_offset"], 4.0)
            finally:
                live.SESSIONS.pop("job_test_live", None)

    def test_av_offset_in_request_render(self):
        """Verify that request_render captures and validates av_offset."""
        dummy_job = {
            "id": "j_test_av",
            "dir": ".",
            "clips": [{"id": "clip001", "start": 10.0, "end": 25.0, "status": "done"}],
            "status": "done",
            "user": "u_local"
        }
        jobs._jobs["j_test_av"] = dummy_job
        try:
            from studio import ctx
            with ctx.as_user("u_local"):
                jobs.request_render("j_test_av", "clip001", {"av_offset": 4.4, "layout": "blur"})
                cl = dummy_job["clips"][0]
                self.assertEqual(cl.get("av_offset"), 4.4)
                self.assertEqual(cl.get("layout"), "blur")
        finally:
            jobs._jobs.pop("j_test_av", None)


if __name__ == "__main__":
    unittest.main()
