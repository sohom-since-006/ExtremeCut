"""Tests for downloading live stream recordings and clipping recorded live streams."""
import unittest
from pathlib import Path
import tempfile
import json
import shutil

from studio import live, jobs, ctx, media

class TestClipRecording(unittest.TestCase):
    def setUp(self):
        ctx._uid.set("u_local")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "clips").mkdir(parents=True, exist_ok=True)
        (self.dir / "buffer").mkdir(parents=True, exist_ok=True)
        self.jid = "test-live-rec-01"
        self.job = {
            "id": self.jid,
            "creator": "ishowspeed",
            "title": "🔴 GTA 6 FINALE LAST MISSION",
            "uploader": "IShowSpeed",
            "kind": "live",
            "status": "done",
            "stage": "Stopped by you",
            "source": {"name": "IShowSpeed Live", "url": "https://youtube.com/@IShowSpeed/live"},
            "clips": [],
            "dir": str(self.dir),
            "user": "u_local"
        }
        with jobs._lock:
            jobs._jobs[self.jid] = self.job
            jobs._by_user.setdefault("u_local", {})[self.jid] = self.job

    def tearDown(self):
        with jobs._lock:
            jobs._jobs.pop(self.jid, None)
            jobs._by_user.get("u_local", {}).pop(self.jid, None)
        self.tmp.cleanup()

    def test_ensure_recording_mp4(self):
        # Create a fake dummy recording.mp4
        rec = self.dir / "recording.mp4"
        rec.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100)
        found = live.ensure_recording_mp4(self.job)
        self.assertIsNotNone(found)
        self.assertEqual(found, rec)

    def test_clip_recording_creates_new_video_job(self):
        # Create fake recording.mp4
        rec = self.dir / "recording.mp4"
        rec.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100)
        thumb = self.dir / "thumb.jpg"
        thumb.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        words = [{"w": "insane", "s": 1.0, "e": 1.5}]
        (self.dir / "words.json").write_text(json.dumps(words), encoding="utf-8")

        from studio.server import clip_recording
        res = clip_recording(self.jid, body={"options": {"style": "neon", "layout": "bottom"}})
        
        self.assertIn("id", res)
        new_id = res["id"]
        new_job = jobs.get(new_id)
        self.assertIsNotNone(new_job)
        self.assertEqual(new_job["kind"], "video")
        self.assertEqual(new_job["options"]["style"], "neon")
        self.assertEqual(new_job["options"]["layout"], "bottom")
        self.assertTrue(Path(new_job["source"]["path"]).is_file())
        
        # Cleanup newly created job
        jobs.delete(new_id)

if __name__ == "__main__":
    unittest.main()
