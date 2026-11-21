"""Tests the visual framing endpoints: frame snapshot, auto face detection, and remake with regions."""
import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starlette.testclient import TestClient

# Use temporary data dir
tmp_dir = tempfile.mkdtemp(prefix="ec_test_framing_")
os.environ["EXTREMECUT_DATA"] = tmp_dir
os.environ["EXTREMECUT_ACCOUNTS"] = "0"

from studio import server, jobs, auth, config, ctx, db

db.init()
client = TestClient(server.app, base_url="http://127.0.0.1:8765")

def test_framing_workflow():
    cid = "c1"
    u = auth.local_user()
    uid = u.id if hasattr(u, "id") else u[0]
    with ctx.as_user(uid):
        # 1. Create a dummy job with a clip
        job = jobs.create(
            {"type": "file", "path": "", "name": "test_gameplay.mp4"},
            {"style": "bounce", "layout": "stack", "quality": 1080},
            enqueue=False
        )
        jid = job["id"]
        job_dir = Path(job["dir"])
        clips_dir = job_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Put a mock source frame and a mock clip cover
        src_frame = clips_dir / "c1_src.jpg"
        src_frame.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9")
        
        clip_cover = clips_dir / "c1.jpg"
        clip_cover.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 150 + b"\xff\xd9")

        clip = {
            "id": "c1",
            "start": 0.0,
            "end": 15.0,
            "file": "c1.mp4",
            "cover": "c1.jpg",
            "hook": "UNBELIEVABLE PLAY",
            "title": "Epic Gameplay Moment",
            "description": "Watch this clip!",
            "style": "bounce",
            "layout": "stack",
            "score": 95,
            "status": "done",
            "version": 1
        }
        job["status"] = "done"
        job["clips"] = [clip]
        jobs.save(job)

    # Test GET /api/jobs/{jid}/clips/{cid}/frame
    res = client.get(f"/api/jobs/{jid}/clips/c1/frame")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    assert res.headers["content-type"] == "image/jpeg"
    print("PASS /api/jobs/{jid}/clips/{cid}/frame returns source frame")

    # Test POST /api/jobs/{jid}/clips/{cid}/autoface
    res_af = client.post(f"/api/jobs/{jid}/clips/c1/autoface")
    assert res_af.status_code == 200
    data_af = res_af.json()
    assert data_af.get("ok") is True
    assert "cam" in data_af
    cam = data_af["cam"]
    assert "x" in cam and "y" in cam and "w" in cam and "h" in cam
    print(f"PASS /api/jobs/{jid}/clips/{cid}/autoface returns box: {cam}")

    # Test POST /api/jobs/{jid}/clips/{cid}/render with custom cam and game (testing 'bottom' layout)
    remake_payload = {
        "hook": "NEW HOOK TITLE",
        "style": "hormozi",
        "layout": "bottom",
        "top": 0.35,
        "cam": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.4},
        "game": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "smart_zoom": True,
        "start": 1.0,
        "end": 14.0
    }
    res_rm = client.post(f"/api/jobs/{jid}/clips/c1/render", json=remake_payload)
    assert res_rm.status_code == 200, f"Expected 200, got {res_rm.status_code}: {res_rm.text}"
    with ctx.as_user(uid):
        updated_job = jobs.get(jid)
        c1 = next(c for c in updated_job["clips"] if c["id"] == "c1")
        assert c1["hook"] == "NEW HOOK TITLE"
        assert c1["style"] == "hormozi"
        assert c1["layout"] == "bottom"
        assert c1["top"] == 0.35
        assert c1["cam"] == {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.4}
        assert c1["game"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        assert c1["smart_zoom"] is True
        assert c1["status"] == "queued"
    print("PASS /api/jobs/{jid}/clips/{cid}/render persists bottom layout, cam, game, top and queues render")

    # Test /api/framing/detect-face with synthetic frame
    import base64
    import numpy as np
    import cv2
    dummy_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy_arr)
    b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")
    res_df = client.post("/api/framing/detect-face", json={"image": b64})
    assert res_df.status_code == 200, f"Expected 200, got {res_df.status_code}: {res_df.text}"
    df_data = res_df.json()
    assert df_data.get("ok") is True
    assert "cam" in df_data
    assert "x" in df_data["cam"] and "w" in df_data["cam"]
    print("PASS /api/framing/detect-face accepts base64 frame and returns cam region")

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("ALL FRAMING ENDPOINT TESTS PASSED")

if __name__ == "__main__":
    test_framing_workflow()
