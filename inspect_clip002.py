import json, subprocess
from pathlib import Path

job_dir = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f")
job = json.loads((job_dir / "job.json").read_text("utf-8"))
clip = next(c for c in job["clips"] if c["id"] == "clip002")

print("=== CLIP002 METADATA ===")
print("id:", clip.get("id"))
print("start:", clip.get("start"))
print("end:", clip.get("end"))
print("src_offset:", clip.get("src_offset"))
print("src_end:", clip.get("src_end"))
print("src:", clip.get("src"))
print("layout:", clip.get("layout"))

# Check words in words.json around start to end
words = json.loads((job_dir / "words.json").read_text("utf-8")) if (job_dir / "words.json").exists() else []
print(f"\nTotal words in words.json: {len(words)}")
clip_words = [w for w in words if clip["start"] - 5 <= w["s"] <= clip["end"] + 5]
print("\nWords in clip (with timestamps):")
for w in clip_words:
    print(f"  {w['s']:.2f} - {w['e']:.2f}: {w['w']}")
