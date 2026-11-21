import json, re, numpy as np
from pathlib import Path
from studio import transcribe, media

job_dir = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f")
words = json.loads((job_dir / "words.json").read_text("utf-8"))
clip002_src = job_dir / "clips" / "clip002.src.mp4"

guess = 72.0
dur = 54.0

wav = clip002_src.with_suffix(".wav")
media.extract_audio(clip002_src, wav)
heard = transcribe.transcribe(media.load_audio(wav))
wav.unlink(missing_ok=True)

key = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
near = [w for w in words if guess - 15 <= w["s"] <= guess + dur + 15]

diffs = []
for h in heard:
    k = key(h["w"])
    if len(k) < 3:
        continue
    m = [w["s"] for w in near if key(w["w"]) == k and abs(w["s"] - (guess + h["s"])) < 8]
    if m:
        diffs.append(min(m, key=lambda s: abs(s - (guess + h["s"]))) - h["s"])

print("Number of matched words:", len(diffs))
if diffs:
    print(f"Median offset found by _align: {float(np.median(diffs)):.2f}s")
    print(f"Original guess was: {guess:.2f}s")
    print(f"DIFFERENCE (DRIFT/LAG): {float(np.median(diffs)) - guess:.2f}s")
