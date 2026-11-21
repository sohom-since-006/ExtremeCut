import subprocess
from pathlib import Path
import cv2

mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.mp4")

# Let's extract frames between 20.0s and 27.0s every 0.2s
out_dir = Path("frames_spike_21")
out_dir.mkdir(exist_ok=True)

subprocess.run(["ffmpeg", "-y", "-ss", "20.0", "-t", "7.0", "-i", str(mp4), "-vf", "fps=5", str(out_dir / "f_%03d.jpg")], capture_output=True)

# Also let's extract uncropped source frames around that time from clip002.src.mp4
# clip002 start was 81.03, src_offset was 72.0 -> local start is 9.03.
# So 21.4s in clip is (9.03 + 21.4) = 30.43s in src!
src_mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.src.mp4")
src_spike_dir = Path("frames_src_spike")
src_spike_dir.mkdir(exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-ss", "28.0", "-t", "7.0", "-i", str(src_mp4), "-vf", "fps=5", str(src_spike_dir / "s_%03d.jpg")], capture_output=True)

print("Extracted spike frames.")
