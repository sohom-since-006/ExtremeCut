import subprocess
from pathlib import Path

mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.mp4")
out_dir = Path("frames_clip002")
out_dir.mkdir(exist_ok=True)

# Extract frames every 1 second
subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vf", "fps=1", str(out_dir / "f_%02d.jpg")], capture_output=True)
print(f"Extracted {len(list(out_dir.glob('*.jpg')))} frames.")
