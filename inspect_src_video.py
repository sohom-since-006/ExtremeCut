import cv2, subprocess
from pathlib import Path

src_mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.src.mp4")
out_dir = Path("full_clip002_src_frames")
out_dir.mkdir(exist_ok=True)

# Extract 1 frame per second across the whole 54s of src
subprocess.run(["ffmpeg", "-y", "-i", str(src_mp4), "-vf", "fps=1", str(out_dir / "sec_%02d.jpg")], capture_output=True)

# Also let's check facecam! Does Speed's facecam exist in the corner or somewhere?
# Let's inspect where Speed's webcam is in the frames!
print(f"Extracted {len(list(out_dir.glob('*.jpg')))} frames.")
