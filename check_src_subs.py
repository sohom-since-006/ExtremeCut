import subprocess
from pathlib import Path

# Let's inspect uncropped clip002.src.mp4
src = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.src.mp4")
out = Path("src_subs")
out.mkdir(exist_ok=True)

# Extract 1 frame per second from src
subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", "fps=1", str(out / "f_%02d.jpg")], capture_output=True)

# Let's check OCR or check image text with pytesseract or simple thresholding
# Or check if any text is readable
print(f"Extracted {len(list(out.glob('*.jpg')))} frames from src.")
