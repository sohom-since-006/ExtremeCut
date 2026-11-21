import cv2, subprocess
from pathlib import Path

mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.mp4")

# Let's inspect subtitles on the video in GTA!
# Does GTA 6 have IN-GAME SUBTITLES?!
# GTA 6 shows in-game subtitles at the bottom of the game!
# Let's crop the bottom of the game area in clip002.mp4 to see if GTA in-game subtitles are visible!
game_sub_dir = Path("gta_in_game_subs")
game_sub_dir.mkdir(exist_ok=True)

# In layout blur: fg_h = 606, top = (1920-606)/2 = 657
# The in-game GTA subtitle is at the bottom of the game frame (y = 657 + 606 - 80 = 1180 to 1260)
subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vf", "fps=2,crop=1080:150:0:1150", str(game_sub_dir / "sub_%03d.jpg")], capture_output=True)

print(f"Extracted {len(list(game_sub_dir.glob('*.jpg')))} game sub frames.")
