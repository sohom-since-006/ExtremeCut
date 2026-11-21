import cv2, numpy as np
from pathlib import Path

frames = sorted(Path("frames_src_spike").glob("s_*.jpg"))
diffs = []
for i in range(1, len(frames)):
    im1 = cv2.imread(str(frames[i-1]))
    im2 = cv2.imread(str(frames[i]))
    diff = np.mean(np.abs(im1.astype(float) - im2.astype(float)))
    t = 28.0 + i * 0.2
    diffs.append((t, diff, frames[i].name))

# Sort by largest visual change/cut
diffs.sort(key=lambda x: x[1], reverse=True)
print("Top 5 visual scene changes in source (28s to 35s):")
for t, d, name in diffs[:5]:
    print(f"  Visual change at {t:.2f}s (diff = {d:.1f}) in {name}")
