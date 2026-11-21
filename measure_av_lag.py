import cv2, subprocess, numpy as np
from pathlib import Path

src_mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.src.mp4")

# Extract audio volume in 50ms windows
wav = Path("test_face_audio.wav")
subprocess.run(["ffmpeg", "-y", "-i", str(src_mp4), "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)
audio_data = np.fromfile(wav, dtype=np.int16, offset=44).astype(np.float32)
wav.unlink(missing_ok=True)

# Audio energy in 50ms windows (800 samples)
win = 800
n_wins = len(audio_data) // win
audio_energy = np.array([np.sqrt(np.mean(audio_data[i*win:(i+1)*win]**2)) for i in range(n_wins)])
audio_times = np.arange(n_wins) * 0.05

# Extract facecam crop frames at 20 fps (every 50ms)
face_dir = Path("face_frames")
face_dir.mkdir(exist_ok=True)

# Facecam box: x=0.0617*1280=79, y=0.4268*720=307, w=85, h=127
subprocess.run(["ffmpeg", "-y", "-i", str(src_mp4), "-vf", "fps=20,crop=100:150:70:300", str(face_dir / "f_%04d.jpg")], capture_output=True)

frames = sorted(face_dir.glob("f_*.jpg"))
print(f"Total face frames at 20fps: {len(frames)}")

motion = []
for i in range(1, len(frames)):
    im1 = cv2.imread(str(frames[i-1]), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
    if im1 is not None and im2 is not None:
        diff = np.mean(np.abs(im1.astype(float) - im2.astype(float)))
        motion.append(diff)
    else:
        motion.append(0.0)

motion = np.array(motion)
motion_times = np.arange(len(motion)) * 0.05

# Let's cross-correlate audio_energy with face motion!
# If audio leads video or video leads audio, cross-correlation peak shows the exact shift!
min_len = min(len(audio_energy), len(motion))
a_sig = audio_energy[:min_len]
m_sig = motion[:min_len]

# Normalize signals
a_norm = (a_sig - np.mean(a_sig)) / (np.std(a_sig) + 1e-6)
m_norm = (m_sig - np.mean(m_sig)) / (np.std(m_sig) + 1e-6)

# Cross correlation
corr = np.correlate(m_norm, a_norm, mode='full')
lags = np.arange(-min_len + 1, min_len) * 0.05

best_lag_idx = np.argmax(corr)
best_lag = lags[best_lag_idx]

print(f"\n==========================================")
print(f"CROSS-CORRELATION PEAK LAG: {best_lag:.3f} seconds")
print(f"Correlation value: {corr[best_lag_idx] / min_len:.3f}")
print(f"==========================================")

# Also let's print top 3 lag peaks
sorted_indices = np.argsort(corr)[::-1]
print("Top 5 candidate lags (seconds):")
seen = []
for idx in sorted_indices:
    l = lags[idx]
    if not any(abs(l - s) < 0.5 for s in seen):
        seen.append(l)
        print(f"  Lag: {l:+.2f}s (corr: {corr[idx]/min_len:.3f})")
    if len(seen) >= 5:
        break
