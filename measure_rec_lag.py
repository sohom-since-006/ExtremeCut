import subprocess, numpy as np, cv2
from pathlib import Path

rec = Path("data/users/u_local/creators/ishowspeed/live/20261120-233825-2bc47d5387/recording.mp4")

# Extract 60 seconds of audio from 30s to 90s
wav = Path("rec_test.wav")
subprocess.run(["ffmpeg", "-y", "-ss", "30.0", "-t", "60.0", "-i", str(rec), "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)
audio_data = np.fromfile(wav, dtype=np.int16, offset=44).astype(np.float32)
wav.unlink(missing_ok=True)

win = 800
n_wins = len(audio_data) // win
audio_energy = np.array([np.sqrt(np.mean(audio_data[i*win:(i+1)*win]**2)) for i in range(n_wins)])

face_dir = Path("face_rec")
face_dir.mkdir(exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-ss", "30.0", "-t", "60.0", "-i", str(rec), "-vf", "fps=20,crop=100:150:70:300", str(face_dir / "f_%04d.jpg")], capture_output=True)

frames = sorted(face_dir.glob("f_*.jpg"))
motion = []
for i in range(1, len(frames)):
    im1 = cv2.imread(str(frames[i-1]), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
    if im1 is not None and im2 is not None:
        motion.append(np.mean(np.abs(im1.astype(float) - im2.astype(float))))
    else:
        motion.append(0.0)
for f in frames:
    f.unlink()
face_dir.rmdir()

motion = np.array(motion)
min_len = min(len(audio_energy), len(motion))
a_norm = (audio_energy[:min_len] - np.mean(audio_energy[:min_len])) / (np.std(audio_energy[:min_len]) + 1e-6)
m_norm = (motion[:min_len] - np.mean(motion[:min_len])) / (np.std(motion[:min_len]) + 1e-6)
corr = np.correlate(m_norm, a_norm, mode='full')
lags = np.arange(-min_len + 1, min_len) * 0.05

best_lag_idx = np.argmax(corr)
print(f"recording.mp4 peak lag: {lags[best_lag_idx]:.3f}s (corr = {corr[best_lag_idx]/min_len:.3f})")
for i in range(len(lags)):
    if 2.0 <= lags[i] <= 6.0 and corr[i] == max(corr[max(0, i-10):min(len(corr), i+10)]):
        print(f"  Candidate: {lags[i]:.2f}s (corr = {corr[i]/min_len:.3f})")
