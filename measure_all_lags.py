import subprocess, numpy as np, cv2
from pathlib import Path

def measure_clip_lag(src_mp4):
    wav = Path("temp_test.wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(src_mp4), "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)
    audio_data = np.fromfile(wav, dtype=np.int16, offset=44).astype(np.float32)
    wav.unlink(missing_ok=True)

    win = 800
    n_wins = len(audio_data) // win
    audio_energy = np.array([np.sqrt(np.mean(audio_data[i*win:(i+1)*win]**2)) for i in range(n_wins)])

    face_dir = Path("face_temp")
    face_dir.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(src_mp4), "-vf", "fps=20,crop=100:150:70:300", str(face_dir / "f_%04d.jpg")], capture_output=True)

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
    if min_len < 100:
        return None
    a_norm = (audio_energy[:min_len] - np.mean(audio_energy[:min_len])) / (np.std(audio_energy[:min_len]) + 1e-6)
    m_norm = (motion[:min_len] - np.mean(motion[:min_len])) / (np.std(motion[:min_len]) + 1e-6)
    corr = np.correlate(m_norm, a_norm, mode='full')
    lags = np.arange(-min_len + 1, min_len) * 0.05
    # Look around 0 to 8s
    valid = [(lags[i], corr[i]) for i in range(len(lags)) if 2.0 <= lags[i] <= 7.0]
    if valid:
        best = max(valid, key=lambda x: x[1])
        return best[0]
    return None

job_dir = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips")
for c in ["clip001", "clip002", "clip003", "clip004"]:
    f = job_dir / f"{c}.src.mp4"
    if f.exists():
        lag = measure_clip_lag(f)
        print(f"{c}: estimated lag = {lag} seconds")
