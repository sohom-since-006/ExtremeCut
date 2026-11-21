import subprocess, numpy as np
from pathlib import Path

mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.mp4")

# Extract audio as wav
wav = Path("test_clip002_full.wav")
subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)

# Load audio samples
data = np.fromfile(wav, dtype=np.int16, offset=44).astype(np.float32)
wav.unlink(missing_ok=True)

# Find top 5 loud audio spikes/peaks in clip002
# Window of 100ms (1600 samples)
win = 1600
energy = np.array([np.sqrt(np.mean(data[i:i+win]**2)) for i in range(0, len(data) - win, win)])
spike_times = np.argsort(energy)[-8:] * (win / 16000.0)
spike_times = sorted(spike_times)

print("Audio energy spikes in clip002.mp4 (seconds):")
for t in spike_times:
    print(f"  Spike at {t:.2f}s (energy = {energy[int(t*10)]:.1f})")
