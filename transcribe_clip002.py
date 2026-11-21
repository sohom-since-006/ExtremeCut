from pathlib import Path
from studio import transcribe, media

mp4 = Path("data/users/u_local/creators/ishowspeed/live/20261120-233057-52ff30009f/clips/clip002.mp4")
wav = Path("test_clip002.wav")
media.extract_audio(mp4, wav)
audio = media.load_audio(wav)
words = transcribe.transcribe(audio)
wav.unlink(missing_ok=True)

print("=== TRANSCRIBED FROM RENDERED clip002.mp4 ===")
for w in words:
    print(f"  {w['s']:.2f} - {w['e']:.2f}: {w['w']}")
