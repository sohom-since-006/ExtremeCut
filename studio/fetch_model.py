"""Download the speech-to-text model once, if it is not already in the models folder.
Run with:  python -m studio.fetch_model

Whisper works too. Set the engine and the size in .env first, then run this the same way:
  EXTREMECUT_ASR_ENGINE=whisper EXTREMECUT_ASR_MODEL=small.en python -m studio.fetch_model
"""
import tarfile
import urllib.request

from . import config

URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{config.ASR_MODEL_NAME}.tar.bz2"


def _whisper():
    """Whisper keeps its model in a folder of its own inside data/models, and downloads it from the model library."""
    import faster_whisper.utils
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking the Whisper model {config.ASR_MODEL_NAME}...")
    faster_whisper.utils.download_model(config.ASR_MODEL_NAME, cache_dir=str(config.MODELS_DIR))
    print("Speech model ready.")


def main():
    if config.ASR_ENGINE.startswith(("whisper", "faster")):
        _whisper()
        return
    if (config.ASR_MODEL_DIR / "tokens.txt").exists():
        print("Speech model already present.")
        return
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.MODELS_DIR / (config.ASR_MODEL_NAME + ".tar.bz2")
    print(f"Downloading the speech model {config.ASR_MODEL_NAME} (a few hundred MB)...")
    urllib.request.urlretrieve(URL, tmp)
    with tarfile.open(tmp, "r:bz2") as t:
        for m in t.getmembers():  # only plain files and folders that stay inside the models folder
            if (m.isfile() or m.isdir()) and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/"):
                t.extract(m, config.MODELS_DIR)
    tmp.unlink()
    print("Speech model ready.")


if __name__ == "__main__":
    main()
