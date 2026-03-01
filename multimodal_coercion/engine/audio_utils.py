from pathlib import Path
import subprocess
import tempfile


def extract_audio_ffmpeg(video_path: str, sample_rate: int = 16000) -> str:
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(str(vp))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(vp),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode("utf-8", errors="ignore"))
    return tmp_path

