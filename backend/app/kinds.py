"""What a user can upload: a picture with sound, or sound alone."""

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"}
ALLOWED = VIDEO_SUFFIXES | AUDIO_SUFFIXES


def is_audio(suffix: str) -> bool:
    return suffix.lower() in AUDIO_SUFFIXES
