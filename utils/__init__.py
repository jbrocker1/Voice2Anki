"""Voice2Anki package root.

Its one job is ordering: pydub picks its ffmpeg binary at *import* time, so the
bootstrap has to run before any submodule does ``from pydub import AudioSegment``.
Putting it here makes that automatic for every ``utils.*`` import.

Scripts run as standalone files from inside this directory (see
``check_done_audio.py``) do not go through this module, so they import
``ffmpeg_bootstrap`` directly instead.
"""
from .ffmpeg_bootstrap import ensure_ffmpeg

ensure_ffmpeg()