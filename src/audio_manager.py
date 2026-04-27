import time
from pathlib import Path
from typing import Optional

import pygame

from .config import AUDIO_COOLDOWN_SECONDS, LEVEL_SOUND_MAP, LEVEL3_VOICE_MAP

class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.last_played_key: Optional[str] = None
        self.last_played_time: float = 0.0

    def stop_audio(self) -> None:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def _can_play(self, key: str) -> bool:
        now = time.time()
        if key == self.last_played_key and (now - self.last_played_time) < AUDIO_COOLDOWN_SECONDS:
            return False
        self.last_played_key = key
        self.last_played_time = now
        return True

    def _play_file(self, sound_path: Path, unique_key: str) -> None:
        if not sound_path.exists():
            return
        if not self._can_play(unique_key):
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(str(sound_path))
            pygame.mixer.music.play()
        except Exception:
            pass

    def play_level_sound(self, sound_key: Optional[str]) -> None:
        if sound_key is None:
            return
        sound_path = LEVEL_SOUND_MAP.get(sound_key)
        if sound_path is None:
            return
        self._play_file(sound_path, unique_key=sound_key)

    def play_level3_voice(self, label: str, direction: str) -> None:
        sound_path = LEVEL3_VOICE_MAP.get((label, direction))
        if sound_path is None:
            return
        self._play_file(sound_path, unique_key=f"voice::{label}::{direction}")
