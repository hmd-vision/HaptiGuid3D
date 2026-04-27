from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None


@dataclass
class GestureDebug:
    gesture: Optional[str]
    confidence: float = 0.0
    handedness: str = "Unknown"
    extended_count: int = 0
    thumb_extended: bool = False
    index_extended: bool = False
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False
    index_tip_x: float = 0.5
    index_tip_y: float = 0.5
    focus_zone: Optional[str] = None
    note: str = ""


class HRIGestureControl:
    def __init__(self, stable_frames: int = 3, cooldown_frames: int = 10):
        if mp is None:
            raise ImportError("MediaPipe is required. Use Python 3.11 and install requirements.txt") from _MP_IMPORT_ERROR

        self.active = False
        self.stable_frames = max(1, stable_frames)
        self.cooldown_frames = max(0, cooldown_frames)
        self.cooldown = 0
        self.history = deque(maxlen=self.stable_frames)
        self.last_debug = GestureDebug(gesture=None)

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self._last_landmarks = None

    def reset(self):
        self.active = False
        self.cooldown = 0
        self.history.clear()
        self.last_debug = GestureDebug(gesture=None)
        self._last_landmarks = None

    @staticmethod
    def _finger_extended(lm, tip: int, pip: int, mcp: int) -> bool:
        return lm[tip].y < lm[pip].y - 0.018 and lm[pip].y < lm[mcp].y + 0.035

    @staticmethod
    def _thumb_extended(lm) -> bool:
        tip, ip, mcp = lm[4], lm[3], lm[2]
        return abs(tip.x - mcp.x) > 0.055 and tip.y < lm[17].y + 0.10 and abs(tip.x - ip.x) > 0.025

    @staticmethod
    def _zone_from_x(x: float) -> str:
        if x < 0.33:
            return "left"
        if x < 0.66:
            return "center"
        return "right"

    def _classify_landmarks(self, hand_landmarks, handedness_label: str, score: float) -> GestureDebug:
        lm = hand_landmarks.landmark
        index_extended = self._finger_extended(lm, 8, 6, 5)
        middle_extended = self._finger_extended(lm, 12, 10, 9)
        ring_extended = self._finger_extended(lm, 16, 14, 13)
        pinky_extended = self._finger_extended(lm, 20, 18, 17)
        thumb_extended = self._thumb_extended(lm)

        extended_no_thumb = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        extended_count = extended_no_thumb + int(thumb_extended)
        focus_zone = self._zone_from_x(float(lm[8].x))

        gesture = None
        note = ""
        confidence = min(0.99, max(0.0, float(score)))

        if extended_count == 0:
            gesture, note, confidence = "FIST_EMERGENCY_STOP", "No fingers extended", max(confidence, 0.90)
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture, note, confidence = "ONE_FINGER_FOCUS", f"Index finger focus zone: {focus_zone}", max(confidence, 0.91)
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture, note, confidence = "VICTORY_MUTE", "Index and middle extended", max(confidence, 0.92)
        elif extended_no_thumb >= 4:
            gesture, note, confidence = "OPEN_PALM", "Open palm / all regions", max(confidence, 0.93)
        elif thumb_extended and extended_no_thumb == 0:
            gesture, note, confidence = "THUMBS_UP", "Thumb extended, other fingers folded", max(confidence, 0.90)

        return GestureDebug(
            gesture=gesture,
            confidence=confidence if gesture else 0.0,
            handedness=handedness_label,
            extended_count=extended_count,
            thumb_extended=thumb_extended,
            index_extended=index_extended,
            middle_extended=middle_extended,
            ring_extended=ring_extended,
            pinky_extended=pinky_extended,
            index_tip_x=float(lm[8].x),
            index_tip_y=float(lm[8].y),
            focus_zone=focus_zone if gesture == "ONE_FINGER_FOCUS" else None,
            note=note,
        )

    def detect_gesture(self, frame_bgr) -> GestureDebug:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._last_landmarks = None
            debug = GestureDebug(gesture=None, note="No hand detected")
            self.last_debug = debug
            return debug

        self._last_landmarks = results.multi_hand_landmarks[0]
        handedness_label, handedness_score = "Unknown", 0.8
        if results.multi_handedness:
            cls = results.multi_handedness[0].classification[0]
            handedness_label, handedness_score = cls.label, cls.score

        debug = self._classify_landmarks(results.multi_hand_landmarks[0], handedness_label, handedness_score)
        self.last_debug = debug
        return debug

    def draw_hand_overlay(self, frame_bgr):
        if self._last_landmarks is None:
            return frame_bgr
        out = frame_bgr.copy()
        self._mp_drawing.draw_landmarks(
            out,
            self._last_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style(),
        )
        return out

    def update(self, frame_bgr) -> Tuple[bool, Optional[str], GestureDebug]:
        debug = self.detect_gesture(frame_bgr)
        gesture = debug.gesture

        if self.cooldown > 0:
            self.cooldown -= 1

        self.history.append(gesture)
        stable_gesture = None
        if len(self.history) == self.stable_frames:
            candidates = ["OPEN_PALM", "THUMBS_UP", "ONE_FINGER_FOCUS", "VICTORY_MUTE", "FIST_EMERGENCY_STOP"]
            counts = {name: sum(1 for g in self.history if g == name) for name in candidates}
            best = max(counts, key=counts.get)
            if counts[best] >= max(1, self.stable_frames - 1):
                stable_gesture = best

        triggered = None
        if stable_gesture and self.cooldown == 0:
            if stable_gesture == "OPEN_PALM":
                self.active = True
                triggered = stable_gesture
            elif stable_gesture == "THUMBS_UP" and self.active:
                self.active = False
                triggered = stable_gesture
            elif stable_gesture in {"ONE_FINGER_FOCUS", "VICTORY_MUTE", "FIST_EMERGENCY_STOP"}:
                triggered = stable_gesture
                if stable_gesture == "FIST_EMERGENCY_STOP":
                    self.active = False

            if triggered:
                self.cooldown = self.cooldown_frames
                self.history.clear()

        return self.active, triggered, debug
