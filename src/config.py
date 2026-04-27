from pathlib import Path

APP_TITLE = "HaptiGuide3D Pro"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SOUNDS_DIR = BASE_DIR / "sounds"

TARGET_CLASSES = {"chair", "dining table", "table"}
LEFT_RIGHT_BINS = [0.33, 0.66]

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 405

INFER_EVERY_N_FRAMES = 8
DEPTH_EVERY_N_FRAMES = 10

AUDIO_COOLDOWN_SECONDS = 1.0

DEPTH_WEIGHT = 0.68
AREA_WEIGHT = 0.32
MEDIUM_THRESHOLD = 0.28
NEAR_THRESHOLD = 0.50

BOX_Y_START_RATIO = 0.35
BOX_X_MARGIN_RATIO = 0.10

YOLO_MODEL = "yolov8n.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

LEVEL_SOUND_MAP = {
    "level_1": SOUNDS_DIR / "1.mp3",
    "level_2": SOUNDS_DIR / "2.mp3",
}

LEVEL3_VOICE_MAP = {
    ("chair", "center"): SOUNDS_DIR / "Chair in front of you.mp3",
    ("chair", "left"): SOUNDS_DIR / "Chair on your left.mp3",
    ("chair", "right"): SOUNDS_DIR / "Chair on your right.mp3",
    ("table", "center"): SOUNDS_DIR / "Table in front of you.mp3",
    ("table", "left"): SOUNDS_DIR / "Table on your left.mp3",
    ("table", "right"): SOUNDS_DIR / "Table on your right.mp3",
    ("dining table", "center"): SOUNDS_DIR / "Table in front of you.mp3",
    ("dining table", "left"): SOUNDS_DIR / "Table on your left.mp3",
    ("dining table", "right"): SOUNDS_DIR / "Table on your right.mp3",
}
