from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DetectedObstacle:
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    depth_score: float
    area_score: float
    fused_score: float
    distance_zone: str
    direction: str
    reaction_text: str
    sound_level: Optional[str]
