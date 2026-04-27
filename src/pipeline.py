from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

from .config import (
    AREA_WEIGHT,
    BOX_X_MARGIN_RATIO,
    BOX_Y_START_RATIO,
    DEPTH_MODEL,
    DEPTH_WEIGHT,
    LEFT_RIGHT_BINS,
    MEDIUM_THRESHOLD,
    NEAR_THRESHOLD,
    OUTPUT_DIR,
    TARGET_CLASSES,
    YOLO_MODEL,
)
from .models import DetectedObstacle

class HaptiGuidePipeline:
    def __init__(self, yolo_model: str = YOLO_MODEL, conf: float = 0.30):
        self.conf = conf
        self.detector = YOLO(yolo_model)
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
        )

    def estimate_depth(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = self.depth_pipe(pil_image)

        depth_pil = result["depth"]
        depth = np.array(depth_pil).astype(np.float32)
        depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

        depth = depth - depth.min()
        if depth.max() > 0:
            depth = depth / depth.max()

        depth = np.power(depth, 0.85)
        return depth

    def detect_objects(self, bgr: np.ndarray):
        return self.detector.predict(source=bgr, conf=self.conf, verbose=False)

    @staticmethod
    def canonical_label(label: str) -> str:
        if label == "dining table":
            return "table"
        return label

    @staticmethod
    def get_direction(x_center: float, width: int) -> str:
        frac = x_center / width
        if frac < LEFT_RIGHT_BINS[0]:
            return "left"
        if frac < LEFT_RIGHT_BINS[1]:
            return "center"
        return "right"

    @staticmethod
    def get_distance_zone(fused_score: float) -> str:
        if fused_score >= NEAR_THRESHOLD:
            return "near"
        if fused_score >= MEDIUM_THRESHOLD:
            return "medium"
        return "far"

    @staticmethod
    def get_reaction(direction: str, distance_zone: str) -> Tuple[str, Optional[str]]:
        if distance_zone == "near":
            if direction == "center":
                return "Object directly ahead - near", "level_3"
            return f"Object on the {direction} side - near", "level_3"

        if distance_zone == "medium":
            if direction == "center":
                return "Object ahead - medium distance", "level_2"
            return f"Object on the {direction} side - medium distance", "level_2"

        return f"Object on the {direction} side - far", "level_1"

    def fuse(self, bgr: np.ndarray, depth_map: np.ndarray, results) -> List[DetectedObstacle]:
        h, w = bgr.shape[:2]
        image_area = float(h * w)
        fused: List[DetectedObstacle] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                raw_label = self.detector.names[cls_id]
                if raw_label not in TARGET_CLASSES:
                    continue

                label = self.canonical_label(raw_label)
                conf = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                box_w = x2 - x1
                box_h = y2 - y1
                xs = x1 + int(box_w * BOX_X_MARGIN_RATIO)
                xe = x2 - int(box_w * BOX_X_MARGIN_RATIO)
                ys = y1 + int(box_h * BOX_Y_START_RATIO)
                ye = y2

                xs = max(0, xs)
                xe = min(w, xe)
                ys = max(0, ys)
                ye = min(h, ye)

                crop = depth_map[ys:ye, xs:xe]
                if crop.size == 0:
                    continue

                depth_score = float(np.median(crop))
                area_ratio = (box_w * box_h) / image_area
                area_score = min(1.0, np.sqrt(area_ratio / 0.12))
                fused_score = (DEPTH_WEIGHT * depth_score) + (AREA_WEIGHT * area_score)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                direction = self.get_direction(cx, w)
                distance_zone = self.get_distance_zone(fused_score)
                reaction_text, sound_level = self.get_reaction(direction, distance_zone)

                fused.append(
                    DetectedObstacle(
                        label=label,
                        confidence=conf,
                        bbox_xyxy=(x1, y1, x2, y2),
                        center_x=cx,
                        center_y=cy,
                        depth_score=depth_score,
                        area_score=area_score,
                        fused_score=fused_score,
                        distance_zone=distance_zone,
                        direction=direction,
                        reaction_text=reaction_text,
                        sound_level=sound_level,
                    )
                )

        fused.sort(key=lambda x: x.fused_score, reverse=True)
        return fused

    def annotate(self, bgr: np.ndarray, obstacles: List[DetectedObstacle]) -> np.ndarray:
        out = bgr.copy()

        for obs in obstacles:
            x1, y1, x2, y2 = obs.bbox_xyxy

            if obs.distance_zone == "near":
                color = (0, 0, 255)
            elif obs.distance_zone == "medium":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            text = f"{obs.label} | {obs.direction} | {obs.distance_zone} | s={obs.fused_score:.2f}"
            cv2.putText(
                out,
                text,
                (x1, max(28, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        return out

    def analyze_image(self, image_path: str):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        depth_map = self.estimate_depth(bgr)
        detections = self.detect_objects(bgr)
        obstacles = self.fuse(bgr, depth_map, detections)
        annotated = self.annotate(bgr, obstacles)

        stem = Path(image_path).stem
        annotated_path = OUTPUT_DIR / f"{stem}_annotated.jpg"
        proximity_path = OUTPUT_DIR / f"{stem}_proximity.png"

        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(proximity_path), (depth_map * 255).astype(np.uint8))
        return obstacles, annotated_path, proximity_path
