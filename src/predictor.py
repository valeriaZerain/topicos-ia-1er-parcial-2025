import math
from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:

    matched_box = None
    (x1, y1), (x2, y2) = segment
    min_distance = float('inf')
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

    for bbox in bboxes:
        dist = distance_point_to_bbox(mid_x, mid_y, bbox)
        if dist < min_distance and dist <= max_distance:
            min_distance = dist
            matched_box = bbox

    return matched_box

def distance_point_to_bbox(px: float, py: float, box: list[int]) -> float:
    x_min, y_min, x_max, y_max = box
    dx = max(x_min - px, 0, px - x_max)
    dy = max(y_min - py, 0, py - y_max)
    return math.hypot(dx, dy)

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    image = image_array.copy()
    for seg_mask, category, bbox in zip(segmentation.masks, segmentation.categories, segmentation.bboxes):
        if category == 'danger':
            color = np.array([0, 0, 255], dtype=np.uint8)
        elif category == 'safe':
            color = np.array([0, 255, 0], dtype=np.uint8)
        else:
            continue 
        mask = seg_mask.astype(bool)
        image[mask] = cv2.addWeighted(image, 0.5, color, 0.5, 0)[mask]
        if draw_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 3)
            cv2.putText(image, category, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color.tolist(), 2)
    return image


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        ### ========================== ###
        ### SU IMPLEMENTACION AQUI     ###
        ### ========================== ###

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=0,
            polygons=[],
            boxes=[],
            labels=[]
        )
