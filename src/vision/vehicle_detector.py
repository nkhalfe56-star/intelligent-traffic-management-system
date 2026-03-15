"""
YOLOv8-based Vehicle Detection Module
Capstone Research: Intelligent Traffic Management System
Author: nkhalfe56-star | IIT Jodhpur

Detects and classifies vehicles in traffic camera frames.
Supports real-time streaming and batch processing.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# Vehicle class mapping (COCO classes relevant to traffic)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    speed_kmh: Optional[float] = None


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    vehicle_count: Dict[str, int] = field(default_factory=dict)
    total_vehicles: int = 0
    lane_counts: Dict[int, int] = field(default_factory=dict)
    lane_density: Dict[int, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class VehicleDetector:
    """
    Real-time vehicle detection using YOLOv8.

    Supports:
    - Multi-class vehicle detection (car, bus, truck, motorcycle, bicycle)
    - Lane-wise vehicle counting
    - Occupancy and density estimation
    - Optional ByteTrack multi-object tracking
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "auto",
        enable_tracking: bool = True,
        lane_boundaries: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Args:
            model_path: Path to YOLOv8 weights (.pt file)
            conf_threshold: Minimum detection confidence
            iou_threshold: NMS IoU threshold
            device: 'auto', 'cpu', 'cuda', or 'mps'
            enable_tracking: Enable ByteTrack for vehicle tracking
            lane_boundaries: List of x-coordinates defining lane boundaries
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_tracking = enable_tracking
        self.lane_boundaries = lane_boundaries or []
        self.frame_count = 0

        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLOv8 model from {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info("YOLOv8 model loaded successfully.")

        # Previous frame for optical flow speed estimation
        self._prev_gray: Optional[np.ndarray] = None
        self._track_positions: Dict[int, Tuple[float, float]] = {}

    def detect(self, frame: np.ndarray) -> FrameResult:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image array (H, W, 3)

        Returns:
            FrameResult with detections and per-lane statistics
        """
        t0 = time.perf_counter()
        self.frame_count += 1

        if self.enable_tracking:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )

        detections = []
        vehicle_count: Dict[str, int] = {v: 0 for v in VEHICLE_CLASSES.values()}

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                track_id = int(box.id.item()) if (self.enable_tracking and box.id is not None) else None
                cls_name = VEHICLE_CLASSES[cls_id]

                det = Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    track_id=track_id,
                )
                detections.append(det)
                vehicle_count[cls_name] += 1

        lane_counts, lane_density = self._compute_lane_statistics(
            detections, frame.shape[1]
        )

        proc_time = (time.perf_counter() - t0) * 1000

        return FrameResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections=detections,
            vehicle_count=vehicle_count,
            total_vehicles=len(detections),
            lane_counts=lane_counts,
            lane_density=lane_density,
            processing_time_ms=proc_time,
        )

    def _compute_lane_statistics(
        self,
        detections: List[Detection],
        frame_width: int,
    ) -> Tuple[Dict[int, int], Dict[int, float]]:
        """Assign each detection to a lane and compute density."""
        if not self.lane_boundaries:
            return {0: len(detections)}, {0: min(len(detections) / 20.0, 1.0)}

        boundaries = [0] + self.lane_boundaries + [frame_width]
        n_lanes = len(boundaries) - 1
        lane_counts: Dict[int, int] = {i: 0 for i in range(n_lanes)}

        for det in detections:
            cx = (det.bbox[0] + det.bbox[2]) / 2
            for lane_idx in range(n_lanes):
                if boundaries[lane_idx] <= cx < boundaries[lane_idx + 1]:
                    lane_counts[lane_idx] += 1
                    break

        lane_density = {
            lane: min(count / 10.0, 1.0)
            for lane, count in lane_counts.items()
        }
        return lane_counts, lane_density

    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """Process a full video file and yield FrameResults."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Processing video at {fps:.1f} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.detect(frame)
            yield result

        cap.release()
        logger.info(f"Video processing complete. Frames: {self.frame_count}")

    def draw_detections(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        vis = frame.copy()
        colors = {
            "car": (0, 255, 0),
            "bus": (255, 128, 0),
            "truck": (0, 128, 255),
            "motorcycle": (255, 0, 255),
            "bicycle": (128, 255, 0),
        }
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = colors.get(det.class_name, (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name}"
            if det.track_id:
                label += f" #{det.track_id}"
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Stats overlay
        cv2.putText(
            vis,
            f"Vehicles: {result.total_vehicles} | {result.processing_time_ms:.1f}ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return vis
