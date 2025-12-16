from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


@dataclass(frozen=True)
class ModelPaths:
    vehicle_detector: str = "yolov8n.pt"
    plate_detector: str = "license_plate_detector.pt"


@dataclass(frozen=True)
class IOPaths:
    video_in: str = "./sample.mp4"
    csv_out: str = "./test.csv"


@dataclass(frozen=True)
class DetectorConfig:
    # COCO ids: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_class_ids: Tuple[int, ...] = (2, 3, 5, 7)
    # plate crop preprocessing
    thresh_value: int = 64
    thresh_max: int = 255


class ALPRPipeline:
    # Vehicle detection -> SORT tracking -> license plate detection -> plate OCR -> CSV export.

    def __init__(
        self,
        models: ModelPaths = ModelPaths(),
        io: IOPaths = IOPaths(),
        cfg: DetectorConfig = DetectorConfig(),
    ) -> None:
        self.io = io
        self.cfg = cfg

        self.tracker = Sort()
        self.vehicle_model = YOLO(models.vehicle_detector)
        self.plate_model = YOLO(models.plate_detector)

        # results[frame_idx][car_id] = {...}
        self.results: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        cap = cv2.VideoCapture(self.io.video_in)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.io.video_in}")

        frame_idx = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                self.results[frame_idx] = {}

                dets_for_sort = self._detect_vehicles_for_sort(frame)
                tracks = self._update_tracks(dets_for_sort)
                self._process_license_plates(frame_idx, frame, tracks)

        finally:
            cap.release()

        write_csv(self.results, self.io.csv_out)
        return self.results

    def _detect_vehicles_for_sort(self, frame: np.ndarray) -> np.ndarray:
        # Returns Nx5 float array: [x1, y1, x2, y2, score] (SORT input).
        # Always returns shape (N, 5), including the empty case.
        preds = self.vehicle_model(frame)[0]

        keep: List[List[float]] = []
        for x1, y1, x2, y2, conf, cls_id in preds.boxes.data.tolist():
            if int(cls_id) in self.cfg.vehicle_class_ids:
                keep.append([float(x1), float(y1), float(x2), float(y2), float(conf)])

        if not keep:
            return np.empty((0, 5), dtype=float)
        return np.asarray(keep, dtype=float)

    def _update_tracks(self, dets: np.ndarray) -> np.ndarray:
        # Calls SORT and returns track array (whatever Sort.update returns in your repo).
        return self.tracker.update(dets)

    def _process_license_plates(self, frame_idx: int, frame: np.ndarray, tracks: np.ndarray) -> None:
        # Detect plates, associate them to tracked cars, OCR them, and store results.
        plates = self.plate_model(frame)[0]

        for px1, py1, px2, py2, pconf, _cls_id in plates.boxes.data.tolist():
            plate_box = (px1, py1, px2, py2, pconf)

            # Match this plate to a tracked vehicle
            car_x1, car_y1, car_x2, car_y2, car_id = get_car(plate_box, tracks)
            if car_id == -1:
                continue

            crop = self._crop_region(frame, px1, py1, px2, py2)
            if crop is None:
                continue

            plate_text, plate_text_score = self._ocr_plate(crop)
            if plate_text is None:
                continue

            self.results[frame_idx][car_id] = {
                "car": {"bbox": [car_x1, car_y1, car_x2, car_y2]},
                "license_plate": {
                    "bbox": [px1, py1, px2, py2],
                    "text": plate_text,
                    "bbox_score": float(pconf),
                    "text_score": plate_text_score,
                },
            }

    # ------------------------ helpers ------------------------

    def _crop_region(self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        roi = frame[iy1:iy2, ix1:ix2, :]
        return None if roi.size == 0 else roi

    def _ocr_plate(self, plate_bgr: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        # Applies the same grayscale + binary inverse threshold preprocessing,
        # then delegates OCR to util.read_license_plate().
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_inv = cv2.threshold(
            gray,
            self.cfg.thresh_value,
            self.cfg.thresh_max,
            cv2.THRESH_BINARY_INV,
        )
        return read_license_plate(binary_inv)


if __name__ == "__main__":
    ALPRPipeline().run()
