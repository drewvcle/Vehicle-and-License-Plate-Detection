import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


BBox = Tuple[float, float, float, float]


def _parse_bbox(cell: Any) -> BBox:
    # Robust-ish parser for bbox strings that may look like:
    #  "[ 12  34  56  78]"  or  "[12,34,56,78]"

    s = str(cell)
    s = (
        s.replace("[ ", "[")
        .replace("   ", " ")
        .replace("  ", " ")
        .replace(" ", ",")
    )
    x1, y1, x2, y2 = ast.literal_eval(s)
    return float(x1), float(y1), float(x2), float(y2)


def _draw_corner_frame(
    img: np.ndarray,
    tl: Tuple[int, int],
    br: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    lx: int,
    ly: int,
) -> None:
    # Draws corner 'brackets' around a rectangle defined by tl/br.
    x1, y1 = tl
    x2, y2 = br

    # top-left
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)

    # bottom-left
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)

    # top-right
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)

    # bottom-right
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)


@dataclass
class PlateSnapshot:
    crop: Optional[np.ndarray]
    text: str


class VideoAnnotator:
    # Reads a results CSV that contains per-frame detections and:
    #  - draws car + plate boxes
    #  - overlays a 'best' plate crop + number above each car

    def __init__(
        self,
        results_csv: str | Path,
        video_path: str | Path,
        out_path: str | Path = "out.mp4",
        *,
        car_box_color: Tuple[int, int, int] = (0, 255, 0),
        plate_box_color: Tuple[int, int, int] = (0, 0, 255),
        car_box_thickness: int = 25,
        plate_box_thickness: int = 12,
        corner_len_x: int = 200,
        corner_len_y: int = 200,
        overlay_plate_height: int = 400,
        overlay_gap_above_car: int = 100,
        overlay_banner_height: int = 300,  # white banner above crop
        text_scale: float = 4.3,
        text_thickness: int = 17,
    ):
        self.results_csv = Path(results_csv)
        self.video_path = Path(video_path)
        self.out_path = Path(out_path)

        self.car_box_color = car_box_color
        self.plate_box_color = plate_box_color
        self.car_box_thickness = car_box_thickness
        self.plate_box_thickness = plate_box_thickness
        self.corner_len_x = corner_len_x
        self.corner_len_y = corner_len_y

        self.overlay_plate_height = overlay_plate_height
        self.overlay_gap_above_car = overlay_gap_above_car
        self.overlay_banner_height = overlay_banner_height
        self.text_scale = text_scale
        self.text_thickness = text_thickness

        self.df = pd.read_csv(self.results_csv)

        # Will be built from the video frames + df
        self.best_plate_by_car: Dict[Any, PlateSnapshot] = {}

    # -------- core pipeline --------

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        writer = self._make_writer_from_capture(cap, self.out_path)

        try:
            self._cache_best_plates(cap)
            self._render_frames(cap, writer)
        finally:
            writer.release()
            cap.release()

    # -------- setup --------

    def _make_writer_from_capture(self, cap: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # -------- caching best plate crops --------

    def _cache_best_plates(self, cap: cv2.VideoCapture) -> None:
        # For each car_id, find the row with max license_number_score, seek to that frame,
        # crop the plate, resize to fixed height, store crop + text.
        self.best_plate_by_car.clear()

        for car_id in np.unique(self.df["car_id"]):
            sub = self.df[self.df["car_id"] == car_id]
            best_score = sub["license_number_score"].max()
            best = sub[sub["license_number_score"] == best_score].iloc[0]

            best_frame = int(best["frame_nmr"])
            plate_text = str(best["license_number"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
            ok, frame = cap.read()
            if not ok:
                self.best_plate_by_car[car_id] = PlateSnapshot(None, plate_text)
                continue

            x1, y1, x2, y2 = _parse_bbox(best["license_plate_bbox"])
            crop = self._safe_crop(frame, x1, y1, x2, y2)

            if crop is None:
                self.best_plate_by_car[car_id] = PlateSnapshot(None, plate_text)
                continue

            crop = self._resize_plate_crop(crop, x1, y1, x2, y2)
            self.best_plate_by_car[car_id] = PlateSnapshot(crop, plate_text)

    def _safe_crop(
        self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = frame[iy1:iy2, ix1:ix2, :]
        if crop.size == 0:
            return None
        return crop

    def _resize_plate_crop(self, crop: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        target_h = int(self.overlay_plate_height)
        denom = max((y2 - y1), 1e-6)
        target_w = int((x2 - x1) * target_h / denom)
        target_w = max(target_w, 1)
        return cv2.resize(crop, (target_w, target_h))

    # -------- rendering --------

    def _render_frames(self, cap: cv2.VideoCapture, writer: cv2.VideoWriter) -> None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            rows = self._rows_for_frame(frame_idx)

            for r in rows:
                car_bbox = _parse_bbox(r["car_bbox"])
                plate_bbox = _parse_bbox(r["license_plate_bbox"])
                car_id = r["car_id"]

                self._draw_car_box(frame, car_bbox)
                self._draw_plate_box(frame, plate_bbox)
                self._overlay_plate_panel(frame, car_bbox, car_id)

            writer.write(frame)

    def _rows_for_frame(self, frame_idx: int) -> Iterable[Dict[str, Any]]:
        df_frame = self.df[self.df["frame_nmr"] == frame_idx]
        # to_dict avoids pandas indexing weirdness and makes iteration cleaner
        return df_frame.to_dict(orient="records")

    # -------- drawing --------

    def _draw_car_box(self, frame: np.ndarray, bbox: BBox) -> None:
        x1, y1, x2, y2 = bbox
        _draw_corner_frame(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=self.car_box_color,
            thickness=self.car_box_thickness,
            lx=self.corner_len_x,
            ly=self.corner_len_y,
        )

    def _draw_plate_box(self, frame: np.ndarray, bbox: BBox) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            self.plate_box_color,
            self.plate_box_thickness,
        )

    def _overlay_plate_panel(self, frame: np.ndarray, car_bbox: BBox, car_id: Any) -> None:
        snap = self.best_plate_by_car.get(car_id)
        if snap is None or snap.crop is None:
            return

        crop = snap.crop
        text = snap.text

        car_x1, car_y1, car_x2, _ = car_bbox
        ph, pw = crop.shape[:2]

        # Center overlay horizontally over the car
        x_left = int((car_x1 + car_x2 - pw) / 2)
        x_right = x_left + pw

        # Place crop above the car
        y_bottom = int(car_y1) - int(self.overlay_gap_above_car)
        y_top = y_bottom - ph

        # White banner above crop
        banner_bottom = y_top
        banner_top = banner_bottom - int(self.overlay_banner_height)

        try:
            # crop
            frame[y_top:y_bottom, x_left:x_right, :] = crop
            # banner
            frame[banner_top:banner_bottom, x_left:x_right, :] = (255, 255, 255)

            # centered text on the banner
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_thickness,
            )
            tx = int((car_x1 + car_x2 - tw) / 2)
            ty = int(banner_top + (self.overlay_banner_height / 2) + (th / 2))

            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                (0, 0, 0),
                self.text_thickness,
            )
        except Exception:
            pass

if __name__ == "__main__":
    annotator = VideoAnnotator(
        results_csv="./test_interpolated.csv",
        video_path="sample.mp4",
        out_path="./out.mp4",
    )
    annotator.run()
