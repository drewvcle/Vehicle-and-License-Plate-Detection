import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d


BBox = np.ndarray  # shape (4,) float


@dataclass(frozen=True)
class Paths:
    src_csv: Path = Path("test.csv")
    dst_csv: Path = Path("test_interpolated.csv")


@dataclass(frozen=True)
class Columns:
    frame: str = "frame_nmr"
    car_id: str = "car_id"
    car_bbox: str = "car_bbox"
    plate_bbox: str = "license_plate_bbox"
    plate_bbox_score: str = "license_plate_bbox_score"
    plate_text: str = "license_number"
    plate_text_score: str = "license_number_score"


def _as_int_frame(v: Any) -> int:
    return int(v)


def _as_int_car_id(v: Any) -> int:
    return int(float(v))


def _parse_bbox_cell(cell: Any) -> BBox:
    """
    Accepts values that look like:
      "[12 34 56 78]"  OR  "12 34 56 78"
    Returns np.array([x1,y1,x2,y2], dtype=float).
    """
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    parts = s.split()
    return np.array([float(p) for p in parts], dtype=float)


def _lerp_fill(
    x0: int, b0: BBox,
    x1: int, b1: BBox
) -> Tuple[np.ndarray, np.ndarray]:
    gap = x1 - x0
    x = np.array([x0, x1], dtype=float)
    x_new = np.linspace(x0, x1, num=gap, endpoint=False, dtype=float)

    f = interp1d(x, np.vstack([b0, b1]), axis=0, kind="linear")
    return x_new, f(x_new)  # includes b0 at index 0


class BBoxInterpolator:
    # Expands per-car sparse detections into a per-frame sequence by linearly
    # interpolating missing car and license-plate bounding boxes.

    def __init__(self, cols: Columns = Columns()):
        self.c = cols

    def interpolate_rows(self, rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        frames = np.array([_as_int_frame(r[self.c.frame]) for r in rows], dtype=int)
        car_ids = np.array([_as_int_car_id(r[self.c.car_id]) for r in rows], dtype=int)
        car_boxes = np.array([_parse_bbox_cell(r[self.c.car_bbox]) for r in rows], dtype=float)
        plate_boxes = np.array([_parse_bbox_cell(r[self.c.plate_bbox]) for r in rows], dtype=float)

        out: List[Dict[str, str]] = []
        for cid in np.unique(car_ids):
            out.extend(self._interpolate_one_car(rows, frames, car_ids, car_boxes, plate_boxes, int(cid)))
        return out

    def _interpolate_one_car(
        self,
        rows: Sequence[Dict[str, str]],
        frames: np.ndarray,
        car_ids: np.ndarray,
        car_boxes: np.ndarray,
        plate_boxes: np.ndarray,
        cid: int,
    ) -> List[Dict[str, str]]:
        mask = car_ids == cid

        car_frames = frames[mask]
        car_b = car_boxes[mask]
        plate_b = plate_boxes[mask]

        # Debug print preserved (but simplified)
        original_frame_list = [r[self.c.frame] for r in rows if _as_int_car_id(r[self.c.car_id]) == cid]
        print(original_frame_list, cid)

        first_f = int(car_frames[0])

        car_seq: List[BBox] = []
        plate_seq: List[BBox] = []

        for i in range(len(car_frames)):
            f_i = int(car_frames[i])
            b_car = car_b[i]
            b_plate = plate_b[i]

            if i > 0:
                f_prev = int(car_frames[i - 1])
                prev_car = car_seq[-1]
                prev_plate = plate_seq[-1]

                if f_i - f_prev > 1:
                    # Fill frames between f_prev and f_i using linear interpolation
                    _, car_interp = _lerp_fill(f_prev, prev_car, f_i, b_car)
                    _, plate_interp = _lerp_fill(f_prev, prev_plate, f_i, b_plate)

                    # Drop the first element (prev frame), keep only the "new" in-between frames
                    car_seq.extend(list(car_interp[1:]))
                    plate_seq.extend(list(plate_interp[1:]))

            car_seq.append(b_car)
            plate_seq.append(b_plate)

        produced: List[Dict[str, str]] = []
        original_frame_set = set(int(x) for x in original_frame_list)

        for i in range(len(car_seq)):
            f = first_f + i
            produced.append(self._make_output_row(rows, cid, f, car_seq[i], plate_seq[i], original_frame_set))

        return produced

    def _make_output_row(
        self,
        rows: Sequence[Dict[str, str]],
        cid: int,
        frame_nmr: int,
        car_box: BBox,
        plate_box: BBox,
        original_frames: set[int],
    ) -> Dict[str, str]:
        row: Dict[str, str] = {
            self.c.frame: str(frame_nmr),
            self.c.car_id: str(cid),
            self.c.car_bbox: " ".join(map(str, car_box.tolist())),
            self.c.plate_bbox: " ".join(map(str, plate_box.tolist())),
        }

        if frame_nmr not in original_frames:
            row[self.c.plate_bbox_score] = "0"
            row[self.c.plate_text] = "0"
            row[self.c.plate_text_score] = "0"
            return row

        src = next(
            (
                r for r in rows
                if _as_int_frame(r[self.c.frame]) == frame_nmr and _as_int_car_id(r[self.c.car_id]) == cid
            ),
            None,
        )

        row[self.c.plate_bbox_score] = (src.get(self.c.plate_bbox_score, "0") if src else "0")
        row[self.c.plate_text] = (src.get(self.c.plate_text, "0") if src else "0")
        row[self.c.plate_text_score] = (src.get(self.c.plate_text_score, "0") if src else "0")
        return row


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    paths = Paths()
    cols = Columns()

    raw_rows = read_csv_dicts(paths.src_csv)

    interpolator = BBoxInterpolator(cols)
    filled_rows = interpolator.interpolate_rows(raw_rows)

    output_header = [
        cols.frame,
        cols.car_id,
        cols.car_bbox,
        cols.plate_bbox,
        cols.plate_bbox_score,
        cols.plate_text,
        cols.plate_text_score,
    ]
    write_csv_dicts(paths.dst_csv, output_header, filled_rows)
