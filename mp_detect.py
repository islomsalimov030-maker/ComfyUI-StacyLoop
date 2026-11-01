"""MediaPipe Tasks (Apache-2.0, code and weights) as a drop-in for the ultralytics YOLO face / hand detectors
(face_yolov8m.pt / hand_yolov8s.pt are Ultralytics-trained -> AGPL-3.0).

`load(name)` returns an object with the slice of the ultralytics API the face pass uses:
`predict(img_or_list, conf=..., verbose=False)` -> list of results with `.boxes.xyxy`, `.boxes.conf`, `.boxes.cls`
(torch tensors, `len(boxes)` = detections). Images are BGR uint8, like for ultralytics.

Names: "mediapipe_face" (BlazeFace full range) and "mediapipe_hand" (Hand Landmarker; box = landmarks' bbox).
Models live in models/mediapipe; missing ones are downloaded from storage.googleapis.com/mediapipe-models.
"""
from __future__ import annotations

import os
import threading
import urllib.request

import numpy as np
import torch

FACE, HAND = "mediapipe_face", "mediapipe_hand"
NAMES = (FACE, HAND)
_URLS = {
    FACE: "face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
    HAND: "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
_BASE = "https://storage.googleapis.com/mediapipe-models/"
# BlazeFace boxes are tighter than face_yolov8m (brow..chin vs forehead..chin): scaled about the box centre so
# crop_factor / AUTO thresholds tuned on YOLO keep their meaning (measured on Stacy frames, WS-15 01.11).
FACE_BOX = (0.90, 1.22, -0.084)  # (width scale, height scale, centre shift in box heights, + = down)
_FACE_MIN_CONF = 0.05         # the detector keeps everything above this; predict(conf=...) filters

_tasks: dict = {}
_lock = threading.Lock()


def is_mediapipe(name) -> bool:
    return str(name or "").startswith("mediapipe")


def _model_dir() -> str:
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "mediapipe")
    except Exception:  # outside ComfyUI
        return os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")


def model_path(name: str) -> str:
    fname = _URLS[name].rsplit("/", 1)[1]
    path = os.path.join(_model_dir(), fname)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(_BASE + _URLS[name], tmp)
        os.replace(tmp, path)
    return path


def _task(name: str, conf: float):
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as v
    key = (name, None if name == FACE else round(float(conf), 3))
    with _lock:
        if key not in _tasks:
            with open(model_path(name), "rb") as f:   # buffer: model paths with non-ASCII characters
                base = mpt.BaseOptions(model_asset_buffer=f.read())
            if name == FACE:
                _tasks[key] = v.FaceDetector.create_from_options(
                    v.FaceDetectorOptions(base_options=base, min_detection_confidence=_FACE_MIN_CONF))
            else:
                c = float(conf)
                _tasks[key] = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
                    base_options=base, num_hands=4, min_hand_detection_confidence=c,
                    min_hand_presence_confidence=c, min_tracking_confidence=c))
        return _tasks[key]


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32).reshape(-1)
        self.cls = torch.zeros_like(self.conf)

    def __len__(self):
        return int(self.xyxy.shape[0])


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


def _mp_image(bgr: np.ndarray):
    import mediapipe as mp
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


TILE_FRAC = 0.5   # tile side as a share of the short frame side; tiles overlap by a third


def _tiles(H, W):
    """Whole frame + overlapping square tiles: BlazeFace sees the image at 192 px, so a 90-px face of a FullHD
    wide shot is a few pixels on the whole frame but a normal face on a tile."""
    out = [(0, 0, W, H)]
    t = int(min(H, W) * TILE_FRAC)
    if t < 160:
        return out
    step = max(1, int(t * 2 / 3))
    ys = list(range(0, max(1, H - t), step)) + [H - t]
    xs = list(range(0, max(1, W - t), step)) + [W - t]
    out += [(x, y, x + t, y + t) for y in sorted(set(ys)) for x in sorted(set(xs))]
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _nms(boxes, scores, thr=0.4):
    keep = []
    for i in sorted(range(len(boxes)), key=lambda k: -scores[k]):
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def _detect_face(bgr, conf):
    H, W = bgr.shape[:2]
    sw, sh, dy = FACE_BOX
    task = _task(FACE, conf)
    boxes, scores = [], []
    for x0, y0, x1, y1 in _tiles(H, W):
        for d in task.detect(_mp_image(np.ascontiguousarray(bgr[y0:y1, x0:x1]))).detections:
            s = float(d.categories[0].score) if d.categories else 0.0
            if s < conf:
                continue
            b = d.bounding_box
            cx, cy = x0 + b.origin_x + b.width / 2.0, y0 + b.origin_y + b.height / 2.0 + dy * b.height
            w, h = b.width * sw, b.height * sh
            boxes.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)])
            scores.append(s)
    return _nms(boxes, scores)


def _detect_hand(bgr, conf):
    H, W = bgr.shape[:2]
    task = _task(HAND, conf)
    boxes, scores = [], []
    # whole frame only: on tiles the landmarker takes ears / chins of the cut face for hands (WS-15 01.11)
    for x0, y0, x1, y1 in [(0, 0, W, H)]:
        r = task.detect(_mp_image(np.ascontiguousarray(bgr[y0:y1, x0:x1])))
        tw, th = x1 - x0, y1 - y0
        for k, lm in enumerate(r.hand_landmarks):
            xs = [x0 + p.x * tw for p in lm]
            ys = [y0 + p.y * th for p in lm]
            boxes.append([max(0.0, min(xs)), max(0.0, min(ys)), min(W, max(xs)), min(H, max(ys))])
            hd = r.handedness[k] if k < len(r.handedness) else []
            scores.append(float(hd[0].score) if hd else 1.0)
    return _nms(boxes, scores)


class MPDetector:
    def __init__(self, name: str):
        if name not in NAMES:
            raise ValueError(f"unknown MediaPipe detector '{name}' (use {', '.join(NAMES)})")
        self.name = name

    def predict(self, source, conf=0.25, verbose=False, **_):
        frames = source if isinstance(source, (list, tuple)) else [source]
        fn = _detect_face if self.name == FACE else _detect_hand
        return [_Result(*fn(np.asarray(f), float(conf))) for f in frames]

    __call__ = predict


def load(name: str) -> MPDetector:
    _task(name, 0.3)   # fail early if mediapipe or the model is missing
    return MPDetector(name)
