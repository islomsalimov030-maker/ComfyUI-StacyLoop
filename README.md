# ComfyUI-StacyLoop

Ноды для зацикленных роликов MiniMax H3 (проект Stacy):
StacyFitFrame, StacyLoopSeam, StacyLoopExtend, StacyLoopFold, StacyFlowAlign, StacyOcclusionDenoise,
StacySigmas, StacyResidualSmooth, StacyFrameCount, StacyHandMask.

Установка: положить папку в `ComfyUI/custom_nodes/ComfyUI-StacyLoop`, `pip install -r requirements.txt`, перезапустить ComfyUI.
StacyOcclusionDenoise / StacyHandMask ищут `hand_yolov8s.pt` в `models/ultralytics/bbox` (или `models/ultralytics`).

## Детекторы без AGPL (01.11)
Веса `face_yolov8m.pt` / `hand_yolov8s.pt` обучены Ultralytics → AGPL-3.0. Вместо них — MediaPipe Tasks
(Apache-2.0, код и веса), модуль `mp_detect.py` с тем же интерфейсом `predict()`:
- `hand_model = "mediapipe_hand"` в StacyOcclusionDenoise / StacyHandMask;
- `detector = "mediapipe_face"` в StacyAutoConfidence, `face_detector` в StacySettings;
- **StacyFaceTrackCrop** — `H3 Face Track + Crop` из ComfyUI-H3-FaceRefine (MIT, пакет нужен) с детектором
  `mediapipe_face`; код трекера оригинальный, входы и выходы те же. `identity_track` держать выключенным
  (веса InsightFace некоммерческие).

Модели (`blaze_face_full_range.tflite`, `hand_landmarker.task`) лежат в `models/mediapipe`, при первом запуске
докачиваются с storage.googleapis.com/mediapipe-models. Умолчания нод — прежние (`*.pt`): старые воркфлоу не меняются.
