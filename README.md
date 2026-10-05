# ComfyUI-StacyLoop

Ноды для зацикленных роликов MiniMax H3 (проект Stacy):
StacyFitFrame, StacyLoopSeam, StacyLoopExtend, StacyLoopFold, StacyFlowAlign, StacyOcclusionDenoise,
StacySigmas, StacyResidualSmooth, StacyFrameCount, StacyHandMask.

Установка: положить папку в `ComfyUI/custom_nodes/ComfyUI-StacyLoop`, `pip install -r requirements.txt`, перезапустить ComfyUI.
StacyOcclusionDenoise / StacyHandMask ищут `hand_yolov8s.pt` в `models/ultralytics/bbox` (или `models/ultralytics`).
