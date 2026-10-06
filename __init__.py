"""ComfyUI-StacyLoop — small, dependency-free helpers for looping MiniMax H3 idle videos.

Nodes
  StacyFitFrame          cover-crop + resize an image/batch to an exact WxH (no stretching, no pad)
  StacySigmas            a real low-denoise schedule for H3 img2img (face pass): starts at a chosen SIGMA,
                         spaced in the shifted flow-time the turbo LoRAs were distilled on
  StacyLoopSeam          close a generated loop: the last N frames cross-fade INTO the first N (replace,
                         not insert) -> length n-N, no frozen frame, no double frame at the wrap
  StacyLoopExtend        append the clip's own head to its tail and pad to H3's 17k+5 grid, so a video
                         pass (face refine) sees the wrap as ordinary continuous motion
  StacyLoopFold          undo StacyLoopExtend: the refined continuation (frames n..) fades into the head,
                         output is exactly n frames and wraps seamlessly
  StacyResidualSmooth    temporal anti-flicker on the RESIDUAL a refine pass added (refined - original):
                         notch at H3's 4-frame latent period, source motion is untouched
  StacyFrameCount        frames in a batch (INT) and the nearest H3 grid length
  StacyFlowAlign         lock a refined face to the ORIGINAL video's geometry and motion: (1) warp each refined
                         crop onto the original crop with a heavily smoothed optical flow (the redrawn eyes/nose/
                         mouth/jaw sit exactly where the source has them), (2) motion-compensated temporal
                         consolidation of the residual the refine added, forward + backward along the source's own
                         optical flow (no ghosting: the averaging follows the motion, occlusions are excluded)
  StacyOcclusionDenoise  hands in front of the face: instead of cutting them out of the paste (hard rectangles that
                         left patches of the raw video), LOWER the face pass strength under the hand, softly in
                         space and time, inside the latent noise mask (after H3PerFrameDenoise). The hand is
                         barely touched, the rest of the face gets the full pass, nothing pops
  StacyHandMask          face-pass paste mask that leaves HANDS alone: the face rect (canvas space, from the
                         H3 face-track transform) minus every detected hand (YOLO hand detector), grown and
                         held over neighbouring frames. A hand in front of the face keeps its original pixels
                         instead of being blended with a redrawn face (= transparent / doubled fingers)

All math is plain torch on whatever device the tensors are on.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

CATEGORY = "StacyLoop"


class _AnyType(str):
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


def _grid_up(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _ease(x: torch.Tensor, curve: str) -> torch.Tensor:
    if curve == "linear":
        return x
    if curve == "cosine":
        return 0.5 - 0.5 * torch.cos(math.pi * x)
    # smootherstep
    return x * x * x * (x * (x * 6 - 15) + 10)


# ------------------------------------------------------------------ fit
class StacyFitFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "width": ("INT", {"default": 1344, "min": 16, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 768, "min": 16, "max": 8192, "step": 8}),
            "method": (["lanczos", "bicubic", "bilinear", "area"], {"default": "lanczos"}),
            "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Vertical crop position when the source is taller than the target "
                                              "aspect: 0 = keep the top, 0.5 = centre, 1 = keep the bottom."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, width, height, method, anchor_y):
        b, h, w, c = images.shape
        tr = width / height
        sr = w / h
        if sr > tr:                       # too wide -> crop sides (centre)
            nw = max(1, round(h * tr))
            x0 = (w - nw) // 2
            img = images[:, :, x0:x0 + nw, :]
        else:                             # too tall -> crop top/bottom at anchor_y
            nh = max(1, round(w / tr))
            y0 = int(round((h - nh) * anchor_y))
            img = images[:, y0:y0 + nh, :, :]
        if img.shape[1] == height and img.shape[2] == width:
            return (img.contiguous(), width, height)
        x = img.movedim(-1, 1)
        if method == "lanczos":
            import comfy.utils
            x = comfy.utils.common_upscale(x, width, height, "lanczos", "disabled")
        else:
            x = F.interpolate(x, size=(height, width), mode=method,
                              align_corners=False if method in ("bilinear", "bicubic") else None,
                              antialias=method in ("bilinear", "bicubic"))
        return (x.movedim(1, -1).clamp(0, 1), width, height)


# ------------------------------------------------------------------ sigmas
def flow_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * t / (1 + (shift - 1) * t)


def flow_unshift(s: float, shift: float) -> float:
    return s / (shift - (shift - 1) * s)


class StacySigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "start_sigma": ("FLOAT", {"default": 0.5, "min": 0.02, "max": 1.0, "step": 0.01,
                                      "tooltip": "Noise level the img2img pass starts from (1.0 = full "
                                                 "generation). 0.35-0.6 re-renders texture and detail while "
                                                 "keeping pose, expression and motion of the source."}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            "shift": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                "tooltip": "Same value as the model's MiniMaxH3SigmaShift (turbo 768p LoRAs: 6)."}),
        }}

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, start_sigma, steps, shift):
        t0 = flow_unshift(min(start_sigma, 0.9999), shift)
        t = torch.linspace(t0, 0.0, steps + 1)
        s = flow_shift(t, shift)
        s[-1] = 0.0
        if start_sigma >= 0.9999:
            s[0] = 1.0
        rep = "sigmas: " + ", ".join(f"{v:.3f}" for v in s.tolist())
        return (s.float(), rep)


# ------------------------------------------------------------------ loop seam (generation)
class StacyLoopSeam:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "crossfade": ("INT", {"default": 8, "min": 0, "max": 96,
                                  "tooltip": "Tail frames that cross-fade into the head. 0 = only drop the "
                                             "duplicated last frame."}),
            "curve": (["smootherstep", "cosine", "linear"], {"default": "smootherstep"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, crossfade, curve):
        n = images.shape[0]
        if crossfade <= 0 or n < 3:
            out = images[:-1] if n > 1 else images   # last frame == first frame of the loop
            return (out, out.shape[0])
        N = min(crossfade, n // 3)
        body = images[: n - N].clone()
        tail = images[n - N:]
        # i = 0: fully the tail's first frame (continues from frame n-N-1 ... wraps), i = N-1: almost head
        w = _ease((torch.arange(N, dtype=torch.float32) + 1) / (N + 1), curve).to(images.device)
        w = w.view(N, 1, 1, 1).to(images.dtype)
        body[:N] = tail * (1 - w) + body[:N] * w
        return (body, body.shape[0])


# ------------------------------------------------------------------ loop extend / fold (video refine passes)
class StacyLoopExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "min_extra": ("INT", {"default": 24, "min": 0, "max": 400,
                                  "tooltip": "At least this many head frames are appended after the tail."}),
            "h3_grid": ("BOOLEAN", {"default": True, "tooltip": "Pad the total to 17k+5 frames (needed by H3)."}),
            "is_loop": ("BOOLEAN", {"default": True,
                                    "tooltip": "Off: pad by repeating the last frame instead (non-loop clips)."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "original_frames", "total_frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, min_extra, h3_grid, is_loop):
        n = images.shape[0]
        total = n + (min_extra if is_loop else 0)
        if h3_grid:
            total = _grid_up(total)
        extra = total - n
        if extra <= 0:
            return (images, n, n)
        if is_loop:
            reps = math.ceil(extra / n)
            head = torch.cat([images] * reps, 0)[:extra]
        else:
            head = images[-1:].repeat(extra, 1, 1, 1)
        return (torch.cat([images, head], 0), n, total)


class StacyLoopFold:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "original_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
            "fade": ("INT", {"default": 16, "min": 0, "max": 200,
                             "tooltip": "Head frames over which the refined continuation fades into the "
                                        "refined head. Must be <= appended frames."}),
            "curve": (["smootherstep", "cosine", "linear"], {"default": "smootherstep"}),
            "is_loop": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, original_frames, fade, curve, is_loop):
        n = original_frames or images.shape[0]
        out = images[:n].clone()
        if not is_loop:
            return (out,)
        extra = images.shape[0] - n
        F_ = min(fade, extra, n // 2)
        if F_ <= 0:
            return (out,)
        cont = images[n:n + F_]                      # refined frames that follow frame n-1 = head content
        w = _ease(torch.arange(F_, dtype=torch.float32) / F_, curve).to(images.device)
        w = w.view(F_, 1, 1, 1).to(images.dtype)
        out[:F_] = cont * (1 - w) + out[:F_] * w
        return (out,)


# ------------------------------------------------------------------ residual anti-flicker
_KERNELS = {
    "notch4": [0.5, 1.0, 1.0, 1.0, 0.5],            # zero at H3's 4-frame latent period (6 Hz @ 24 fps)
    "notch4_wide": [0.25, 0.75, 1.0, 1.0, 1.0, 0.75, 0.25],
    "box3": [1.0, 1.0, 1.0],
}


class StacyResidualSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "refined": ("IMAGE",),
            "kernel": (list(_KERNELS), {"default": "notch4"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "wrap": ("BOOLEAN", {"default": False, "tooltip": "Circular padding (clip is a closed loop)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, original, refined, kernel, strength, wrap):
        n = min(original.shape[0], refined.shape[0])
        if n < 3 or strength <= 0:
            return (refined,)
        o = original[:n].to(refined.device, refined.dtype)
        if o.shape[1:3] != refined.shape[1:3]:
            o = F.interpolate(o.movedim(-1, 1), size=refined.shape[1:3], mode="bilinear",
                              align_corners=False).movedim(1, -1)
        r = refined[:n] - o
        k = torch.tensor(_KERNELS[kernel], dtype=r.dtype, device=r.device)
        k = k / k.sum()
        p = len(k) // 2
        out = torch.empty_like(r)
        # frame-chunked to keep memory flat on long 1080p batches
        if wrap:
            idx = lambda i: i % n
        else:
            idx = lambda i: min(max(i, 0), n - 1)
        for t in range(n):
            acc = None
            for j, kv in enumerate(k):
                f = r[idx(t + j - p)] * kv
                acc = f if acc is None else acc + f
            out[t] = acc
        res = r * (1 - strength) + out * strength
        return ((o + res).clamp(0, 1),)


class StacyFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("frames", "h3_grid_frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images):
        n = images.shape[0]
        return (n, _grid_up(n))


def _np_gray(img_t, size):
    import numpy as np
    import cv2
    a = (img_t[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


class StacyFlowAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE", {"tooltip": "Source face crops (H3 Face Track Crop output)."}),
            "refined": ("IMAGE", {"tooltip": "Refined crops (decoded face pass), same canvas."}),
            "align": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "How strongly the refined geometry is pulled onto the source's."}),
            "align_blur": ("FLOAT", {"default": 0.04, "min": 0.005, "max": 0.2, "step": 0.005,
                                     "tooltip": "Smoothing of the alignment flow, fraction of the canvas. Larger = "
                                                "only position/shape drift is corrected, fine identity detail kept."}),
            "temporal": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05,
                                   "tooltip": "Motion-compensated consolidation of the refine residual over time."}),
            "flow_size": ("INT", {"default": 384, "min": 128, "max": 1024, "step": 32}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, original, refined, align, align_blur, temporal, flow_size):
        import numpy as np
        import cv2
        n = min(original.shape[0], refined.shape[0])
        H, W = int(refined.shape[1]), int(refined.shape[2])
        o = original[:n].to(refined.device, refined.dtype)
        if o.shape[1:3] != refined.shape[1:3]:
            o = F.interpolate(o.movedim(-1, 1), size=(H, W), mode="bilinear", align_corners=False).movedim(1, -1)
        O = o[..., :3].float().cpu().numpy()
        R = refined[:n, ..., :3].float().cpu().numpy().copy()      # never write into ComfyUI's cached input
        S = int(flow_size)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        go = [_np_gray(o[i], S) for i in range(n)]
        gr = [_np_gray(refined[i], S) for i in range(n)]
        sx, sy = W / S, H / S
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        def up(fl):
            f = cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR)
            f[..., 0] *= sx
            f[..., 1] *= sy
            return f

        # 1) geometry lock: sample the refined crop where the source has each feature (in place: memory flat)
        shift = []
        if align > 0:
            sig = max(1.0, align_blur * S)
            for i in range(n):
                fl = dis.calc(go[i], gr[i], None)                     # go(p) ~ gr(p + fl)
                fl = cv2.GaussianBlur(fl, (0, 0), sig) * float(align)
                F_ = up(fl)
                shift.append(float(np.sqrt((F_ ** 2).sum(-1)).mean()))
                R[i] = cv2.remap(R[i], xx + F_[..., 0], yy + F_[..., 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        res = R
        res -= O                                                      # residual the refine added
        # 2) motion-compensated temporal consolidation of the residual (forward + backward, averaged)
        if temporal > 0 and n > 2:
            flows = {}
            def flow(i, j):                                           # go[i](p) ~ go[j](p + f)
                if (i, j) not in flows:
                    flows[(i, j)] = dis.calc(go[i], go[j], None)
                return up(flows[(i, j)])
            def warp(img, f):
                return cv2.remap(img, xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            def sweep(order, out):
                out[order[0]] = res[order[0]]
                for k in range(1, len(order)):
                    i, j = order[k], order[k - 1]
                    f = flow(i, j)
                    pred = warp(out[j], f)
                    # occlusion / mismatch: where the warped SOURCE does not match, do not carry the residual
                    err = np.abs(warp(O[j], f) - O[i]).mean(-1)
                    w = np.clip(1.0 - err / 0.06, 0.0, 1.0)[..., None] * float(temporal)
                    out[i] = res[i] + w * (pred - res[i])
                return out
            fwd = sweep(list(range(n)), np.empty_like(res))
            fwd = 0.5 * fwd
            bwd = sweep(list(range(n - 1, -1, -1)), np.empty_like(res))
            fwd += 0.5 * bwd
            del bwd
            res = fwd
        res += O
        out = np.clip(res, 0.0, 1.0, out=res)
        rep = (f"flow align: mean geometric correction {np.mean(shift) if shift else 0:.2f}px "
               f"(canvas {W}x{H}), temporal {temporal}")
        print("[StacyLoop] " + rep)
        t = torch.from_numpy(out).to(refined.device, refined.dtype)
        if refined.shape[-1] == 4:
            t = torch.cat([t, refined[:n, ..., 3:]], -1)
        return (t, rep)


_HAND_CACHE: dict = {}


def _load_yolo(name: str):
    if name in _HAND_CACHE:
        return _HAND_CACHE[name]
    import os
    path = None
    try:
        import folder_paths
        for key in ("ultralytics_bbox", "ultralytics"):
            try:
                path = folder_paths.get_full_path(key, name)
            except Exception:
                path = None
            if path:
                break
        if path is None:
            base = getattr(folder_paths, "models_dir", "models")
            for sub in ("ultralytics/bbox", "ultralytics"):
                cand = os.path.join(base, *sub.split("/"), name)
                if os.path.exists(cand):
                    path = cand
                    break
    except Exception:
        pass
    if path is None and os.path.exists(name):
        path = name
    if path is None:
        _HAND_CACHE[name] = None
        return None
    from ultralytics import YOLO
    _HAND_CACHE[name] = YOLO(path)
    return _HAND_CACHE[name]


class StacyHandMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "crops": ("IMAGE", {"tooltip": "The ORIGINAL face crops (H3 Face Track Crop output)."}),
            "transform": ("H3FACEXFORM",),
            "hand_model": ("STRING", {"default": "hand_yolov8s.pt"}),
            "confidence": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
            "face_dilation": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2,
                                      "tooltip": "Grow the face rect (canvas px), as H3FaceStitch mask_dilation."}),
            "hand_grow": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Grow each hand box by this fraction of its size (covers the "
                                               "feather of the stitch and motion blur)."}),
            "hold_frames": ("INT", {"default": 2, "min": 0, "max": 12,
                                    "tooltip": "A hand found on frame t also protects t-k..t+k."}),
        }}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, crops, transform, hand_model, confidence, face_dilation, hand_grow, hold_frames):
        import numpy as np
        n = int(crops.shape[0])
        ch, cw = int(crops.shape[1]), int(crops.shape[2])
        rects = transform.get("face_rect") or []
        face = torch.zeros((n, ch, cw), dtype=torch.float32)
        for i in range(n):
            fx, fy, fw, fh = rects[i] if i < len(rects) else (cw * .25, ch * .25, cw * .5, ch * .5)
            x0 = max(0, int(round(fx - face_dilation))); y0 = max(0, int(round(fy - face_dilation)))
            x1 = min(cw, int(round(fx + fw + face_dilation))); y1 = min(ch, int(round(fy + fh + face_dilation)))
            if x1 > x0 and y1 > y0:
                face[i, y0:y1, x0:x1] = 1.0
        model = None
        err = ""
        try:
            model = _load_yolo(hand_model)
        except Exception as e:                      # never break the face pass over the hand guard
            err = str(e)[:200]
        if model is None:
            rep = f"hand mask: detector '{hand_model}' unavailable ({err or 'not found'}) - face rect only"
            print("[StacyLoop] " + rep)
            return (face, rep)
        hands = torch.zeros((n, ch, cw), dtype=torch.float32)
        found = 0
        frames_with = []
        bs = 16
        for s in range(0, n, bs):
            batch = [(crops[i, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1].copy()
                     for i in range(s, min(n, s + bs))]
            res = model.predict(batch, conf=float(confidence), verbose=False)
            for k, r in enumerate(res):
                i = s + k
                b = getattr(r, "boxes", None)
                if b is None or len(b) == 0:
                    continue
                frames_with.append(i)
                for x0, y0, x1, y1 in b.xyxy.cpu().numpy().tolist():
                    gw, gh = (x1 - x0) * hand_grow, (y1 - y0) * hand_grow
                    a0 = max(0, int(x0 - gw)); c0 = max(0, int(y0 - gh))
                    a1 = min(cw, int(x1 + gw)); c1 = min(ch, int(y1 + gh))
                    if a1 > a0 and c1 > c0:
                        hands[i, c0:c1, a0:a1] = 1.0
                        found += 1
        if hold_frames > 0 and found:
            k = 2 * int(hold_frames) + 1
            hands = F.max_pool1d(hands.permute(1, 2, 0).reshape(-1, 1, n), k, stride=1,
                                 padding=hold_frames).reshape(ch, cw, n).permute(2, 0, 1).contiguous()
        out = (face * (1.0 - hands)).clamp(0, 1)
        protected = int((hands.amax(dim=(1, 2)) > 0).sum())
        rep = (f"hand mask: {found} hand box(es) on {len(set(frames_with))}/{n} frames, "
               f"{protected} frame(s) protected (hold +-{hold_frames})")
        print("[StacyLoop] " + rep)
        return (out, rep)


def _hand_boxes(crops, hand_model, confidence):
    model = _load_yolo(hand_model)
    if model is None:
        return None
    n = int(crops.shape[0])
    out = [[] for _ in range(n)]
    for s in range(0, n, 16):
        batch = [(crops[i, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1].copy()
                 for i in range(s, min(n, s + 16))]
        for k, r in enumerate(model.predict(batch, conf=float(confidence), verbose=False)):
            b = getattr(r, "boxes", None)
            if b is not None and len(b):
                out[s + k] = b.xyxy.cpu().numpy().tolist()
    return out


class StacyOcclusionDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "av_latent": ("LATENT", {"tooltip": "Output of H3PerFrameDenoise (its noise_mask is refined here)."}),
            "crops": ("IMAGE", {"tooltip": "The ORIGINAL face crops (H3 Face Track Crop output)."}),
            "hand_model": ("STRING", {"default": "hand_yolov8s.pt"}),
            "confidence": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
            "hand_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Face-pass strength left under a hand (x the per-frame value)."}),
            "spread": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "Soft falloff around the hand box, fraction of the box size."}),
            "time_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 12.0, "step": 0.5,
                                     "tooltip": "Temporal softness in frames: no pop when a hand appears."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, av_latent, crops, hand_model, confidence, hand_strength, spread, time_sigma):
        import numpy as np
        import cv2
        try:
            import comfy.nested_tensor as _nt
        except Exception:
            _nt = None
        prev = av_latent.get("noise_mask")
        if prev is None or _nt is None or not (isinstance(prev, _nt.NestedTensor) or getattr(prev, "is_nested", False)):
            rep = "occlusion denoise: no per-frame noise mask upstream (put it after H3PerFrameDenoise) - unchanged"
            print("[StacyLoop] " + rep)
            return (av_latent, rep)
        try:
            boxes = _hand_boxes(crops, hand_model, confidence)
        except Exception as e:
            boxes = None
            print("[StacyLoop] occlusion denoise: hand detector failed:", str(e)[:200])
        if boxes is None:
            rep = f"occlusion denoise: hand detector '{hand_model}' unavailable - unchanged"
            print("[StacyLoop] " + rep)
            return (av_latent, rep)
        n = int(crops.shape[0])
        ch, cw = int(crops.shape[1]), int(crops.shape[2])
        S = 96                                               # work grid, then resampled to the latent grid
        occ = np.zeros((n, S, S), np.float32)
        found = 0
        for i, bl in enumerate(boxes):
            for x0, y0, x1, y1 in bl:
                found += 1
                m = np.zeros((S, S), np.float32)
                a0, a1 = int(x0 / cw * S), int(np.ceil(x1 / cw * S))
                b0, b1 = int(y0 / ch * S), int(np.ceil(y1 / ch * S))
                m[max(0, b0):min(S, b1), max(0, a0):min(S, a1)] = 1.0
                sig = max(0.5, spread * 0.5 * ((a1 - a0) + (b1 - b0)) / 2)
                m = cv2.GaussianBlur(m, (0, 0), sig)
                m = np.clip(m / max(m.max(), 1e-6) * 1.0, 0, 1)
                occ[i] = np.maximum(occ[i], m)
        if time_sigma > 0 and n > 1:
            k = np.exp(-0.5 * (np.arange(-int(3 * time_sigma), int(3 * time_sigma) + 1) / time_sigma) ** 2)
            pad = len(k) // 2
            P = np.pad(occ, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            occ = np.stack([np.max(P[t:t + len(k)] * k[:, None, None], axis=0) for t in range(n)])
        strength = 1.0 - (1.0 - float(hand_strength)) * np.clip(occ, 0, 1)      # [n,S,S]
        pm = list(prev.unbind())
        v = pm[0]                                                                # [B,C,T,H,W]
        T, H, W = int(v.shape[-3]), int(v.shape[-2]), int(v.shape[-1])
        st = torch.from_numpy(strength).float().unsqueeze(0).unsqueeze(0)       # [1,1,n,S,S]
        st = F.interpolate(st, size=(T, H, W), mode="trilinear", align_corners=False)
        st = st.to(v.device, v.dtype)
        pm[0] = (v * st).clamp(0, 1)
        out = dict(av_latent)
        out["noise_mask"] = _nt.NestedTensor(tuple(pm))
        hit = int((occ.max(axis=(1, 2)) > 0.05).sum())
        rep = (f"occlusion denoise: {found} hand box(es), {hit}/{n} frames softened, strength under a hand "
               f"x{hand_strength}, min mask {float(pm[0].min()):.2f}")
        print("[StacyLoop] " + rep)
        return (out, rep)


# ---------------------------------------------------------------- controls panel / report / VRAM
def _sl(kind, default, lo, hi, step, tip):
    d = {"default": default, "min": lo, "max": hi, "step": step, "display": "slider", "tooltip": tip}
    if kind == "FLOAT":
        d["round"] = step
    return (kind, d)


class StacyControls:
    """All the knobs of the Stacy H3 loop workflow in one panel (sliders and toggles)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": ("BOOLEAN", {"default": True, "label_on": "GENERATE a new clip",
                                 "label_off": "FACE PASS of a ready video",
                                 "tooltip": "On: generate from the keyframe + prompt. Off: only run the face pass on "
                                            "the video named in 'video' (nothing is generated)."}),
            "video": ("STRING", {"default": "stacy_h3/Stacy_00001.mp4", "multiline": False,
                                 "tooltip": "FACE PASS mode only, used when nothing is uploaded in the 'VIDEO for face pass' node: a file "
                                            "name in ComfyUI's output or input folder (e.g. stacy_h3/Stacy_00003.mp4) or a full path."}),
            "duration_sec": _sl("FLOAT", 8.0, 5.0, 10.0, 0.25,
                                "Clip length. Snapped to H3's 17k+5 frame grid at 24 fps (7 s = 175, 8 s = 192, "
                                "9 s = 209, 10 s = 243 frames). The report prints the exact length used."),
            "seed": ("INT", {"default": 777, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed",
                             "tooltip": "Generation seed. The face pass uses seed + 17."}),
            "loop": ("BOOLEAN", {"default": True, "label_on": "loop (first = last frame)",
                                 "label_off": "entry clip (no loop)",
                                 "tooltip": "Off for the pose-entry clips (Standing-4 / Standing-9): no last-frame "
                                            "anchor, no loop seam; the last frame is saved for the next loop."}),
            "end_frame": ("BOOLEAN", {"default": False, "label_on": "END FRAME image = last frame",
                                      "label_off": "no end frame",
                                      "tooltip": "Only when loop is OFF: anchor the last frame to the END FRAME image "
                                                 "(first and last frame differ). With loop ON the keyframe is always "
                                                 "the last frame and this switch is ignored."}),
            "steps": _sl("INT", 20, 8, 40, 1, "Sampling steps (official H3 scheme: 20, res_multistep)."),
            "shift": _sl("FLOAT", 12.0, 6.0, 16.0, 0.5,
                         "Sigma shift. 12 = official. Lower = more motion freedom / less adherence to the "
                         "keyframe, higher = stiffer."),
            "color_lock": _sl("FLOAT", 1.0, 0.0, 1.0, 0.05,
                              "Colour match of every frame to the keyframe (0 = off)."),
            "seam_crossfade": _sl("INT", 8, 0, 24, 1, "Frames the tail cross-fades into the head (loop only)."),
            "free_vram": ("BOOLEAN", {"default": True, "label_on": "free VRAM after sampling",
                                      "label_off": "keep models loaded",
                                      "tooltip": "On for 24-32 GB GPUs (colour lock / RTX / face pass need the "
                                                 "memory). Off on a 96 GB GPU = faster batches."}),
            "face_pass": ("BOOLEAN", {"default": True, "label_on": "face pass ON", "label_off": "face pass OFF",
                                      "tooltip": "Off = generation + FullHD only (the face pass can be run later "
                                                 "on the FullHD video)."}),
            "face_denoise": _sl("FLOAT", 0.30, 0.10, 0.50, 0.01,
                                "How much the face pass redraws the face. 0.25 softer ... 0.40 stronger identity."),
            "face_lora": _sl("FLOAT", 1.0, 0.0, 1.5, 0.05, "Stacy face LoRA strength in the face pass."),
            "large_face_mult": _sl("FLOAT", 0.35, 0.10, 1.0, 0.05,
                                   "Denoise multiplier for LARGE faces (close-ups need less redraw); small faces "
                                   "always get the full face_denoise."),
            "face_lock": _sl("FLOAT", 1.0, 0.0, 1.0, 0.05,
                             "Pull the redrawn face onto the source geometry (anti-jitter). 0 = off."),
            "face_lock_temporal": _sl("FLOAT", 0.8, 0.0, 0.95, 0.05,
                                      "Motion-compensated smoothing of the face-pass detail over time. Higher = "
                                      "steadier, lower = livelier micro-expressions."),
            "hand_strength": _sl("FLOAT", 0.2, 0.0, 1.0, 0.05,
                                 "Face-pass strength under a hand near the face (0 = leave the hand untouched, "
                                 "1 = same as the rest of the face)."),
            "stitch_feather": _sl("INT", 24, 4, 64, 2, "Softness of the pasted face edge, px."),
            "face_confidence": _sl("FLOAT", 0.35, 0.15, 0.60, 0.05,
                                   "Face detector confidence. Lower it if a small / turned face is missed."),
            "crop_factor": _sl("FLOAT", 2.5, 1.8, 3.5, 0.1,
                               "Face crop size as a multiple of the face height (2.5 = face fills ~40%)."),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "INT", "INT", "BOOLEAN", "INT", "FLOAT", "FLOAT", "INT", "BOOLEAN", "BOOLEAN",
                    "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN",
                    "STRING", "BOOLEAN")
    RETURN_NAMES = ("length", "seconds", "seed", "face_seed", "loop", "steps", "shift", "color_lock",
                    "seam_crossfade", "free_vram", "face_pass", "face_denoise", "face_lora", "large_face_mult",
                    "face_lock", "face_lock_temporal", "hand_strength", "stitch_feather", "face_confidence",
                    "crop_factor", "generate", "video_mode", "video", "last_anchor")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, duration_sec, seed, loop, steps, shift, color_lock, seam_crossfade, free_vram, face_pass,
            face_denoise, face_lora, large_face_mult, face_lock, face_lock_temporal, hand_strength,
            stitch_feather, face_confidence, crop_factor, mode=True, video="", end_frame=False):
        k = max(1, round((float(duration_sec) * 24 - 5) / 17))
        n = min(max(17 * k + 5, 124), 362)
        return (n, round((n - 1) / 24.0, 2), int(seed), int(seed) + 17, bool(loop), int(steps), float(shift),
                float(color_lock), int(seam_crossfade), bool(free_vram), bool(face_pass), float(face_denoise),
                float(face_lora), float(large_face_mult), float(face_lock), float(face_lock_temporal),
                float(hand_strength), int(stitch_feather), float(face_confidence), float(crop_factor),
                bool(mode), not bool(mode), str(video), bool(loop) or bool(end_frame))


class StacyFreeVRAM:
    """Pass-through that unloads the models and empties the CUDA cache before the post-processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, enabled):
        if enabled:
            import gc
            import comfy.model_management as mm
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            print("[StacyLoop] free VRAM: models unloaded before post-processing")
        return (images,)


class StacyReport:
    """One readable report for the whole run. Face-pass parts are lazy: never computed when the pass is off."""

    @classmethod
    def INPUT_TYPES(cls):
        lz = {"lazy": True, "forceInput": True}
        return {"required": {"face_pass": ("BOOLEAN", {"forceInput": True}),
                             "loop": ("BOOLEAN", {"forceInput": True}),
                             "seconds": ("FLOAT", {"forceInput": True}),
                             "length": ("INT", {"forceInput": True})},
                "optional": {"final_frames": ("INT", {"forceInput": True}),
                             "face_track": ("STRING", lz), "face_inject": ("STRING", lz),
                             "face_denoise": ("STRING", lz), "hands": ("STRING", lz), "face_lock": ("STRING", lz)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    FACE = ("face_track", "face_inject", "face_denoise", "hands", "face_lock")

    def check_lazy_status(self, face_pass, loop, seconds, length, **kw):
        return [k for k in self.FACE if face_pass and kw.get(k) is None]

    def run(self, face_pass, loop, seconds, length, final_frames=None, **kw):
        out = ["=== GENERATION ===",
               f"mode: {'loop (keyframe = first and last frame)' if loop else 'entry clip (no loop)'}",
               f"length: {length} frames = {seconds:.2f} s @ 24 fps"
               + (f"; after the loop seam: {final_frames} frames" if final_frames else "")]
        if not face_pass:
            out += ["", "=== FACE PASS ===", "off (FullHD output without face refine)"]
        else:
            heads = {"face_track": "FACE TRACK", "face_inject": "FACE CROPS -> LATENT",
                     "face_denoise": "PER-FRAME DENOISE", "hands": "HANDS NEAR THE FACE", "face_lock": "FACE LOCK"}
            for k in self.FACE:
                if kw.get(k):
                    out += ["", f"=== {heads[k]} ===", str(kw[k]).strip()]
        rep = "\n".join(out)
        print("[StacyLoop] report\n" + rep)
        return (rep,)


class StacyPromptRetime:
    """Keep the prompt's clip length in step with the duration slider. The H3 prompts state the clip length
    ('static shot of 8.0 seconds', 'From 4.80 to 7.96 seconds', 'At 7.96 seconds the shot ends'): the final
    time is replaced by the new one, the action phases keep their times, the closing rest phase stretches or
    shrinks. Warns when an action phase would end after the new clip end."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"forceInput": True}),
                             "seconds": ("FLOAT", {"forceInput": True})},
                "optional": {"loop": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, prompt, seconds, loop=True):
        import re
        notes = []
        if not loop:            # entry clip: drop every "ends on the keyframe" instruction of a loop prompt
            before = prompt
            prompt = prompt.replace(" and also its last frame", "").replace(" and ends on <Picture 1>", "")
            prompt = prompt.replace(", then returns to exactly the starting pose so the clip loops seamlessly", "")
            prompt = re.sub(r" At \d+\.\d+ seconds the shot ends on <Picture 1>:[^.]*\.", "", prompt)
            if prompt != before:
                notes.append("loop OFF: the 'ends on the keyframe' lines were removed from the prompt")
        new_end = f"{float(seconds):.2f}"
        m = re.search(r"At (\d+\.\d+) seconds the shot ends", prompt)
        if not m:
            ends_ = re.findall(r"to (\d+\.\d+) seconds", prompt)
            if not ends_:
                return (prompt, "\n".join(notes + ["prompt timing: no time marks found - prompt used as is"]))
            old_end = max(ends_, key=float)
        else:
            old_end = m.group(1)
        out = prompt.replace(f"{old_end} seconds", f"{new_end} seconds")
        out = re.sub(r"(static shot of )(\d+(?:\.\d+)?)( seconds)", lambda k: f"{k.group(1)}{float(seconds):.1f}{k.group(3)}", out)
        ends = [float(b) for a, b in re.findall(r"From (\d+\.\d+) to (\d+\.\d+) seconds", out)]
        starts = [float(a) for a, b in re.findall(r"From (\d+\.\d+) to (\d+\.\d+) seconds", out)]
        warn = ""
        if starts and max(starts) >= float(seconds) - 0.3:
            warn = (f"\nWARNING: the last phase starts at {max(starts):.2f} s - the clip is too short for this "
                    f"prompt's actions; make it longer.")
        rep = f"prompt timing: clip end {old_end} s -> {new_end} s" + ("" if old_end != new_end else " (unchanged)") + warn
        return (out, "\n".join(notes + [rep]))


class StacyGate:
    """Pass the value through when enabled; otherwise block everything downstream silently (and, being lazy,
    never compute the branch that feeds it). Used for the GENERATE / FACE PASS-of-a-video mode switch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enabled": ("BOOLEAN", {"forceInput": True}),
                             "value": (_ANY, {"lazy": True})}}

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, enabled, value=None):
        return ["value"] if enabled and value is None else []

    def run(self, enabled, value=None):
        if not enabled:
            from comfy_execution.graph_utils import ExecutionBlocker
            return (ExecutionBlocker(None),)
        return (value,)


class StacyLoadVideo:
    """Load a ready video (for the face pass) by file name: ComfyUI output folder, input folder or full path.
    Nothing is checked before the run, so the workflow validates even when this branch is switched off."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "frames", "fps", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, video):
        p = cls._find(video)
        return f"{p}:{os.path.getmtime(p)}" if p else video

    @staticmethod
    def _find(video):
        import folder_paths
        v = str(video).strip().strip('"')
        for base in ("", folder_paths.get_output_directory(), folder_paths.get_input_directory()):
            p = os.path.join(base, v) if base else v
            if os.path.isfile(p):
                return p
        return None

    def run(self, video):
        import cv2
        import numpy as np
        p = self._find(video)
        if not p:
            raise FileNotFoundError(f"StacyLoadVideo: '{video}' not found in output/, input/ or as a full path")
        cap = cv2.VideoCapture(p)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            raise ValueError(f"StacyLoadVideo: no frames decoded from {p}")
        out = torch.stack(frames).float().div_(255.0)
        rep = f"video: {os.path.basename(p)}  {out.shape[0]} frames  {out.shape[2]}x{out.shape[1]}  {fps:.2f} fps"
        print("[StacyLoop] " + rep)
        return (out, int(out.shape[0]), fps, rep)


class StacyColorLock:
    """Colour lock of every frame to the keyframe: the same Reinhard transfer in Lab as KJNodes ColorMatchV2
    'reinhard_lab_gpu' (per-frame mean/std matched to the reference), but done a few frames at a time on the GPU
    and returned to the CPU - identical result, a few hundred MB of VRAM instead of 3-6 GB (no OOM on 24-32 GB
    cards while the video model is still staged)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_target": ("IMAGE",), "image_ref": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "chunk": ("INT", {"default": 16, "min": 1, "max": 256})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image_target, image_ref, strength, chunk):
        if strength <= 0:
            return (image_target,)
        import kornia
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        ref = image_ref[:1].to(dev).permute(0, 3, 1, 2).contiguous()
        ref_lab = kornia.color.rgb_to_lab(ref[:, :3]).flatten(2)
        ref_std, ref_mean = torch.std_mean(ref_lab, dim=-1, keepdim=True, unbiased=False)
        B, H, W, C = image_target.shape
        out = torch.empty((B, H, W, 3), dtype=torch.float32)
        for i in range(0, B, chunk):
            src = image_target[i:i + chunk, ..., :3].to(dev).permute(0, 3, 1, 2).float().contiguous()
            lab = kornia.color.rgb_to_lab(src)
            b = lab.shape[0]
            flat = lab.view(b, 3, -1)
            s_std, s_mean = torch.std_mean(flat, dim=-1, keepdim=True, unbiased=False)
            flat = (flat - s_mean) * (ref_std / s_std.clamp_min(1e-6)) + ref_mean
            rgb = kornia.color.lab_to_rgb(flat.view(b, 3, H, W))
            res = (1.0 - strength) * src + strength * rgb
            out[i:i + b] = res.permute(0, 2, 3, 1).clamp_(0, 1).cpu()
            del src, lab, flat, rgb, res
        return (out,)


_VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")


def _read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return torch.stack(frames).float().div_(255.0), fps


class StacyVideoInput:
    """Video for the FACE PASS mode: upload / pick a file here (button 'choose video to upload'), or plug any
    Load Video node into 'frames', or leave '(none)' to use the path typed on the panel. Nothing is checked or
    decoded before the run, and nothing is decoded at all in GENERATE mode."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        d = folder_paths.get_input_directory()
        files = []
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(_VIDEO_EXT):
                    files.append(os.path.relpath(os.path.join(root, n), d).replace(os.sep, "/"))
        return {"required": {"video": (["(none)"] + sorted(files), {"video_upload": True,
                             "tooltip": "Upload or pick the FullHD video to refine. '(none)' = use the panel's path."})},
                "optional": {"frames": ("IMAGE", {"tooltip": "Optional: frames from any Load Video node."}),
                             "fallback_path": ("STRING", {"forceInput": True}),
                             "settings": ("STACY_SETTINGS", {"tooltip": "The panel: its 'video' path is the fallback."})}}

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "frames", "fps", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return True

    @classmethod
    def IS_CHANGED(cls, video, frames=None, fallback_path=None, settings=None):
        if fallback_path is None and isinstance(settings, dict):
            fallback_path = settings.get("video")
        p = cls._pick(video, fallback_path)
        return f"{p}:{os.path.getmtime(p)}" if p else f"{video}|{fallback_path}"

    @staticmethod
    def _pick(video, fallback_path):
        import folder_paths
        if video and video != "(none)":
            p = os.path.join(folder_paths.get_input_directory(), video)
            if os.path.isfile(p):
                return p
        if fallback_path:
            return StacyLoadVideo._find(fallback_path)
        return None

    def run(self, video, frames=None, fallback_path=None, settings=None):
        if fallback_path is None and isinstance(settings, dict):
            fallback_path = settings.get("video")
        if frames is not None:
            rep = f"video: {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]} from the connected Load Video node"
            print("[StacyLoop] " + rep)
            return (frames[..., :3], int(frames.shape[0]), 24.0, rep)
        p = self._pick(video, fallback_path)
        if not p:
            raise FileNotFoundError("StacyVideoInput: no video - upload one in this node or type a path on the panel "
                                    f"(got '{video}' / '{fallback_path}')")
        out, fps = _read_video(p)
        rep = f"video: {os.path.basename(p)}  {out.shape[0]} frames  {out.shape[2]}x{out.shape[1]}  {fps:.2f} fps"
        print("[StacyLoop] " + rep)
        return (out, int(out.shape[0]), fps, rep)


_PANEL_ORDER = ("mode", "video", "seed", "duration_sec", "loop", "end_frame", "steps", "shift", "color_lock",
                "seam_crossfade", "free_vram", "face_pass", "face_denoise", "face_lora", "large_face_mult", "face_lock",
                "face_lock_temporal", "hand_strength", "stitch_feather", "face_confidence", "crop_factor")


class StacyPanel:
    """THE control panel: every knob of the workflow, one 'settings' wire out. Widgets that do not apply to the
    current mode are greyed out (web/stacy_panel.js): FACE PASS mode disables the generation knobs and the
    face_pass switch; GENERATE with face_pass OFF disables the face knobs; end_frame only with loop OFF."""

    @classmethod
    def INPUT_TYPES(cls):
        req = StacyControls.INPUT_TYPES()["required"]
        return {"required": {k: req[k] for k in _PANEL_ORDER}}

    RETURN_TYPES = ("STACY_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        vals = StacyControls().run(**kw)
        return (dict(zip(StacyControls.RETURN_NAMES, vals)),)


class StacySettings:
    """Unpack the panel's settings wire into the individual values (lives inside the engine subgraphs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"settings": ("STACY_SETTINGS",)}}

    RETURN_TYPES = StacyControls.RETURN_TYPES
    RETURN_NAMES = StacyControls.RETURN_NAMES
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, settings):
        return tuple(settings[n] for n in self.RETURN_NAMES)


NODE_CLASS_MAPPINGS = {
    "StacyFitFrame": StacyFitFrame,
    "StacySigmas": StacySigmas,
    "StacyLoopSeam": StacyLoopSeam,
    "StacyLoopExtend": StacyLoopExtend,
    "StacyLoopFold": StacyLoopFold,
    "StacyResidualSmooth": StacyResidualSmooth,
    "StacyFrameCount": StacyFrameCount,
    "StacyHandMask": StacyHandMask,
    "StacyFlowAlign": StacyFlowAlign,
    "StacyOcclusionDenoise": StacyOcclusionDenoise,
    "StacyControls": StacyControls,
    "StacyFreeVRAM": StacyFreeVRAM,
    "StacyReport": StacyReport,
    "StacyPromptRetime": StacyPromptRetime,
    "StacyGate": StacyGate,
    "StacyLoadVideo": StacyLoadVideo,
    "StacyColorLock": StacyColorLock,
    "StacyVideoInput": StacyVideoInput,
    "StacyPanel": StacyPanel,
    "StacySettings": StacySettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StacyFitFrame": "Stacy · Fit frame (cover crop)",
    "StacySigmas": "Stacy · Low-denoise sigmas (H3)",
    "StacyLoopSeam": "Stacy · Loop seam (tail→head)",
    "StacyLoopExtend": "Stacy · Loop extend (+head, H3 grid)",
    "StacyLoopFold": "Stacy · Loop fold (refined wrap)",
    "StacyResidualSmooth": "Stacy · Residual anti-flicker",
    "StacyFrameCount": "Stacy · Frame count",
    "StacyHandMask": "Stacy · Face paste mask without hands",
    "StacyFlowAlign": "Stacy · Lock refined face to source geometry + motion",
    "StacyOcclusionDenoise": "Stacy · Gentler face pass under hands (soft mask)",
    "StacyControls": "Stacy · Controls (all knobs)",
    "StacyFreeVRAM": "Stacy · Free VRAM (pass-through)",
    "StacyReport": "Stacy · Run report",
    "StacyPromptRetime": "Stacy · Prompt length = duration",
    "StacyGate": "Stacy · Gate (mode switch)",
    "StacyLoadVideo": "Stacy · Load ready video (by name)",
    "StacyColorLock": "Stacy · Colour lock to keyframe (low VRAM)",
    "StacyVideoInput": "Stacy · Video for face pass (upload)",
    "StacyPanel": "Stacy · PANEL (all knobs)",
    "StacySettings": "Stacy · Settings (unpack)",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
