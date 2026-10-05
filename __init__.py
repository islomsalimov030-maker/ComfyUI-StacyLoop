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

import torch
import torch.nn.functional as F

CATEGORY = "StacyLoop"


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
}
