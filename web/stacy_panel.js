import { app } from "../../scripts/app.js";

// Stacy PANEL: grey out the knobs that do not apply to the current mode.
// 'loop' stays active in both modes: the face pass of a ready video also needs to know if it is a loop.
const GEN = ["duration_sec", "end_frame", "steps", "shift", "color_lock", "seam_crossfade", "free_vram", "face_pass"];
const FACE = ["face_denoise", "face_lora", "large_face_mult", "face_lock", "face_lock_temporal", "hand_strength",
              "stitch_feather", "face_confidence", "crop_factor"];

function update(node) {
    const W = (n) => node.widgets?.find((w) => w.name === n);
    const on = (n) => W(n)?.value !== false;
    const generate = on("mode"), facePass = on("face_pass"), loop = on("loop");
    const set = (n, disabled) => { const w = W(n); if (w) w.disabled = disabled; };
    for (const n of GEN) set(n, !generate);
    set("video", generate);
    set("end_frame", !generate || loop);
    set("seam_crossfade", !generate || !loop);
    for (const n of FACE) set(n, generate && !facePass);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Stacy.Panel",
    nodeCreated(node) {
        if (node.comfyClass !== "StacyPanel") return;
        for (const n of ["mode", "face_pass", "loop"]) {
            const w = node.widgets?.find((x) => x.name === n);
            if (!w) continue;
            const cb = w.callback;
            w.callback = function (...args) { const r = cb?.apply(this, args); update(node); return r; };
        }
        const oc = node.onConfigure;
        node.onConfigure = function (...args) { const r = oc?.apply(this, args); setTimeout(() => update(node), 0); return r; };
        setTimeout(() => update(node), 0);
    },
});
