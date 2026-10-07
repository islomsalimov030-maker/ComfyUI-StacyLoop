import { app } from "../../scripts/app.js";

// Stacy PANEL: grey out the knobs that do not apply to the current mode / are set to AUTO.
// 'loop' stays active in both modes: the face pass of a ready video also needs to know if it is a loop.
const GEN = ["duration_sec", "end_frame", "free_vram", "face_pass", "char_lora"];
const FACE = ["face_denoise_auto", "face_denoise", "face_lora", "face_lock_temporal"];
const AUTO = ["face_denoise"];
const WATCH = ["mode", "face_pass", "loop", ...AUTO.map((n) => n + "_auto")];

function update(node) {
    const W = (n) => node.widgets?.find((w) => w.name === n);
    const val = (n) => W(n)?.value;
    const generate = val("mode") !== false, loop = val("loop") !== false;
    const facePass = val("face_pass") !== "off" && val("face_pass") !== false;
    const dis = {};
    for (const n of GEN) dis[n] = !generate;
    for (const n of FACE) dis[n] = generate && !facePass;
    if (!generate || loop) dis["end_frame"] = true;
    for (const n of AUTO) if (val(n + "_auto") === true) dis[n] = true;
    for (const [n, d] of Object.entries(dis)) { const w = W(n); if (w) w.disabled = d; }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Stacy.Panel",
    nodeCreated(node) {
        if (node.comfyClass !== "StacyPanel") return;
        for (const n of WATCH) {
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
