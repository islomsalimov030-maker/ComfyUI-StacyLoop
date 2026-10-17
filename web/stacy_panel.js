import { app } from "../../scripts/app.js";

// Stacy PANEL: grey out the knobs that do not apply to the current mode / are set to AUTO.
// 'loop' stays active in both modes: the face pass of a ready video also needs to know if it is a loop.
const GEN = ["duration_sec", "end_frame", "model_10eros", "steps", "shift", "char_lora", "free_vram", "face_pass"];
const FACE = ["face_denoise_auto", "face_denoise", "face_lora", "face_lock_temporal"];
const AUTO = ["face_denoise"];
const WATCH = ["mode", "face_pass", "loop", ...AUTO.map((n) => n + "_auto")];
// Panels saved before feff837 (no model_10eros / steps / shift): their widgets_values get spread over the new
// widgets by position -> steps = NaN, shift = 1.0, char_lora = face_denoise ... Re-map such values by name.
const OLD_ORDER = ["mode", "seed", "control_after_generate", "duration_sec", "loop", "end_frame", "free_vram",
    "face_pass", "face_denoise_auto", "face_denoise", "face_lora", "face_lock_temporal", "char_lora"];
const NEW_DEFAULTS = { model_10eros: false, steps: 20, shift: 12.0 };

function migrate(node, info) {
    const v = info?.widgets_values;
    // old layout: 12 values (before char_lora) or 13; index 6 is the free_vram combo, in the new one a boolean
    if (!Array.isArray(v) || (v.length !== 12 && v.length !== 13) || typeof v[6] !== "string") return;
    const vals = { ...NEW_DEFAULTS };
    OLD_ORDER.forEach((n, i) => { if (i < v.length) vals[n] = v[i]; });
    // the seed's control widget has no stable name in the frontend; it is always the 3rd widget
    (node.widgets ?? []).forEach((w, i) => {
        const n = i === 2 ? "control_after_generate" : w.name;
        if (n in vals) w.value = vals[n];
    });
    console.info("[Stacy.Panel] migrated an old PANEL (" + v.length + " values) by name; steps/shift = 20/12");
}

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
        node.onConfigure = function (...args) {
            const r = oc?.apply(this, args);
            migrate(node, args[0]);
            setTimeout(() => update(node), 0);
            return r;
        };
        setTimeout(() => update(node), 0);
    },
});
