"""Known-answer test of the Image Quantifier's analyze() (Hugging Face Space
smiler488/image-quantifier): leaves of known area next to a 25 mm reference
square, upright and rotated 45 degrees, and an image without a reference.

Prepare: clone the Space to /private/tmp/claude-501/hfspace, copy it to
hfpatched and apply scripts/patches/image-quantifier-reference-scale.patch,
then extract the analysis code without the Gradio interface into
hfspace_core.py and hfpatched_core.py (everything before "with gr.Blocks(",
with a stub for gr). Run with Python + OpenCV + pandas.
"""
import importlib.util, sys, numpy as np, cv2

def load(name):
    spec = importlib.util.spec_from_file_location(name, f"/private/tmp/claude-501/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

PX_PER_MM = 8.0          # image scale used to draw the scene
REF_MM = 25.0            # reference square side
def scene(rot_deg, with_ref=True):
    h, w = 1600, 2000
    img = np.full((h, w, 3), 245, np.uint8)
    if with_ref:
        side = REF_MM * PX_PER_MM
        box = cv2.boxPoints(((260, 260), (side, side), rot_deg)).astype(np.int32)
        cv2.fillPoly(img, [box], (40, 40, 40))
    truth = []
    for i, (a, b) in enumerate([(20, 8), (25, 10), (15, 6)]):  # semi-axes in mm
        c = (650 + i * 500, 1000)
        ax = (int(round(a * PX_PER_MM)), int(round(b * PX_PER_MM)))
        cv2.ellipse(img, c, ax, 20 * i, 0, 360, (60, 150, 50), -1)  # green leaf (BGR)
        m = np.zeros((h, w), np.uint8); cv2.ellipse(m, c, ax, 20 * i, 0, 360, 255, -1)
        truth.append(np.count_nonzero(m) / PX_PER_MM ** 2)
    return img, truth

for name in ["hfspace_core", "hfpatched_core"]:
    app = load(name)
    print(f"== {'current Space' if name == 'hfspace_core' else 'patched'}")
    for label, rot, ref in [("reference 0°", 0, True), ("reference 45°", 45, True), ("no reference", 0, False)]:
        img, truth = scene(rot, ref)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay, df, csv, js, state = app.analyze(rgb, "leaves", 3, "auto", REF_MM, 500, 200000, 40, 35, 85)
        cols = [c for c in df.columns if "area" in c.lower()]
        area_col = next((c for c in cols if "mm" in c.lower()), None)
        if area_col is None or df.empty:
            print(f"  {label:14s} px/mm={state.get('px_per_mm')}  columns: {list(df.columns)[:8]}  js: {str(js)[:300]}")
            continue
        meas = list(df[area_col])
        errs = [100 * (m - t) / t for m, t in zip(meas, truth)]
        print(f"  {label:14s} px/mm={state.get('px_per_mm')!s:>7.7}  true mm²={[round(t) for t in truth]}  measured={[round(float(m)) if m == m else None for m in meas]}  error%={[round(e,1) for e in errs]}")
