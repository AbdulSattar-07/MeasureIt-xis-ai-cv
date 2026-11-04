# Loom complete guide (≤ 5 minutes) — REQUIRED

Assessment: one Loom, **5 minutes or less**.  
If missing / over 5 min / answers don’t match the repo → **automatic rejection**.

**Do not read from notes on camera.** Memorize this once, then speak naturally.

---

## Before you press Record (2 minutes prep)

1. Close extra windows. Open only:
   - Cursor/VS Code on this repo
   - Terminal in `C:\Users\PMLS\Desktop\Asssistement`
   - Phone camera ready (earbuds case + bank card on table)
2. In terminal, test once (off camera):

```powershell
cd C:\Users\PMLS\Desktop\Asssistement
python -m inference.measure --image dataset\images\raw\both_045.jpeg --out outputs\loom_demo.png --json-out outputs\loom_demo.json
```

3. Open these files in the editor tabs:
   - `inference/measure.py`
   - `inference/geometry.py`
   - `calibration/undistort.py`
4. Have a ruler or calliper ready to show on the real earbuds case.
5. Loom: screen + camera (or screen + face). English or Urdu both OK if clear.

---

## Timeline (stick to this)

| Time | What you do |
|------|-------------|
| **0:00–1:00** | New photo → run command → show mm output → measure object with ruler on camera |
| **1:00–2:30** | Walk **one image** through code: undistort → model → reference → mm |
| **2:30–5:00** | Answer **K1 K2 K3 K4** out loud (no reading) |

Stop before 5:00. Export link. Put link in DoD + send with PR.

---

## Minute 1 — Live demo (script)

Say roughly:

> “I take a new photo of my earbuds case with the bank card in the same plane.”

Actions:
1. Show phone photo (or use existing `both_045.jpeg` if time tight — better: take fresh photo, save into folder, run on it).
2. Run:

```powershell
python -m inference.measure --image dataset\images\raw\both_045.jpeg --out outputs\loom_demo.png --json-out outputs\loom_demo.json
type outputs\loom_demo.json
```

3. Point to `width_mm`, `height_mm`, `confidence` in the JSON / overlay.
4. Put ruler/calliper on the physical case on camera: “Hand measurement is about X mm by Y mm.”

---

## Minutes 1–2.5 — Code walk (same image)

Open files and point while speaking:

1. **`inference/measure.py` ~line 127–134**  
   “Here we load the image and call `undistort_bgr` using `calibration/camera_params.npz`.”

2. **Same file** — model inference  
   “Then DeepLabV3 predicts masks for object (class 1) and reference card (class 2).”

3. **`inference/geometry.py` lines 42 and 75**  
   “`mm_per_pixel_from_reference` gets mm per pixel from the known card size.  
   `measure_object_mm` multiplies the object bbox by that scale.”

4. Show overlay `outputs/loom_demo.png` again.

---

## Minutes 2.5–5 — Four questions (memorize answers)

### K1 — Which file and line turns pixels into millimetres? What must be true?

**Say:**
> “Pixels become millimetres in `inference/geometry.py`.  
> Function `mm_per_pixel_from_reference` around line **42**, and `measure_object_mm` around line **75**.  
> For this to be valid, the card and object must be in the **same plane**, the card size must be correct (85.6 × 53.98 mm), the camera must stay locked, and the image must be **undistorted**.”

### K2 — Where is undistortion applied, and how much did skipping it change error?

**Say:**
> “Undistortion is applied in `inference/measure.py` around line **134**, calling `undistort_bgr` from `calibration/undistort.py`.  
> You can skip it with `--skip-undistort`.  
> On our table, width MAE went from about **28.2 mm** on raw images to **22.1 mm** after undistortion, and height from about **14.0 mm** to **13.4 mm**. So undistortion reduced error — those numbers are in `measurement/data/undistortion_proof.summary.json`.”

### K3 — One thing in your work you would reverse, and what it would cost?

**Say:**
> “I would reverse using **auto heuristic labels** instead of careful hand masks in CVAT.  
> Cost: I would spend more labelling time, but mask IoU would rise above 0.53 and millimetre error would drop. Right now MO-1 and MO-2 are missed because of those draft labels.”

### K4 — One thing you did not understand, and what you did about it?

**Say:**
> “I did not understand why DeepLab training crashed with BatchNorm and the MobileNet aux head channel size.  
> I read the error, checked torchvision’s aux head `in_channels`, set `drop_last=True` so batch size stayed >1, fixed the aux conv, and re-ran training until it finished.”

---

## Hard rules (auto-reject)

- Do **not** read from this file on camera.
- Do **not** describe code that is not in the repo.
- Keep video **≤ 5:00**.
- If you don’t know something: say what you tried — that is OK. Confident wrong answer fails.

---

## After recording

1. Copy Loom link.
2. Paste into `docs/Definition_of_Done.md` (Loom row).
3. Reply to assessment email/message with:
   - PR: https://github.com/AbdulSattar-07/MeasureIt-xis-ai-cv/pull/1
   - Loom: _(your link)_

---

## Optional helper during recording

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loom_demo.ps1
```
