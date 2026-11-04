"""Deep-analyze board+card scale photos: crop screen, find corners, estimate mm."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration" / "measure"
CARD_LONG_MM = 85.60
CARD_SHORT_MM = 53.98


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img: np.ndarray, quad: np.ndarray, out_w: int = 1200, out_h: int = 800):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32
    )
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (out_w, out_h)), m


def find_board(gray: np.ndarray):
    for cx, cy in [(9, 6), (8, 5), (7, 5), (10, 7), (6, 4), (7, 4)]:
        for src in (gray, cv2.equalizeHist(gray)):
            ok, corners = cv2.findChessboardCorners(
                src,
                (cx, cy),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ok:
                corners = cv2.cornerSubPix(
                    src,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
                )
                return (cx, cy), corners
    return None, None


def square_px_from_corners(corners: np.ndarray, cfg: tuple[int, int]) -> float:
    cx, cy = cfg
    grid = corners.reshape(cy, cx, 2)
    dists = []
    for y in range(cy):
        for x in range(cx - 1):
            dists.append(np.linalg.norm(grid[y, x + 1] - grid[y, x]))
    for y in range(cy - 1):
        for x in range(cx):
            dists.append(np.linalg.norm(grid[y + 1, x] - grid[y, x]))
    return float(np.median(dists))


def detect_blue_card(img: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    card = max(cnts, key=cv2.contourArea)
    h, w = img.shape[:2]
    if cv2.contourArea(card) < 0.008 * h * w:
        return None, None
    rect = cv2.minAreaRect(card)
    box = cv2.boxPoints(rect).astype(int)
    rw, rh = rect[1]
    return {
        "long_px": float(max(rw, rh)),
        "short_px": float(min(rw, rh)),
        "angle": float(rect[2]),
        "area": float(cv2.contourArea(card)),
    }, box


def process(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"unreadable {path}")
    h, w = img.shape[:2]
    vis = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.15 * h * w:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and (best is None or area > best[0]):
            best = (area, approx)

    if best:
        quad = order_quad(best[1])
        cv2.polylines(vis, [quad.astype(int)], True, (0, 255, 0), 3)
        crop, _ = warp_quad(img, quad)
        crop_mode = "perspective_screen"
    else:
        crop = img[int(0.02 * h) : int(0.78 * h), int(0.05 * w) : int(0.95 * w)].copy()
        crop_mode = "fallback_roi"

    stem = path.stem.replace("WhatsApp Image ", "scale_")
    crop_path = OUT / f"{stem}_screen_crop.png"
    cv2.imwrite(str(crop_path), crop)

    cgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cfg, corners = find_board(cgray)
    # also try original (sometimes warp hurts)
    if cfg is None:
        cfg, corners = find_board(gray)
        board_img = img
        board_from = "original"
    else:
        board_img = crop
        board_from = "crop"

    square_px = None
    if cfg is not None:
        square_px = square_px_from_corners(corners, cfg)
        board_vis = board_img.copy()
        cv2.drawChessboardCorners(board_vis, cfg, corners, True)
        cv2.imwrite(str(OUT / f"{stem}_corners.png"), board_vis)

    card, box = detect_blue_card(img)
    if box is not None:
        cv2.drawContours(vis, [box], 0, (255, 0, 0), 3)

    estimate = {}
    method = None
    if square_px and card:
        sq_long = CARD_LONG_MM * (square_px / card["long_px"])
        sq_short = CARD_SHORT_MM * (square_px / card["short_px"])
        estimate = {
            "square_mm_from_card_long": round(sq_long, 3),
            "square_mm_from_card_short": round(sq_short, 3),
            "square_mm_avg": round(0.5 * (sq_long + sq_short), 3),
            "squares_spanned_by_card_long": round(card["long_px"] / square_px, 3),
            "squares_spanned_by_card_short": round(card["short_px"] / square_px, 3),
        }
        method = "corners_plus_card_px"
    elif card:
        # visual span priors from these photos (~3.5 long, ~2.2 short)
        e1 = CARD_LONG_MM / 3.5
        e2 = CARD_SHORT_MM / 2.2
        estimate = {
            "square_mm_from_card_long": round(e1, 3),
            "square_mm_from_card_short": round(e2, 3),
            "square_mm_avg": round(0.5 * (e1 + e2), 3),
        }
        method = "visual_span_prior"

    cv2.imwrite(str(OUT / f"{stem}_annotate.png"), vis)
    return {
        "image": path.name,
        "crop_mode": crop_mode,
        "board_from": board_from if cfg else None,
        "chessboard_inner_corners": list(cfg) if cfg else None,
        "square_px": square_px,
        "card": card,
        "estimate": estimate,
        "method": method,
        "outputs": {
            "crop": crop_path.name,
            "annotate": f"{stem}_annotate.png",
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(ROOT.glob("WhatsApp Image*.jpeg"))
    if not paths:
        # already moved?
        paths = sorted((ROOT / "calibration" / "measure").glob("scale_*.jpeg"))
        paths = [p for p in paths if "_screen" not in p.name and "_annotate" not in p.name and "_corners" not in p.name]
    print("inputs:", [p.name for p in paths])
    results = [process(p) for p in paths]

    valid = [r for r in results if r.get("estimate")]
    if not valid:
        raise SystemExit("No usable estimate")

    def score(r):
        est = r["estimate"]
        disagree = abs(
            est.get("square_mm_from_card_long", 0) - est.get("square_mm_from_card_short", 0)
        )
        return (0 if r["chessboard_inner_corners"] else 1, disagree)

    best = sorted(valid, key=score)[0]
    sq = best["estimate"]["square_mm_avg"]
    cfg_path = ROOT / "calibration" / "board_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if best["chessboard_inner_corners"]:
        cfg["inner_corners_x"] = int(best["chessboard_inner_corners"][0])
        cfg["inner_corners_y"] = int(best["chessboard_inner_corners"][1])
    cfg["square_size_mm"] = round(float(sq), 2)
    cfg["notes"] = (
        f"Estimated from on-screen board + UBL card (ID-1 {CARD_LONG_MM}x{CARD_SHORT_MM} mm). "
        f"source={best['image']} method={best['method']} "
        f"long={best['estimate'].get('square_mm_from_card_long')} "
        f"short={best['estimate'].get('square_mm_from_card_short')}."
    )
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    report = {
        "results": results,
        "chosen": best,
        "updated_config": cfg,
    }
    (OUT / "measure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    # move originals into measure folder
    for p in sorted(ROOT.glob("WhatsApp Image*.jpeg")):
        dest = OUT / p.name.replace("WhatsApp Image ", "scale_")
        if p.resolve() != dest.resolve():
            p.replace(dest)
            print("moved", dest.name)
    for z in ROOT.glob("WhatsApp*.zip"):
        z.unlink()
        print("removed", z.name)


if __name__ == "__main__":
    main()
