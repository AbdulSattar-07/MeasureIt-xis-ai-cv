"""Show checkerboard fullscreen so you can photograph it with your phone (no printer).

Measure ONE square on the SCREEN with a ruler (mm), put that number in
calibration/board_config.json -> square_size_mm, then take 20+ phone photos
of this window from different angles into calibration/images/.
"""

from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "calibration" / "print_me_checkerboard.png"


def main() -> None:
    img = cv2.imread(str(BOARD), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Missing {BOARD}")
    # upscale for screen sharpness
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    win = "CALIB BOARD - photograph this with phone | ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    print("Fullscreen board ready.")
    print("1) Ruler se screen pe 1 square mm mein measure karo")
    print("2) board_config.json me square_size_mm update")
    print("3) Phone se 20+ photos lo -> calibration/images/")
    print("ESC to close")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
