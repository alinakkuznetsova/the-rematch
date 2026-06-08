#Projector calibration. The camera calibration tells the system where the board is in the
#camera image. This file does the opposite direction: it works out how to warp the rendered
#800x800 board image so that, once it goes through the projector's optics and angle, the
#projected squares land exactly on the squares of my physical printed board.

#There's no camera in this loop - I just look at the projected grid on the real board and
#nudge the four corners of the projected image until they sit on the four corners of the
#printed board. Corrects offset, scale, and keystone (projector not perfectly square-on) in one go.

#Run from repo root, with the projector connected and set to EXTEND (Win+P -> Extend):
#    python -m chess_code.projector_calibration
#the tool auto-detects the projector (the non-primary monitor) and opens the window there.
#if auto-detection picks the wrong screen, force a specific monitor by index:
#    python -m chess_code.projector_calibration --monitor 1
#or fall back to a manual window offset (x y) on the extended desktop:
#    python -m chess_code.projector_calibration --offset 1920 0

#requires: pip install screeninfo  (for monitor auto-detection)

#Controls (keyboard only - reliable under fullscreen, unlike the mouse):
#    1 / 2 / 3 / 4   -- select which corner to move (a8 / h8 / h1 / a1)
#    arrow keys      -- nudge the selected corner by 2px (fine)
#    w / a / s / d    -- nudge the selected corner by 20px (coarse: up/left/down/right)
#    r               -- reset corners to fill the whole projector frame
#    enter           -- save
#    q               -- quit without saving


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

from chess_code.board_calibration import WARPED_SIZE, SQUARE_SIZE  # 800, 100


#anchor the saved file to the repo root, same as the other calibration files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTOR_PATH = PROJECT_ROOT / "calibration" / "projector_calibration.npz"

#the source image is the rendered board canvas: 800x800, corners in order a8,h8,h1,a1
#(top-left, top-right, bottom-right, bottom-left) - same ordering as board_calibration
SOURCE_CORNERS = np.array([
    [0,           0],
    [WARPED_SIZE, 0],
    [WARPED_SIZE, WARPED_SIZE],
    [0,           WARPED_SIZE],
], dtype=np.float32)

CORNER_LABELS = ["a8 (top-left)", "h8 (top-right)",
                 "h1 (bottom-right)", "a1 (bottom-left)"]

PROJECTION_WINDOW = "projector calibration"



@dataclass
class ProjectorCalibration:
    #transform: 3x3 homography mapping the 800x800 board canvas onto the projector canvas
    #proj_size: (width, height) of the projector canvas the transform maps into
    #offset: (x, y) position of the projection window on the extended desktop
    transform: np.ndarray
    proj_size: tuple[int, int]
    offset: tuple[int, int]

    def warp(self, board_canvas: np.ndarray) -> np.ndarray:
        #take the 800x800 board image (pieces composited on black) and warp it into the
        #projector canvas so it lands aligned on the physical board
        w, h = self.proj_size
        return cv2.warpPerspective(board_canvas, self.transform, (w, h))

    def save(self, path: Path = PROJECTOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 transform=self.transform,
                 proj_size=np.array(self.proj_size),
                 offset=np.array(self.offset))

    @classmethod
    def load(cls, path: Path = PROJECTOR_PATH) -> "ProjectorCalibration":
        if not path.exists():
            raise FileNotFoundError(
                f"No projector calibration at {path}. "
                "Run `python -m chess_code.projector_calibration` first."
            )
        data = np.load(path)
        return cls(transform=data["transform"],
                   proj_size=tuple(int(v) for v in data["proj_size"]),
                   offset=tuple(int(v) for v in data["offset"]))


def detect_projector(force_index: int | None = None
                     ) -> tuple[int, int, int, int]:
    #find the projector screen automatically and return (offset_x, offset_y, width, height).
    #strategy: enumerate monitors; the projector is the non-primary monitor. if there's only
    #one monitor, or detection fails, raise so the caller can fall back to manual args.
    try:
        from screeninfo import get_monitors
    except ImportError:
        raise RuntimeError(
            "screeninfo not installed. Run `pip install screeninfo`, "
            "or pass --offset x y manually."
        )

    monitors = get_monitors()
    if not monitors:
        raise RuntimeError("No monitors detected.")

    print(f"Detected {len(monitors)} monitor(s):")
    for i, m in enumerate(monitors):
        primary = " (primary)" if getattr(m, "is_primary", False) else ""
        print(f"  [{i}] {m.width}x{m.height} at ({m.x},{m.y}){primary}")

    if force_index is not None:
        if force_index < 0 or force_index >= len(monitors):
            raise RuntimeError(f"--monitor {force_index} out of range "
                               f"(0..{len(monitors) - 1}).")
        m = monitors[force_index]
        print(f"Using forced monitor [{force_index}].")
        return (m.x, m.y, m.width, m.height)

    if len(monitors) == 1:
        raise RuntimeError(
            "Only one monitor detected. Connect the projector and set Win+P -> Extend, "
            "then re-run. Or pass --offset x y manually."
        )

    #pick the first non-primary monitor as the projector
    for i, m in enumerate(monitors):
        if not getattr(m, "is_primary", False):
            print(f"Auto-selected projector: monitor [{i}].")
            return (m.x, m.y, m.width, m.height)

    #no monitor flagged primary - fall back to the second one
    m = monitors[1]
    print("No primary flag found; using monitor [1] as projector.")
    return (m.x, m.y, m.width, m.height)


def compute_transform(dest_corners: np.ndarray) -> np.ndarray:
    #homography from the fixed 800x800 source corners to wherever I've dragged the dest corners
    return cv2.getPerspectiveTransform(SOURCE_CORNERS,
                                       dest_corners.astype(np.float32))


def _make_test_pattern() -> np.ndarray:
    #an 800x800 board grid on black, so I can see where the projection lands and line it up
    #with the printed board. red corner dots mark the four corners I'm aligning
    img = np.zeros((WARPED_SIZE, WARPED_SIZE, 3), dtype=np.uint8)
    #8x8 grid lines
    for i in range(9):
        x = min(i * SQUARE_SIZE, WARPED_SIZE - 1)
        cv2.line(img, (x, 0), (x, WARPED_SIZE - 1), (0, 180, 0), 1)
        cv2.line(img, (0, x), (WARPED_SIZE - 1, x), (0, 180, 0), 1)
    #thick outer border so the board edge is obvious
    cv2.rectangle(img, (1, 1), (WARPED_SIZE - 2, WARPED_SIZE - 2), (0, 255, 0), 3)
    #corner markers, inset slightly so they aren't clipped
    insets = [(18, 18), (WARPED_SIZE - 18, 18),
              (WARPED_SIZE - 18, WARPED_SIZE - 18), (18, WARPED_SIZE - 18)]
    for (x, y) in insets:
        cv2.circle(img, (x, y), 9, (0, 0, 255), -1)
    return img


def _draw_handles(canvas: np.ndarray,
                  dest_corners: np.ndarray,
                  selected: int) -> None:
    #draw the corner handles + labels + the currently selected one highlighted,
    #plus a short instruction banner. drawn directly onto the projector canvas
    for i, (x, y) in enumerate(dest_corners):
        x, y = int(x), int(y)
        colour = (0, 255, 255) if i == selected else (255, 255, 255)
        radius = 14 if i == selected else 9
        cv2.circle(canvas, (x, y), radius, colour, 2)
        cv2.circle(canvas, (x, y), 2, colour, -1)
        cv2.putText(canvas, CORNER_LABELS[i], (x + 18, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.putText(canvas,
                f"corner [{selected + 1}] {CORNER_LABELS[selected]}  "
                f"arrows=fine  wasd=coarse  [r]eset  [enter]save  [q]uit",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def _default_dest(proj_w: int, proj_h: int) -> np.ndarray:
    #start with the projected board filling the whole projector frame; I then pull the
    #corners in to match the physical board
    return np.array([
        [0,        0],
        [proj_w,   0],
        [proj_w,   proj_h],
        [0,        proj_h],
    ], dtype=np.float32)


def run_interactive_calibration(proj_w: int, proj_h: int,
                                offset_x: int, offset_y: int
                                ) -> ProjectorCalibration | None:
    pattern = _make_test_pattern()
    dest = _default_dest(proj_w, proj_h)
    selected = 0

    cv2.namedWindow(PROJECTION_WINDOW, cv2.WINDOW_NORMAL)
    #position the window on the projector, then make it fullscreen there
    cv2.moveWindow(PROJECTION_WINDOW, offset_x, offset_y)
    cv2.setWindowProperty(PROJECTION_WINDOW,
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nProjector calibration started.")
    print("Look at the PROJECTED grid on the physical board.")
    print("Select a corner with 1-4, then move it with arrows (fine) / wasd (coarse)")
    print("until the projected board corners sit on the printed board corners.")
    print("Press Enter to save, 'q' to quit.\n")

    FINE = 2
    COARSE = 20

    #arrow key codes from waitKeyEx differ by platform; cover common Windows + GTK values
    ARROWS = {
        2490368: (0, -FINE),  2621440: (0, FINE),    #up, down (Windows)
        2424832: (-FINE, 0),  2555904: (FINE, 0),    #left, right (Windows)
        65362:   (0, -FINE),  65364:   (0, FINE),    #up, down (GTK)
        65361:   (-FINE, 0),  65363:   (FINE, 0),    #left, right (GTK)
    }

    try:
        while True:
            transform = compute_transform(dest)
            warped = cv2.warpPerspective(pattern, transform, (proj_w, proj_h))
            _draw_handles(warped, dest, selected)
            cv2.imshow(PROJECTION_WINDOW, warped)

            key = cv2.waitKeyEx(20)
            if key == -1:
                continue

            k = key & 0xFF

            if k == ord('q'):
                print("Quit without saving.")
                return None
            if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                selected = k - ord('1')
            elif k == ord('r'):
                dest = _default_dest(proj_w, proj_h)
                print("Reset corners to full frame.")
            elif k in (13, 10):  #enter
                return ProjectorCalibration(transform=compute_transform(dest),
                                            proj_size=(proj_w, proj_h),
                                            offset=(offset_x, offset_y))
            #coarse nudge with wasd
            elif k == ord('w'):
                dest[selected][1] -= COARSE
            elif k == ord('s'):
                dest[selected][1] += COARSE
            elif k == ord('a'):
                dest[selected][0] -= COARSE
            elif k == ord('d'):
                dest[selected][0] += COARSE
            #fine nudge with arrow keys (full key code, not masked)
            elif key in ARROWS:
                dx, dy = ARROWS[key]
                dest[selected][0] += dx
                dest[selected][1] += dy

    finally:
        cv2.destroyAllWindows()


def _parse_args(argv: list[str]) -> dict:
    #tiny arg parser: --monitor N  |  --offset X Y [W H]
    opts: dict = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--monitor" and i + 1 < len(argv):
            opts["monitor"] = int(argv[i + 1]); i += 2
        elif a == "--offset" and i + 2 < len(argv):
            opts["offset"] = (int(argv[i + 1]), int(argv[i + 2])); i += 3
            #optional explicit size after offset
            if i + 1 < len(argv) and argv[i].lstrip("-").isdigit() \
                    and argv[i + 1].lstrip("-").isdigit():
                opts["size"] = (int(argv[i]), int(argv[i + 1])); i += 2
        else:
            i += 1
    return opts


def _main():
    opts = _parse_args(sys.argv[1:])

    if "offset" in opts:
        #manual mode - use given offset; size from --offset's trailing pair or sensible default
        offset_x, offset_y = opts["offset"]
        proj_w, proj_h = opts.get("size", (1920, 1200))
        print(f"Manual offset ({offset_x},{offset_y}), canvas {proj_w}x{proj_h}")
    else:
        #auto-detect (optionally forced to a specific monitor index)
        try:
            offset_x, offset_y, proj_w, proj_h = detect_projector(
                force_index=opts.get("monitor"))
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return
        print(f"Projector at ({offset_x},{offset_y}), resolution {proj_w}x{proj_h}")

    cal = run_interactive_calibration(proj_w, proj_h, offset_x, offset_y)

    if cal is not None:
        cal.save()
        print(f"\nProjector calibration saved to {PROJECTOR_PATH}")
        print("main.py will now warp the projection to land on the board.")
    else:
        print("\nNo projector calibration saved.")


if __name__ == "__main__":
    _main()