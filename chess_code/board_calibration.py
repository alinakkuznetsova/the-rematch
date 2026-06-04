#This file is for the one-time setup where I click the four corners of the board
#in the camera image, so the rest of the pipeline knows where the board is and can warp it
#into a clean top-down square.

#The reason I need this is because my iPhone camera is mounted overhead but it's never going
#to be perfectly square to the board - there will always be some tilt or angle, so I use
#perspective transform to flatten it out.

#Usage in the lab:
#    python -m chess_code.board_calibration
#A live camera window opens, I click the 4 corners in order (a8, h8, h1, a1), then press 's' to save.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


#where the calibration file gets saved + loaded from. main.py and occupancy_detector.py both
#load from the same path
CALIBRATION_PATH = Path("calibration/board_calibration.npz")

#size of the warped (top-down) board image. I picked 800x800 because that makes each chess
#square 100x100 pixels, which is plenty of pixels for the occupancy detector to read from
#must be divisible by 8 (8x8 grid of squares)
WARPED_SIZE = 800
SQUARE_SIZE = WARPED_SIZE // 8  # = 100

#labels for each corner, in the order I want the user to click them
#I had to think about this carefully - the order matters because it defines how the warp
#maps the camera image onto the output square
CORNER_LABELS = ["a8 (top-left)", "h8 (top-right)",
                 "h1 (bottom-right)", "a1 (bottom-left)"]



#using a dataclass here just to bundle the calibration data together neatly
#it gets created once and then loaded every time the main pipeline runs
@dataclass
class Calibration:

    #the four corner points I clicked, in click order [a8, h8, h1, a1]
    #shape (4, 2), stored as float32 because cv2.getPerspectiveTransform needs float32
    source_corners: np.ndarray

    #the 3x3 perspective transform matrix - this is what actually does the warping
    #computed from the four corners by cv2.getPerspectiveTransform
    transform: np.ndarray

    def warp(self, frame: np.ndarray) -> np.ndarray:
        #apply the perspective transform to whatever frame I pass in
        #returns a clean 800x800 top-down image of the board
        return cv2.warpPerspective(frame, self.transform,
                                   (WARPED_SIZE, WARPED_SIZE))

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 source_corners=self.source_corners,
                 transform=self.transform)

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        #raise a clear error if calibration hasn't been run yet
        #originally i just let numpy throw a FileNotFoundError but the message was confusing
        if not path.exists():
            raise FileNotFoundError(
                f"No calibration file at {path}. "
                "Run `python -m chess_code.board_calibration` first."
            )
        data = np.load(path)
        return cls(
            source_corners=data["source_corners"],
            transform=data["transform"],
        )



def _destination_corners() -> np.ndarray:
    #defines where each clicked corner should map to in the output (warped) image
    #the order has to match the order the user clicks them in, otherwise the warp comes out wrong
    #    a8 (top-left)     -> (0, 0)
    #    h8 (top-right)    -> (WARPED_SIZE, 0)
    #    h1 (bottom-right) -> (WARPED_SIZE, WARPED_SIZE)
    #    a1 (bottom-left)  -> (0, WARPED_SIZE)
    return np.array([
        [0,           0],
        [WARPED_SIZE, 0],
        [WARPED_SIZE, WARPED_SIZE],
        [0,           WARPED_SIZE],
    ], dtype=np.float32)


def compute_calibration(source_corners: np.ndarray) -> Calibration:
    #takes the four clicked points and computes the actual perspective transform
    #cv2.getPerspectiveTransform does the maths - i just need to give it the source and destination corners
    src = np.array(source_corners, dtype=np.float32)
    dst = _destination_corners()
    transform = cv2.getPerspectiveTransform(src, dst)
    return Calibration(source_corners=src, transform=transform)



def _draw_calibration_overlay(frame: np.ndarray,
                              points: list[tuple[int, int]],
                              hover: Optional[tuple[int, int]] = None
                             ) -> np.ndarray:
    #draws the clicked points + labels on top of the live camera feed so I can see what I've clicked
    display = frame.copy()

    #green circle for each clicked corner + its label
    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(display, (x, y), 12, (0, 255, 0), 2)
        cv2.putText(display, CORNER_LABELS[i], (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #once 2+ corners are clicked, draw lines between them to visualise the board outline
    #after the 4th point, close the loop back to the first to make a quadrilateral
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], (0, 255, 0), 1)
        if len(points) == 4:
            cv2.line(display, points[3], points[0], (0, 255, 0), 1)

    #black banner at the top with instructions for what to click next
    #i added this because without it I had to keep checking the terminal to see what corner was next,
    #which was annoying when I had to look up at the board to click
    next_corner = (CORNER_LABELS[len(points)]
                   if len(points) < 4 else "all corners clicked")
    banner = np.zeros((50, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner,
                f"Click corner {len(points) + 1}/4: {next_corner}"
                if len(points) < 4
                else "Press 's' to save, 'r' to reset, 'q' to quit",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    display = np.vstack([banner, display])

    return display


def _draw_warped_preview(warped: np.ndarray) -> np.ndarray:
    #adds a grid overlay on top of the warped board so I can verify the calibration is correct
    #if the grid lines up with the actual squares on the board, the calibration worked
    display = warped.copy()
    for i in range(1, 8):
        x = i * SQUARE_SIZE
        cv2.line(display, (x, 0), (x, WARPED_SIZE), (0, 255, 0), 1)
        cv2.line(display, (0, x), (WARPED_SIZE, x), (0, 255, 0), 1)

    cv2.putText(display,
                "Warped board preview -- 's' to save, 'r' to reset",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return display


def run_interactive_calibration(camera_index: int = 0) -> Optional[Calibration]:
    #main interactive loop - opens the camera, lets the user click 4 corners, then shows the warped preview
    #returns the Calibration if saved, or None if the user quit
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    clicked_points: list[tuple[int, int]] = []
    window_name = "Board calibration -- click 4 corners"

    def on_mouse(event, x, y, flags, userdata):
        #mouse callback - opencv calls this whenever the user clicks
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            #the displayed frame has a 50px banner stacked on top of it, so the y coords from
            #opencv are off by 50 - need to subtract that to get the real camera-image coords
            #(took me ages to figure out why my corners were saving wrong, this was why)
            real_y = y - 50
            if real_y >= 0:
                clicked_points.append((x, real_y))
                print(f"  Clicked {CORNER_LABELS[len(clicked_points) - 1]} "
                      f"at ({x}, {real_y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Board calibration started.")
    print("Click corners in order:")
    for label in CORNER_LABELS:
        print(f"  - {label}")
    print("Press 'r' to reset, 'q' to quit.")

    calibration: Optional[Calibration] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            if len(clicked_points) < 4:
                #still collecting clicks - show live feed with the overlay
                display = _draw_calibration_overlay(frame, clicked_points)
                cv2.imshow(window_name, display)
            else:
                #all 4 corners clicked - compute the calibration once, then show the warped preview
                #i only compute it once (when calibration is None) so it doesn't redo the maths every frame
                if calibration is None:
                    calibration = compute_calibration(np.array(clicked_points))
                warped = calibration.warp(frame)
                cv2.imshow(window_name, _draw_warped_preview(warped))

            #key handling - q quits, r resets the clicks, s saves
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit without saving.")
                return None
            if key == ord('r'):
                print("Reset.")
                clicked_points.clear()
                calibration = None
            if key == ord('s') and calibration is not None:
                calibration.save()
                print(f"Calibration saved to {CALIBRATION_PATH}")
                return calibration

    finally:
        #release the camera + close the windows no matter how the loop exits
        cap.release()
        cv2.destroyAllWindows()



def _main():
    import sys

    #allow passing the camera index as a CLI arg, since my iPhone is index 1 not 0 (built-in mac camera is 0)
    camera_index = 0
    if len(sys.argv) >= 2:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Camera index must be an integer; got {sys.argv[1]!r}")
            sys.exit(1)

    print(f"Opening camera {camera_index}...")
    calibration = run_interactive_calibration(camera_index)

    if calibration is not None:
        print("\nDone. The calibration is now loaded and saved.")
        print(f"  Source corners (in camera image):\n{calibration.source_corners}")
    else:
        print("\nNo calibration saved.")


if __name__ == "__main__":
    _main()