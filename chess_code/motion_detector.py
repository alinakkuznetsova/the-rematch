#This file watches the camera feed and figures out when the board is "settled" - i.e. when
#i've finished moving a piece and the rest of the pipeline can safely try to read the position.

#The state machine has three states:
#    MOVING    -- significant motion happening right now (my hand moving a piece)
#    SETTLING  -- motion just stopped but we wait a bit before declaring it stable,
#                 because i sometimes pause mid-move (hovering over a piece, hesitating)
#                 and we don't want to trigger a read during those false pauses
#    STABLE    -- no motion for a full settling period (1 second by default)

#The key trigger is the SETTLING -> STABLE transition. That's the moment main.py reads the
#board and figures out what move i played. just_became_stable fires only once per move,
#not every frame the board is stable.

#Run directly to see a live debug window for tuning thresholds:
#    python motion_detector.py
#Press 'q' in the debug window to quit.


import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class MotionState(Enum):
    MOVING = auto()    #motion right now
    SETTLING = auto()  #motion just stopped, waiting to confirm
    STABLE = auto()    #no motion for the full settling period


@dataclass
class MotionResult:
    #what the detector returns each frame
    state: MotionState
    motion_level: float       #0.0 = nothing moving, higher = more motion
    just_became_stable: bool  #True only on the single frame STABLE is first entered
                              #this is what main.py uses as the move-read trigger
    seconds_since_motion: float


class MotionDetector:
    #detects motion between consecutive frames using simple frame differencing
    #(subtracting one frame from the next and seeing how much changed)

    #tuning parameters - the defaults work in my lighting but might need adjusting in the lab:
    #  motion_threshold: how different a pixel must be to count as "changed" (0-255 grayscale)
    #                    higher = less sensitive to small changes
    #  motion_area_fraction: what fraction of the frame must be changed to call it motion
    #                        0.005 = 0.5% of pixels. small enough to catch a moving hand,
    #                        big enough to ignore camera noise
    #  settle_seconds: how long the frame must be still before we call STABLE
    #                  1 second feels right - long enough to ignore hesitations, short enough
    #                  that the engine response doesn't feel sluggish

    def __init__(self,
                 motion_threshold: int = 25,
                 motion_area_fraction: float = 0.005,
                 settle_seconds: float = 1.0):
        self.motion_threshold = motion_threshold
        self.motion_area_fraction = motion_area_fraction
        self.settle_seconds = settle_seconds

        self._previous_gray: Optional[np.ndarray] = None
        self._state: MotionState = MotionState.STABLE
        self._last_motion_time: float = time.time()

        #stored for the debug visualisation
        self.last_motion_mask: Optional[np.ndarray] = None
        self.last_motion_level: float = 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> MotionResult:
        #call once per frame in the main loop. returns current state + whether
        #this is the special frame where we just became stable

        #convert to greyscale + apply a Gaussian blur. the blur is important because camera
        #sensors produce per-pixel noise that would otherwise register as "motion" even on
        #a completely still scene. blurring averages that noise out
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        #first frame ever - nothing to compare to, just store it and report no motion
        if self._previous_gray is None:
            self._previous_gray = gray
            return MotionResult(
                state=self._state,
                motion_level=0.0,
                just_became_stable=False,
                seconds_since_motion=0.0,
            )

        #absolute difference between current frame and previous frame
        #for each pixel: how much brighter/darker did it get? returns a greyscale image
        diff = cv2.absdiff(self._previous_gray, gray)

        #threshold: any pixel that changed by more than motion_threshold counts as motion
        #result is a binary mask: 255 where motion, 0 where still
        _, motion_mask = cv2.threshold(
            diff, self.motion_threshold, 255, cv2.THRESH_BINARY
        )

        #dilate the mask to merge nearby motion blobs together. without this, a moving hand
        #shows up as 50 disconnected pixels instead of one coherent region, which messes up
        #the area-fraction calculation
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        #count what fraction of the frame is currently moving
        moving_pixels = cv2.countNonZero(motion_mask)
        total_pixels = motion_mask.size
        motion_fraction = moving_pixels / total_pixels

        #now run the state machine
        now = time.time()
        is_motion_now = motion_fraction > self.motion_area_fraction
        just_became_stable = False

        if is_motion_now:
            #any motion at all kicks us to MOVING and resets the settle timer
            self._state = MotionState.MOVING
            self._last_motion_time = now
        else:
            #no motion this frame. depending on previous state, either:
            #  - were MOVING, now go to SETTLING
            #  - were SETTLING, check if enough time has passed to go STABLE
            #  - were already STABLE, just stay stable
            seconds_since = now - self._last_motion_time

            if self._state == MotionState.MOVING:
                self._state = MotionState.SETTLING
            elif self._state == MotionState.SETTLING:
                if seconds_since >= self.settle_seconds:
                    self._state = MotionState.STABLE
                    #fire just_became_stable on this single frame - the main loop uses this
                    #to trigger the move read exactly once per move
                    just_became_stable = True

        #save current frame as previous for next iteration, also stash for debug viewer
        self._previous_gray = gray
        self.last_motion_mask = motion_mask
        self.last_motion_level = motion_fraction

        return MotionResult(
            state=self._state,
            motion_level=motion_fraction,
            just_became_stable=just_became_stable,
            seconds_since_motion=now - self._last_motion_time,
        )


def run_debug_window(camera_index: int = 0):
    #live debug visualisation - lets me see exactly what the detector is reading
    #i used this a lot when tuning the thresholds in different lighting conditions
    #
    #the window shows:
    #  - live camera feed on the left
    #  - motion mask on the right (white = motion detected, black = still)
    #  - top banner with current state + motion level

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    detector = MotionDetector()
    print("Motion detector running. Press 'q' to quit.")
    print(f"Settle time: {detector.settle_seconds}s | "
          f"Threshold: {detector.motion_threshold} | "
          f"Area fraction: {detector.motion_area_fraction}")

    #colour-code the state so it's instantly readable in the debug window
    state_colours = {
        MotionState.MOVING: (0, 0, 255),     #red
        MotionState.SETTLING: (0, 165, 255), #orange
        MotionState.STABLE: (0, 255, 0),     #green
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            result = detector.process_frame(frame)

            #skip the first frame - the motion mask isn't built yet because process_frame
            #returns early when there's no previous frame to compare against
            if detector.last_motion_mask is None:
                continue

            #stitch the camera feed and motion mask side by side
            mask_bgr = cv2.cvtColor(detector.last_motion_mask, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]))
            display = np.hstack([frame, mask_bgr])

            #black banner at top with current state + motion level numbers
            colour = state_colours[result.state]
            cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(display, result.state.name, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2)
            cv2.putText(display,
                        f"motion: {result.motion_level:.4f}   "
                        f"still for: {result.seconds_since_motion:.2f}s",
                        (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            #log to terminal whenever the board just settled - useful for debugging timing issues
            if result.just_became_stable:
                print(f"[{time.strftime('%H:%M:%S')}] STABLE -- "
                      "board can be read now")

            cv2.imshow("motion detector debug", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    camera_index = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    run_debug_window(camera_index)