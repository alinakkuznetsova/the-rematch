#This is the conductor file - the one I actually run for the performance.
#It ties everything together: loads calibration data, opens the camera, starts the engine,
#and runs the main perception-action loop:

#    camera frame -> motion detector -> occupancy detector ->
#        game state -> engine response -> renderer -> projection

#Usage (must run board_calibration and occupancy_detector first to set up the calibration files;
#run projector_calibration too so the projected pieces land on the physical board):
#    python -m chess_code.main /opt/homebrew/bin/stockfish

#Key bindings during performance:
#    r  -- resign (i give up; engine wins)
#    p  -- pause/unpause the clock
#    q  -- quit immediately (no save)


from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chess_code.board_calibration import Calibration, WARPED_SIZE, SQUARE_SIZE
from chess_code.occupancy_detector import OccupancyBaseline, OccupancyDetector
from chess_code.motion_detector import MotionDetector, MotionState
from chess_code.game_state import GameState
from chess_code.renderer import Renderer, CANVAS_SIZE
from chess_code.projector_calibration import ProjectorCalibration


#repo root (this file is in chess_code/, so parent.parent is performing-project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent



#Deep Blue's estimated 1997 strength - this is the whole point conceptually so it stays at 2750 for the performance
#i drop this to 2000-2200 for rehearsal so i can actually study the position without losing instantly
ENGINE_ELO = 2750

#how long the engine thinks per move. 3s - long enough for the silence to register
#as the machine "thinking", short enough that the piece doesn't drag
ENGINE_THINK_SECONDS = 3.0

#tournament time control per side, in seconds. 10 minutes each
CLOCK_SECONDS_PER_SIDE = 10 * 60

#occupancy detection threshold. tuned in the lab on 08/06 against the performance start position -
#0.20 was the value where all 15 black pieces read solid (including dark-square pieces) with no false positives
OCCUPANCY_THRESHOLD = 0.20



@dataclass
class Clock:
    #standard chess clock - each side has its own countdown
    #when it's my turn, my clock ticks down. when i make a move, my clock stops + engine's starts
    seconds_per_side: float = CLOCK_SECONDS_PER_SIDE
    black_remaining: float = field(init=False)
    white_remaining: float = field(init=False)
    active_side: Optional[chess.Color] = None
    last_tick: float = field(default_factory=time.time)
    paused: bool = False

    def __post_init__(self):
        self.black_remaining = self.seconds_per_side
        self.white_remaining = self.seconds_per_side

    def start_for(self, side: chess.Color) -> None:
        #start the clock running for one side. _settle first so we don't lose any time
        #from the previous active side
        self._settle()
        self.active_side = side
        self.last_tick = time.time()

    def stop(self) -> None:
        self._settle()
        self.active_side = None

    def toggle_pause(self) -> None:
        #pause/resume. when resuming, reset last_tick so we don't deduct the paused time
        if self.paused:
            self.last_tick = time.time()
            self.paused = False
        else:
            self._settle()
            self.paused = True

    def update(self) -> None:
        #called every loop iteration - deducts elapsed time from whichever side's clock is active
        if self.paused or self.active_side is None:
            return
        self._settle()

    def _settle(self) -> None:
        #actually does the time deduction. separated out so start_for/stop can also call it
        #to make sure no time is lost between transitions
        if self.paused or self.active_side is None:
            self.last_tick = time.time()
            return
        now = time.time()
        elapsed = now - self.last_tick
        if self.active_side == chess.BLACK:
            self.black_remaining = max(0.0, self.black_remaining - elapsed)
        else:
            self.white_remaining = max(0.0, self.white_remaining - elapsed)
        self.last_tick = now

    def fmt(self, seconds: float) -> str:
        #format seconds as M:SS for the debug overlay
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    def flag_fallen(self) -> Optional[chess.Color]:
        #returns whichever side has run out of time, or None if both still have time
        #in chess this is called "flag fall" - on physical clocks a small flag falls when time runs out
        if self.black_remaining <= 0:
            return chess.BLACK
        if self.white_remaining <= 0:
            return chess.WHITE
        return None



#two separate windows - one for the projector (clean, just the projected pieces on black bg)
#and one for my debug view (camera feed + warped board + occupancy overlay + clock)
#the debug window stays on my laptop screen, the projection window goes fullscreen on the projector
DEBUG_WINDOW = "conductor [debug]"
PROJECTION_WINDOW = "conductor [projection]"


def make_projection_frame(piece_layer_rgba: np.ndarray) -> np.ndarray:
    #takes the renderer's transparent piece layer (BGRA) and composites it onto pure black
    #black background is important - the projector treats black as "emit no light", so the physical
    #board shows through and only the pieces appear as light
    #returns an 800x800 board-space image; projector warp (if any) is applied by the caller
    alpha = piece_layer_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = piece_layer_rgba[:, :, :3].astype(np.float32)
    background = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.float32)
    composited = rgb * alpha + background * (1 - alpha)
    return composited.astype(np.uint8)


#retro monospace font for the projected status messages (same family as the intro video).
#the ttf is bundled in the repo root next to chess_code/. fall back to a default if it's missing.
_FONT_PATH = PROJECT_ROOT / "VT323-Regular.ttf"


def _load_font(size: int):
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_status_overlay(projection_bgr: np.ndarray,
                        lines: list[str],
                        face_white: bool) -> np.ndarray:
    #draw centred retro-font text onto the 800x800 board-space projection (before flip/warp).
    #lines: one or more lines of text, stacked and centred in the middle of the board.
    #face_white: if True the text is rotated 180 deg so it reads upright from White's side of the
    #board; if False it reads upright from Black's side (where the performer sits). this is applied
    #in board space - the later 180 deg projection flip is accounted for so the side ends up correct.
    if not lines:
        return projection_bgr

    #render the text on its own transparent layer with PIL (for the TTF font), then composite
    layer = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _load_font(96 if len(lines) == 1 else 84)

    #measure total block height to centre vertically
    line_heights = []
    line_widths = []
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    gap = 20
    total_h = sum(line_heights) + gap * (len(lines) - 1)
    y = (CANVAS_SIZE - total_h) // 2

    for ln, lw, lh in zip(lines, line_widths, line_heights):
        x = (CANVAS_SIZE - lw) // 2
        #soft glow: draw a dim wider pass then the bright text, for a CRT feel
        draw.text((x, y), ln, font=font, fill=(120, 255, 120, 120))
        draw.text((x, y), ln, font=font, fill=(220, 255, 220, 255))
        y += lh + gap

    #the projection is flipped 180 later (cv2.flip(-1)). so to make text read upright from
    #BLACK's side after that flip, we pre-rotate it 180 here; to read from WHITE's side we leave it.
    if not face_white:
        layer = layer.rotate(180)

    #composite the text layer onto the projection
    text_rgba = np.array(layer)
    alpha = text_rgba[:, :, 3:4].astype(np.float32) / 255.0
    #PIL is RGB, projection is BGR - swap channels
    text_bgr = text_rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    out = projection_bgr.astype(np.float32) * (1 - alpha) + text_bgr * alpha
    return out.astype(np.uint8)


def status_lines_for_board(board: chess.Board) -> tuple[list[str], bool]:
    #work out what status text (if any) to project, and which side it should face.
    #returns (lines, face_white). empty lines => nothing to draw.
    #
    #checkmate: two lines, the side that delivered mate wins. faces the winner.
    #check (not mate): one line naming the side in check. faces the side in check.
    if board.is_checkmate():
        #the side to move has been mated, so the OTHER side wins
        winner_white = (board.turn == chess.BLACK)
        winner = "WHITE WINS" if winner_white else "BLACK WINS"
        return ["CHECKMATE", winner], winner_white
    if board.is_check():
        in_check_white = (board.turn == chess.WHITE)
        side = "WHITE IN CHECK" if in_check_white else "BLACK IN CHECK"
        return [side], in_check_white
    return [], False


def draw_debug_panel(camera_frame: np.ndarray,
                     warped: np.ndarray,
                     motion_state: MotionState,
                     occupied: set[chess.Square],
                     game: GameState,
                     clock: Clock,
                     status_text: str) -> np.ndarray:
    #builds the debug window with: raw camera | warped board with occupancy overlay | clocks + state
    #this is just for me to see what the system is reading - the audience never sees this

    #left side: raw camera feed (resized to match the warped board size for layout)
    cam_small = cv2.resize(camera_frame, (WARPED_SIZE, WARPED_SIZE))

    #right side: the warped board with green boxes overlaid on detected-occupied squares
    overlay = warped.copy()
    for sq in occupied:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        x = file_idx * SQUARE_SIZE
        y = (7 - rank_idx) * SQUARE_SIZE
        cv2.rectangle(overlay, (x + 5, y + 5),
                      (x + SQUARE_SIZE - 5, y + SQUARE_SIZE - 5),
                      (0, 255, 0), 2)

    panel = np.hstack([cam_small, overlay])

    #top banner with status info - motion state, whose turn it is, both clocks, current status text
    banner_h = 80
    banner = np.zeros((banner_h, panel.shape[1], 3), dtype=np.uint8)
    #motion state colour-coded: red while moving, orange settling, green stable
    state_colour = {
        MotionState.MOVING:   (0, 0, 255),
        MotionState.SETTLING: (0, 165, 255),
        MotionState.STABLE:   (0, 255, 0),
    }[motion_state]
    cv2.putText(banner, f"motion: {motion_state.name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colour, 2)
    turn_text = "Black to move" if game.status().turn == chess.BLACK else "White to move"
    cv2.putText(banner, turn_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(banner,
                f"BLACK {clock.fmt(clock.black_remaining)}",
                (panel.shape[1] - 360, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(banner,
                f"WHITE {clock.fmt(clock.white_remaining)}",
                (panel.shape[1] - 360, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(banner, status_text,
                (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return np.vstack([banner, panel])



def run(engine_path: str, camera_index: int = 0):
    #load the calibration files. these were created earlier by board_calibration.py and occupancy_detector.py
    #if either is missing, print a clear message telling me to run those first instead of crashing weirdly
    try:
        board_cal = Calibration.load()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run `python -m chess_code.board_calibration` first.")
        return

    try:
        baseline = OccupancyBaseline.load()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run `python -m chess_code.occupancy_detector` first.")
        return

    #projector calibration is optional - if it's missing, fall back to projecting the raw
    #800x800 board image fullscreen (you then align by physically moving the projector).
    #if present, the projection gets warped so it lands square on the physical board.
    try:
        proj_cal = ProjectorCalibration.load()
        print("Projector calibration loaded - projection will be warped onto the board.")
    except FileNotFoundError:
        proj_cal = None
        print("No projector calibration found - projecting raw board image. "
              "Run `python -m chess_code.projector_calibration` to align it in software.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return

    #instantiate all the components - they were each tested individually before being plugged together here
    motion = MotionDetector()
    occupancy = OccupancyDetector(baseline, OCCUPANCY_THRESHOLD)
    renderer = Renderer()
    clock = Clock()

    #use GameState as a context manager so the stockfish subprocess gets cleaned up no matter what
    with GameState(engine_path, engine_elo=ENGINE_ELO,
                   engine_think_seconds=ENGINE_THINK_SECONDS) as game:

        cv2.namedWindow(PROJECTION_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)

        #if we have projector calibration, put the projection window on the projector at the
        #same offset used during calibration, and make it fullscreen there so the geometry matches
        if proj_cal is not None:
            cv2.moveWindow(PROJECTION_WINDOW, proj_cal.offset[0], proj_cal.offset[1])
            cv2.setWindowProperty(PROJECTION_WINDOW,
                                  cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        #start the clock for black (me) since I move first in this position
        clock.start_for(chess.BLACK)
        status = "Position loaded -- Black to move. Refuse 7...h6."
        #sticky timeout overlay - None until a clock flags, then holds the "TIME / X WINS" message
        timeout_lines: Optional[list[str]] = None
        timeout_face_white: bool = False
        print("\n" + status)
        print("Press 'r' to resign, 'p' to pause, 'q' to quit.\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                #grab a frame, check for motion, warp it, detect which squares are occupied
                #all of this runs every iteration even when nothing's happening
                motion_result = motion.process_frame(frame)
                warped = board_cal.warp(frame)
                occupied = occupancy.detect(warped)

                #the key trigger: when the board JUST became stable (i.e. i just finished moving),
                #AND it's my turn AND the game isn't over, try to infer which move I made
                #the "just_became_stable" thing only fires once per move, not every frame the board is stable
                if (motion_result.just_became_stable
                        and game.status().turn == chess.BLACK
                        and not game.status().is_over):
                    move = game.infer_black_move(occupied)
                    if move is None:
                        #CV read didn't match any legal move - probably a misread or a bumped piece
                        #just wait for me to fix the position; we'll try again next time the board settles
                        status = ("Board reading didn't match a legal move. "
                                  "Adjust pieces and wait.")
                    else:
                        #move recognised - apply it, switch clock to white, then ask engine for its move
                        black_result = game.apply_black_move(move)
                        status = f"You played: {black_result.san}"
                        print(status)
                        clock.start_for(chess.WHITE)

                        if not game.status().is_over:
                            print("  Engine thinking...")
                            white_result = game.request_white_move()
                            status = (f"You: {black_result.san}   "
                                      f"Engine: {white_result.san}")
                            print(f"  Engine played: {white_result.san}")
                            clock.start_for(chess.BLACK)

                clock.update()

                #render the white pieces onto a transparent canvas, then composite onto black for projection
                piece_layer = renderer.render_from_board(game.board)
                projection = make_projection_frame(piece_layer)

                #status overlay (check / checkmate / time out) in retro font, centred, facing the
                #relevant side. a sticky timeout message (set when a clock flags) takes priority;
                #otherwise derive check/checkmate live from the board.
                if timeout_lines is not None:
                    projection = draw_status_overlay(projection, timeout_lines, timeout_face_white)
                else:
                    ov_lines, ov_face_white = status_lines_for_board(game.board)
                    if ov_lines:
                        projection = draw_status_overlay(projection, ov_lines, ov_face_white)

                #the physical board is rotated 180 degrees from the renderer's image convention
                #(I sit on Black's side), so flip both axes in board space before the warp
                projection = cv2.flip(projection, -1)  # -1 = flip both axes = 180 rotation
                #if calibrated, warp the 800x800 board image into the projector canvas so it
                #lands aligned on the physical board. otherwise project the raw board image
                if proj_cal is not None:
                    projection = proj_cal.warp(projection)
                cv2.imshow(PROJECTION_WINDOW, projection)

                #the debug window (for my eyes only, not on the projector)
                debug = draw_debug_panel(frame, warped, motion_result.state,
                                         occupied, game, clock, status)
                cv2.imshow(DEBUG_WINDOW, debug)

                #check end conditions: game over, clock flag fall, or key press
                game_status = game.status()
                if game_status.is_over:
                    status = game_status.result_text
                    print(f"\n{status}")
                    print("Press any key to exit.")
                    cv2.waitKey(0)
                    break

                fallen = clock.flag_fallen()
                if fallen is not None:
                    #whoever's clock fell loses on time; the other side wins. "TIME" / "X WINS".
                    side = "Black" if fallen == chess.BLACK else "White"
                    print(f"\nFlag fall: {side}'s clock ran out.")
                    winner_white = (fallen == chess.BLACK)
                    timeout_lines = ["TIME", "WHITE WINS" if winner_white else "BLACK WINS"]
                    timeout_face_white = winner_white
                    #render one final frame so the timeout message is projected, then hold
                    piece_layer = renderer.render_from_board(game.board)
                    final_proj = make_projection_frame(piece_layer)
                    final_proj = draw_status_overlay(final_proj, timeout_lines, timeout_face_white)
                    final_proj = cv2.flip(final_proj, -1)
                    if proj_cal is not None:
                        final_proj = proj_cal.warp(final_proj)
                    cv2.imshow(PROJECTION_WINDOW, final_proj)
                    print("Press any key to exit.")
                    cv2.waitKey(0)
                    break

                #key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit.")
                    break
                if key == ord('p'):
                    clock.toggle_pause()
                    status = "Paused." if clock.paused else "Resumed."
                #only allow resign on my turn - resigning on the engine's turn doesn't make sense
                if key == ord('r') and game.status().turn == chess.BLACK:
                    print("\nYou resign. Engine wins.")
                    break

        finally:
            #always release the camera + close the windows, even if something above crashed
            cap.release()
            cv2.destroyAllWindows()


def _main():
    if len(sys.argv) < 2:
        print("Usage: python -m chess_code.main /path/to/stockfish [camera_index]")
        sys.exit(1)
    engine_path = sys.argv[1]
    #camera index defaults to 0 (built-in webcam) but i pass 1 in the lab because the iPhone via Camo is index 1
    camera_index = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    run(engine_path, camera_index)


if __name__ == "__main__":
    _main()