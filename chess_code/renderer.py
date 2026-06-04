#This file produces the image that actually gets projected onto the physical board.
#The image is 800x800 with a transparent background - when sent to the projector with black bg behind,
#only the white pieces appear as light, and the audience's physical wooden board shows through underneath.

#Coordinate system:
#    output image is 800x800 pixels (matches WARPED_SIZE in board_calibration.py)
#    board divided into 8x8 squares of 100x100 each
#    chess square (file, rank) maps to pixel:
#        pixel_x = file * 100
#        pixel_y = (7 - rank) * 100    <- rank flipped because image y goes down but chess ranks go up

#Test from terminal - saves a preview PNG so i can see what the rendering looks like
#without needing the projector hooked up:
#    python -m chess_code.renderer


from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import cv2
import numpy as np



#these have to match the warped board size set in board_calibration.py
#otherwise the projected pieces wouldn't line up with the physical squares
CANVAS_SIZE = 800
SQUARE_SIZE = CANVAS_SIZE // 8  # = 100

#folder where the piece PNGs live. structure is just pieces/king_white.png, pieces/queen_white.png etc
PIECES_DIR = Path("pieces")

#mapping from chess piece type to filename. i only have white pieces because that's all i project -
#my own black pieces are physical so they don't need rendering. if i ever want to project black too
#i'd just add them here
PIECE_FILENAMES = {
    chess.KING:   "king_white.png",
    chess.QUEEN:  "queen_white.png",
    chess.ROOK:   "rook_white.png",
    chess.BISHOP: "bishop_white.png",
    chess.KNIGHT: "knight_white.png",
    chess.PAWN:   "pawn_white.png",
}



def _load_piece(path: Path) -> np.ndarray:
    #load a PNG keeping its alpha channel intact. returns HxWx4 BGRA image
    if not path.exists():
        raise FileNotFoundError(f"Missing piece image: {path}")

    #IMREAD_UNCHANGED is the important flag here - the default cv2.imread strips the alpha
    #channel and you end up with opaque white blocks instead of nicely cut-out pieces
    piece = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if piece is None:
        raise IOError(f"Could not read {path} as an image")

    #if the source PNG was BGR only (no alpha), tack on a fully-opaque alpha channel
    #so the rest of the rendering pipeline can assume BGRA without special-casing
    if piece.shape[2] == 3:
        alpha = np.full(piece.shape[:2], 255, dtype=np.uint8)
        piece = np.dstack([piece, alpha])

    return piece


def _resize_piece(piece: np.ndarray, size: int) -> np.ndarray:
    #resize the piece to size x size preserving alpha
    #use INTER_AREA when shrinking (better quality for downscaling)
    #and INTER_LINEAR when growing (good general-purpose upscaler)
    h, w = piece.shape[:2]
    interp = cv2.INTER_AREA if size < max(h, w) else cv2.INTER_LINEAR
    return cv2.resize(piece, (size, size), interpolation=interp)



class Renderer:
    #loads all piece images once at startup and renders board states on demand
    #
    #usage:
    #    renderer = Renderer()
    #    frame = renderer.render_from_board(game.board)   # 800x800x4 BGRA
    #    # ... then composite onto black + send to projector

    def __init__(self,
                 pieces_dir: Path = PIECES_DIR,
                 piece_size: Optional[int] = None):
        #pieces_dir: folder with king_white.png etc
        #piece_size: side length in pixels for each rendered piece
        #    defaults to 85% of square size so pieces have a small margin and don't touch the gridlines
        #    (looks nicer + gives a bit of breathing room)
        if piece_size is None:
            piece_size = int(SQUARE_SIZE * 0.85)
        self.piece_size = piece_size

        #preload all 6 piece images, resize them once at startup
        #means render_from_board() doesn't need to do any disk reads or resizes - super fast
        self._pieces: dict[int, np.ndarray] = {}
        for piece_type, filename in PIECE_FILENAMES.items():
            piece = _load_piece(pieces_dir / filename)
            self._pieces[piece_type] = _resize_piece(piece, piece_size)


    @staticmethod
    def square_to_pixel(square: chess.Square) -> tuple[int, int]:
        #given a chess square, returns the top-left pixel of where to draw it on the canvas
        #
        #chess.square_file(sq) -> 0-7 for files a-h (left to right)
        #chess.square_rank(sq) -> 0-7 for ranks 1-8 (bottom to top)
        #image y-axis points down, but chess ranks go up - so we flip with (7 - rank_idx)
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        x = file_idx * SQUARE_SIZE
        y = (7 - rank_idx) * SQUARE_SIZE
        return x, y


    def render_from_board(self, board: chess.Board) -> np.ndarray:
        #the main rendering call - takes the current board state and returns an 800x800 BGRA image
        #with only the white pieces drawn on a fully transparent background

        #start with a fully transparent canvas - all zeros, all 4 channels including alpha
        #alpha=0 means fully transparent, which is what we want for the empty squares
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)

        #loop through all 64 squares - if there's a white piece, blit its image onto the canvas
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != chess.WHITE:
                continue
            piece_image = self._pieces[piece.piece_type]
            self._blit_piece(canvas, piece_image, square)

        return canvas

    def _blit_piece(self,
                    canvas: np.ndarray,
                    piece_image: np.ndarray,
                    square: chess.Square) -> None:
        #pastes a piece onto the canvas, centred within its square, with proper alpha compositing

        sq_x, sq_y = self.square_to_pixel(square)

        #centre the piece in the square - offset compensates for piece_size being smaller than SQUARE_SIZE
        offset = (SQUARE_SIZE - self.piece_size) // 2
        x0 = sq_x + offset
        y0 = sq_y + offset
        x1 = x0 + self.piece_size
        y1 = y0 + self.piece_size

        #alpha blending - the piece's alpha channel determines how much it shows through where
        #it overlaps the canvas. without proper blending you get white rectangles around each piece
        #instead of nice cutout silhouettes
        piece_rgb = piece_image[:, :, :3]
        piece_alpha = piece_image[:, :, 3:4].astype(np.float32) / 255.0

        canvas_region = canvas[y0:y1, x0:x1]
        canvas_rgb = canvas_region[:, :, :3].astype(np.float32)
        canvas_alpha = canvas_region[:, :, 3:4].astype(np.float32) / 255.0

        #standard "over" compositing formula: result = src + dst * (1 - src_alpha)
        #this is the same maths used by image editors when you put a transparent layer on top of another
        out_rgb = piece_rgb * piece_alpha + canvas_rgb * (1 - piece_alpha)
        out_alpha = piece_alpha + canvas_alpha * (1 - piece_alpha)

        canvas_region[:, :, :3] = out_rgb.astype(np.uint8)
        canvas_region[:, :, 3:4] = (out_alpha * 255).astype(np.uint8)



def _main():
    #renders the starting position (Kasparov G6 right before 7...h6) and saves it as a preview PNG
    #useful for checking the rendering looks right without needing the projector set up
    #
    #saves two versions:
    #   1. transparent background (the actual output)
    #   2. composited onto dark grey (so i can see what the projected pieces will look like
    #      against a dark physical board - transparent PNGs are hard to preview otherwise)
    #
    #usage:
    #    python -m chess_code.renderer

    from chess_code.game_state import STARTING_FEN

    board = chess.Board(STARTING_FEN)
    renderer = Renderer()
    frame = renderer.render_from_board(board)

    #version 1: transparent background as-is
    out_transparent = Path("calibration/render_preview_transparent.png")
    out_transparent.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_transparent), frame)

    #version 2: composite onto dark grey to simulate the dark physical board behind the projection
    #without this it's really hard to visually check the rendering - most image viewers show
    #transparent PNGs against white, which makes white pieces invisible
    dark_bg = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 30, dtype=np.uint8)
    piece_alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
    composited = (frame[:, :, :3].astype(np.float32) * piece_alpha +
                  dark_bg.astype(np.float32) * (1 - piece_alpha))
    out_dark = Path("calibration/render_preview_on_dark.png")
    cv2.imwrite(str(out_dark), composited.astype(np.uint8))

    print(f"Wrote preview to:\n  {out_transparent}\n  {out_dark}")
    print(f"\nOpen the 'on_dark' version to see what the projected white")
    print(f"pieces will look like over a dark physical board.")


if __name__ == "__main__":
    _main()