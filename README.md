# The Rematch

A live computer vision performance system that restages Game 6 of the
1997 Deep Blue vs. Kasparov match. The player faces off against
Stockfish AI on a real physical chess board. Moves are detected in
real time via webcam, and the AI's responses are projected back onto
the pieces.

Conceptually, the piece sits in the tradition of the 18th-century
Mechanical Turk and the 1997 Deep Blue match itself: staged encounters
between human intelligence and its mechanical or computational
counterpart. Technically, it's a real-time computer vision and
projection system built from scratch, with the physical board as the
interface between the human and the AI.

## Stack

- **Language:** Python
- **Computer vision:** OpenCV
- **Chess engine:** Stockfish (2750 Elo)
- **Hardware:** PS3 Eye webcam, projector, physical chess board

## Architecture

Seven modules working together in real time:

- **`board_calibration.py`:** corner detection, perspective correction, homography for the physical board
- **`projector_calibration.py`:** calibrating the projector so overlays land accurately on the board
- **`motion_detector.py`:** detecting when a move has occurred on the board
- **`occupancy_detector.py`:** determining which squares are occupied and by which piece, using a custom colour-based approach built after off-the-shelf ML classifiers proved unreliable under stage lighting
- **`game_state.py`:** maintaining the full game history and translating board state changes into chess notation
- **`renderer.py`:** rendering visual overlays to be projected back onto the board
- **`main.py`:** orchestrating the pipeline in real time and integrating with Stockfish
