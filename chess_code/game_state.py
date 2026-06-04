#This is the source of truth for the chess game itself - wraps a python-chess Board
#and a UCI engine (Stockfish) behind one clean interface so the CV layer and the renderer
#can both use it without knowing the chess internals.

#The starting position is set to the position right before Kasparov's 7...h6 in
#Deep Blue vs Kasparov Game 6 (11 May 1997) - the move widely considered to have lost him
#the game. The first move on stage is my refusal to play it - that's the whole conceptual
#move of the piece, so this FEN is the most important constant in the codebase.

#Test from terminal:
#    python -m code.game_state /path/to/stockfish


from __future__ import annotations

import chess
import chess.engine
from dataclasses import dataclass
from typing import Optional



#FEN for the position right before Kasparov's 7...h6 in Game 6, 1997. Black to move.
#The moves played up to here:
#   1. e4   c6       (Caro-Kann Defence)
#   2. d4   d5
#   3. Nc3  dxe4
#   4. Nxe4 Nd7
#   5. Ng5  Ngf6
#   6. Bd3  e6
#   7. N1f3  <-- Black to move. Kasparov played h6. I do not.

#FEN format breakdown: pieces / side-to-move / castling-rights / en-passant / halfmoves / fullmoves
STARTING_FEN = "r1bqkb1r/pp1n1ppp/2p1pn2/6N1/3P4/3B1N2/PPP2PPP/R1BQK2R b KQkq - 1 7"



#dataclass for what happened when a move was applied - the renderer uses this to know
#whether to animate a capture, castle etc
@dataclass
class MoveResult:
    move: chess.Move
    is_capture: bool
    captured_square: Optional[chess.Square]
    is_castling: bool
    is_en_passant: bool
    is_check: bool
    san: str  #standard algebraic notation like "Nxe6" - useful for debug logs


#high-level summary of where the game is - main.py checks this every loop to know if the game ended
@dataclass
class GameStatus:
    is_over: bool
    result_text: str
    in_check: bool
    move_number: int
    turn: chess.Color



class GameState:
    #wraps python-chess Board + a UCI engine
    #usage with a context manager so the engine process gets killed cleanly when done:
    #    with GameState("/path/to/stockfish") as game:
    #        while not game.status().is_over:
    #            move = ...                          (from CV, or typed in)
    #            game.apply_black_move(move)
    #            if game.status().is_over:
    #                break
    #            game.request_white_move()

    def __init__(self,
                 engine_path: str,
                 starting_fen: str = STARTING_FEN,
                 engine_elo: Optional[int] = None,
                 engine_think_seconds: float = 2.0):
        #engine_path: path to the Stockfish binary (on my Mac it's /opt/homebrew/bin/stockfish)
        #starting_fen: the position to start from - defaults to G6 right before 7...h6
        #engine_elo: None = full strength. For the performance i set this to 2750 (Deep Blue's
        #            estimated 1997 strength). For rehearsal i drop it to 2000-2200 so i'm not
        #            instantly defeated and can actually study the position
        #engine_think_seconds: how long the engine thinks per move - 2s feels about right,
        #                     long enough for tension during the silence but not so long it drags

        self.board = chess.Board(starting_fen)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine_think_seconds = engine_think_seconds

        #Stockfish has a UCI_LimitStrength option that clamps it to a given Elo
        #only configure it if I actually passed an elo, otherwise it runs at full strength
        if engine_elo is not None:
            self.engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": int(engine_elo),
            })

        #keep history of all MoveResults for the debug overlay + later analysis
        self.history: list[MoveResult] = []


    def black_squares(self) -> set[chess.Square]:
        #squares currently occupied by a black piece
        #the CV layer compares its camera reading to this set to detect when I've moved
        return {sq for sq in chess.SQUARES
                if (p := self.board.piece_at(sq)) and p.color == chess.BLACK}

    def white_squares(self) -> set[chess.Square]:
        #squares currently occupied by a white piece
        #the renderer uses this to know where to draw the projected white pieces
        return {sq for sq in chess.SQUARES
                if (p := self.board.piece_at(sq)) and p.color == chess.WHITE}

    def piece_at(self, square: chess.Square) -> Optional[chess.Piece]:
        return self.board.piece_at(square)

    def fen(self) -> str:
        return self.board.fen()

    def status(self) -> GameStatus:
        #checks all the end-game conditions in order and returns a clean status object
        #i had to check each draw type separately because python-chess doesn't bundle them
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return GameStatus(True, f"Checkmate. {winner} wins.", True,
                              self.board.fullmove_number, self.board.turn)
        if self.board.is_stalemate():
            return GameStatus(True, "Stalemate. Draw.", False,
                              self.board.fullmove_number, self.board.turn)
        if self.board.is_insufficient_material():
            return GameStatus(True, "Insufficient material. Draw.", False,
                              self.board.fullmove_number, self.board.turn)
        if self.board.can_claim_threefold_repetition():
            return GameStatus(True, "Threefold repetition. Draw.", False,
                              self.board.fullmove_number, self.board.turn)
        if self.board.can_claim_fifty_moves():
            return GameStatus(True, "Fifty-move rule. Draw.", False,
                              self.board.fullmove_number, self.board.turn)

        #game is still ongoing - return normal in-progress status
        return GameStatus(
            is_over=False,
            result_text="Game in progress.",
            in_check=self.board.is_check(),
            move_number=self.board.fullmove_number,
            turn=self.board.turn,
        )


    def infer_black_move(self,
                         observed_black_squares: set[chess.Square]
                        ) -> Optional[chess.Move]:
        #this is the bridge between the CV layer and the chess engine
        #the camera tells me which squares it thinks have black pieces on them, and i need to
        #figure out which legal black move would produce that exact set

        #returns the move if exactly one legal move matches, otherwise None
        #if zero matches it's usually a misread or a bumped piece - main.py just waits and tries again
        #if multiple match (rare) it's also None to avoid guessing wrong

        if self.board.turn != chess.BLACK:
            return None

        #brute-force approach: try every legal move, push it, check if the resulting black-squares set
        #matches what the camera saw, then pop it back. this is fast enough because there's rarely
        #more than ~40 legal moves to try
        matches: list[chess.Move] = []
        for move in self.board.legal_moves:
            self.board.push(move)
            resulting = {sq for sq in chess.SQUARES
                         if (p := self.board.piece_at(sq))
                         and p.color == chess.BLACK}
            self.board.pop()
            if resulting == observed_black_squares:
                matches.append(move)

        if len(matches) == 1:
            return matches[0]
        return None


    def apply_black_move(self, move: chess.Move) -> MoveResult:
        #called once the CV layer has confirmed which move I played
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")
        return self._apply(move)

    def request_white_move(self) -> MoveResult:
        #engine plays for white. blocks for engine_think_seconds - main.py handles the
        #visual feedback during this time so the audience knows the machine is "thinking"
        if self.board.turn != chess.WHITE:
            raise RuntimeError("Not White's turn.")
        limit = chess.engine.Limit(time=self.engine_think_seconds)
        result = self.engine.play(self.board, limit)
        return self._apply(result.move)

    def _apply(self, move: chess.Move) -> MoveResult:
        #captures all the metadata about the move BEFORE pushing it onto the board
        #(once you push, board.is_capture etc are about the *next* position, not this one)
        san = self.board.san(move)
        is_capture = self.board.is_capture(move)
        is_en_passant = self.board.is_en_passant(move)
        is_castling = self.board.is_castling(move)

        #figure out which square the captured piece was on
        #for normal captures it's just the destination square, but for en passant it's
        #the square one rank behind (because the captured pawn moved past)
        captured_square: Optional[chess.Square] = None
        if is_capture:
            if is_en_passant:
                direction = -8 if self.board.turn == chess.WHITE else 8
                captured_square = move.to_square + direction
            else:
                captured_square = move.to_square

        self.board.push(move)

        result = MoveResult(
            move=move,
            is_capture=is_capture,
            captured_square=captured_square,
            is_castling=is_castling,
            is_en_passant=is_en_passant,
            is_check=self.board.is_check(),
            san=san,
        )
        self.history.append(result)
        return result


    def close(self):
        #kill the engine subprocess. wrap in try/except because if it's already dead
        #we don't want to crash on shutdown
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()



def _main():
    #terminal test mode - lets me play black against the engine by typing UCI moves
    #this is what I use to test the engine setup before plugging in the CV layer
    #type 'q' to quit, 'fen' to print current position

    #Usage:
    #    python -m code.game_state /path/to/stockfish [elo]

    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m code.game_state /path/to/stockfish [elo]")
        sys.exit(1)

    engine_path = sys.argv[1]
    elo = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    with GameState(engine_path, engine_elo=elo) as game:
        print("Starting position (Kasparov G6, right before 7...h6):")
        #invert_color=True flips the unicode display so it reads correctly in a dark terminal
        print(game.board.unicode(invert_color=True))
        print(f"FEN: {game.fen()}")
        print("Black to move. Enter moves in UCI format (e.g. e7e5).")
        print("Type 'q' to quit, 'fen' to print position.")

        while not game.status().is_over:
            user = input("\nBlack > ").strip()
            if user == "q":
                break
            if user == "fen":
                print(game.fen())
                continue

            try:
                move = chess.Move.from_uci(user)
            except Exception:
                print("Not a valid UCI string. Try again, e.g. 'e7e5'.")
                continue

            try:
                black_result = game.apply_black_move(move)
            except ValueError as e:
                print(f"  {e}")
                continue

            print(f"  You played: {black_result.san}")
            if game.status().is_over:
                print(game.status().result_text)
                break

            print("  Engine thinking...")
            white_result = game.request_white_move()
            print(f"  Engine played: {white_result.san}")
            print(game.board.unicode(invert_color=True))

            status = game.status()
            if status.is_over:
                print(status.result_text)
                break
            if status.in_check:
                print("  (you are in check)")


if __name__ == "__main__":
    _main()