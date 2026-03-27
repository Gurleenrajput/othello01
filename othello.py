"""
Othello (Reversi) - A fully playable implementation with an AI opponent.

Modes:
  1. Player vs Player
  2. Player vs AI  (minimax + alpha-beta pruning, depth-4 by default)

Rules:
  - Black moves first.
  - A move is valid only if it outflanks at least one opponent piece.
  - Outflanked pieces are flipped to the current player's colour.
  - If a player has no valid move, their turn is skipped.
  - The game ends when neither player can move.
  - The player with more pieces on the board wins.
"""

import sys
import copy
import time

# ── Constants ────────────────────────────────────────────────────────────────

EMPTY = 0
BLACK = 1
WHITE = 2

BOARD_SIZE = 8

# All 8 directions (row_delta, col_delta)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0,  -1),           (0,  1),
              (1,  -1), (1,  0), (1,  1)]

# Positional weight table – corners are most valuable
WEIGHTS = [
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
]

# ── Board helpers ─────────────────────────────────────────────────────────────

def initial_board():
    """Return an 8×8 board with the four starting pieces."""
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    mid = BOARD_SIZE // 2
    board[mid - 1][mid - 1] = WHITE
    board[mid - 1][mid]     = BLACK
    board[mid][mid - 1]     = BLACK
    board[mid][mid]         = WHITE
    return board


def on_board(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def flipped_in_direction(board, player, r, c, dr, dc):
    """
    Return list of (row, col) positions that would be flipped when *player*
    places a piece at (r, c) in direction (dr, dc).
    Returns an empty list if the move is invalid in this direction.
    """
    opponent = WHITE if player == BLACK else BLACK
    flips = []
    nr, nc = r + dr, c + dc
    while on_board(nr, nc) and board[nr][nc] == opponent:
        flips.append((nr, nc))
        nr += dr
        nc += dc
    if flips and on_board(nr, nc) and board[nr][nc] == player:
        return flips
    return []


def get_flips(board, player, r, c):
    """Return all positions that would be flipped by placing *player* at (r, c)."""
    if board[r][c] != EMPTY:
        return []
    flips = []
    for dr, dc in DIRECTIONS:
        flips.extend(flipped_in_direction(board, player, r, c, dr, dc))
    return flips


def valid_moves(board, player):
    """Return list of (row, col) positions where *player* may legally place a piece."""
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if get_flips(board, player, r, c):
                moves.append((r, c))
    return moves


def make_move(board, player, r, c):
    """
    Apply *player*'s move at (r, c) to *board* (in-place).
    Returns the list of positions that were flipped.
    Raises ValueError if the move is illegal.
    """
    flips = get_flips(board, player, r, c)
    if not flips:
        raise ValueError(f"Illegal move: ({r}, {c})")
    board[r][c] = player
    for fr, fc in flips:
        board[fr][fc] = player
    return flips


def count_pieces(board):
    """Return (black_count, white_count)."""
    black = sum(board[r][c] == BLACK for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    white = sum(board[r][c] == WHITE for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    return black, white


# ── Display ───────────────────────────────────────────────────────────────────

PIECE = {EMPTY: "·", BLACK: "●", WHITE: "○"}
PLAYER_NAME = {BLACK: "Black (●)", WHITE: "White (○)"}
PLAYER_LETTER = {BLACK: "B", WHITE: "W"}


def display_board(board, highlights=None):
    """
    Print the board with optional *highlights* – a set/list of (r, c) tuples
    that mark valid moves for the current player.
    """
    highlights = set(highlights or [])
    col_labels = "  " + "  ".join(str(c + 1) for c in range(BOARD_SIZE))
    print(col_labels)
    print("  " + "─" * (BOARD_SIZE * 3 - 1))
    for r in range(BOARD_SIZE):
        row_label = chr(ord('A') + r)
        cells = []
        for c in range(BOARD_SIZE):
            if (r, c) in highlights:
                cells.append("·")   # valid-move marker (same as empty but hinted)
            else:
                cells.append(PIECE[board[r][c]])
        print(f"{row_label} │ " + "  ".join(cells))
    print()


def display_score(board):
    black, white = count_pieces(board)
    print(f"  Score ── Black (●): {black}  │  White (○): {white}\n")


# ── AI (minimax + alpha-beta pruning) ─────────────────────────────────────────

def heuristic(board, player):
    """
    Evaluate the board from *player*'s perspective using three signals:
      1. Positional weights
      2. Mobility (number of valid moves)
      3. Piece count (more important near end-game)
    """
    opponent = WHITE if player == BLACK else BLACK

    # 1. Positional score
    pos_score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                pos_score += WEIGHTS[r][c]
            elif board[r][c] == opponent:
                pos_score -= WEIGHTS[r][c]

    # 2. Mobility score
    my_moves = len(valid_moves(board, player))
    opp_moves = len(valid_moves(board, opponent))
    if my_moves + opp_moves > 0:
        mob_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)
    else:
        mob_score = 0

    # 3. Piece-count score (weighted by game phase)
    my_pieces, opp_pieces = count_pieces(board)
    total = my_pieces + opp_pieces
    if player == WHITE:
        my_pieces, opp_pieces = opp_pieces, my_pieces
    if my_pieces + opp_pieces > 0:
        piece_score = 100 * (my_pieces - opp_pieces) / (my_pieces + opp_pieces)
    else:
        piece_score = 0

    # Late-game: weight piece count more heavily
    phase = total / (BOARD_SIZE * BOARD_SIZE)
    return pos_score + mob_score * (1 - phase) + piece_score * phase * 2


def minimax(board, player, depth, alpha, beta, maximising):
    """
    Minimax with alpha-beta pruning.
    Returns (score, move) where *move* may be None if the game is over.
    """
    current_player = player if maximising else (WHITE if player == BLACK else BLACK)
    moves = valid_moves(board, current_player)

    if depth == 0 or not moves:
        # Check if the other player can move (pass scenario)
        other = WHITE if current_player == BLACK else BLACK
        if not moves and not valid_moves(board, other):
            # Game over – evaluate final score
            my_pieces, opp_pieces = count_pieces(board)
            if player == WHITE:
                my_pieces, opp_pieces = opp_pieces, my_pieces
            if my_pieces > opp_pieces:
                return 10000, None
            elif my_pieces < opp_pieces:
                return -10000, None
            else:
                return 0, None
        return heuristic(board, player), None

    best_move = None
    if maximising:
        best_score = float('-inf')
        for r, c in moves:
            child = copy.deepcopy(board)
            make_move(child, current_player, r, c)
            score, _ = minimax(child, player, depth - 1, alpha, beta, False)
            if score > best_score:
                best_score, best_move = score, (r, c)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move
    else:
        best_score = float('inf')
        for r, c in moves:
            child = copy.deepcopy(board)
            make_move(child, current_player, r, c)
            score, _ = minimax(child, player, depth - 1, alpha, beta, True)
            if score < best_score:
                best_score, best_move = score, (r, c)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_move


def ai_move(board, player, depth=5):
    """Choose the best move for *player* using minimax search."""
    _, move = minimax(board, player, depth, float('-inf'), float('inf'), True)
    return move


# ── Input parsing ─────────────────────────────────────────────────────────────

def parse_move(text):
    """
    Parse a move string such as 'A1', 'b3', or '3 b'.
    Returns (row, col) as 0-based indices, or raises ValueError.
    """
    text = text.strip().upper().replace(" ", "")
    if len(text) == 2:
        if text[0].isalpha() and text[1].isdigit():
            r = ord(text[0]) - ord('A')
            c = int(text[1]) - 1
        elif text[0].isdigit() and text[1].isalpha():
            c = int(text[0]) - 1
            r = ord(text[1]) - ord('A')
        else:
            raise ValueError(f"Cannot parse move: '{text}'")
    else:
        raise ValueError(f"Cannot parse move: '{text}'")
    if not on_board(r, c):
        raise ValueError(f"Move out of board range: ({r}, {c})")
    return r, c


# ── Game loop ─────────────────────────────────────────────────────────────────

def choose_mode():
    print("╔══════════════════════════════╗")
    print("║        OTHELLO / REVERSI     ║")
    print("╚══════════════════════════════╝")
    print()
    print("Select game mode:")
    print("  1 – Player vs Player")
    print("  2 – Player vs AI  (you play Black)")
    print("  3 – AI vs AI      (watch two AIs play)")
    print()
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        print("Please enter 1, 2, or 3.")


def choose_ai_depth():
    print()
    print("Choose AI difficulty:")
    print("  1 – Easy   (depth 2)")
    print("  2 – Medium (depth 4)")
    print("  3 – Hard   (depth 6)")
    while True:
        choice = input("Enter 1, 2, or 3 [default: 2]: ").strip() or "2"
        if choice == "1":
            return 2
        if choice == "2":
            return 4
        if choice == "3":
            return 6
        print("Please enter 1, 2, or 3.")


def human_turn(board, player):
    moves = valid_moves(board, player)
    print(f"\n{PLAYER_NAME[player]}'s turn")
    display_board(board, highlights=moves)
    display_score(board)
    print(f"Valid moves: {', '.join(chr(ord('A') + r) + str(c + 1) for r, c in moves)}")
    while True:
        raw = input("Enter your move (e.g. D3): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            print("Thanks for playing!")
            sys.exit(0)
        try:
            r, c = parse_move(raw)
        except ValueError as e:
            print(f"  ✗ {e}. Try again.")
            continue
        if (r, c) not in moves:
            print("  ✗ That is not a valid move. Try again.")
            continue
        make_move(board, player, r, c)
        print(f"  ✓ Placed at {chr(ord('A') + r)}{c + 1}")
        return


def computer_turn(board, player, depth):
    moves = valid_moves(board, player)
    print(f"\n{PLAYER_NAME[player]}'s turn  [AI thinking…]")
    t0 = time.time()
    move = ai_move(board, player, depth)
    elapsed = time.time() - t0
    if move is None:
        return  # no valid moves – will be handled by caller
    r, c = move
    make_move(board, player, r, c)
    display_board(board)
    display_score(board)
    print(f"  AI played {chr(ord('A') + r)}{c + 1}  ({elapsed:.2f}s)")


def play():
    mode = choose_mode()
    depth = choose_ai_depth() if mode in (2, 3) else 4

    board = initial_board()
    current = BLACK  # Black always goes first
    skips = 0

    while True:
        moves = valid_moves(board, current)

        if not moves:
            skips += 1
            if skips >= 2:
                break  # Both players passed – game over
            opponent_name = PLAYER_NAME[WHITE if current == BLACK else BLACK]
            print(f"\n{PLAYER_NAME[current]} has no valid moves – turn skipped.")
            current = WHITE if current == BLACK else BLACK
            continue

        skips = 0

        is_human = (
            mode == 1
            or (mode == 2 and current == BLACK)
        )

        if is_human:
            human_turn(board, current)
        else:
            computer_turn(board, current, depth)

        current = WHITE if current == BLACK else BLACK

    # ── Game over ──────────────────────────────────────────────────────────
    print("\n" + "═" * 36)
    print("           GAME OVER")
    print("═" * 36)
    display_board(board)
    black, white = count_pieces(board)
    print(f"  Final score ── Black (●): {black}  │  White (○): {white}")
    if black > white:
        print("  🏆  Black wins!")
    elif white > black:
        print("  🏆  White wins!")
    else:
        print("  🤝  It's a draw!")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    play()
