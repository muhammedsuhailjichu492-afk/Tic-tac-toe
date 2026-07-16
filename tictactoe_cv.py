
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import math
import random

# --------------------------------------------------------------------------
# Optional audio (pygame). The game runs fine with no sound files present;
# it just stays silent. Drop your own .mp3 files into the sounds/ folder:
#   sounds/click.mp3   -> played on every valid move
#   sounds/win.mp3      -> played when someone wins
#   sounds/draw.mp3     -> played on a draw
#   sounds/bgm.mp3       -> looping background music
# --------------------------------------------------------------------------
try:
    import pygame

    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


# ==========================================================================
# CONFIG
# ==========================================================================
FRAME_W, FRAME_H = 1280, 720

BOARD_SIZE = 480                       # board square in pixels
BOARD_MARGIN_TOP = 120
BOARD_X0 = (FRAME_W - BOARD_SIZE) // 2
BOARD_Y0 = BOARD_MARGIN_TOP
CELL = BOARD_SIZE // 3

PINCH_THRESHOLD_PX = 40                # thumb-index distance to count as a pinch
MOVE_COOLDOWN = 1.0                    # seconds between accepted moves
WINNER_FACE_DISPLAY_TIME = 6.0         # seconds to show winner face overlay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
FACES_DIR = os.path.join(BASE_DIR, "saved_faces")
SCORES_FILE = os.path.join(BASE_DIR, "scores.json")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(SOUNDS_DIR, exist_ok=True)

COLOR_BG_PANEL = (30, 30, 30)
COLOR_GRID = (230, 230, 230)
COLOR_X = (60, 60, 240)      # red-ish (BGR)
COLOR_O = (240, 170, 60)     # blue-ish (BGR)
COLOR_HOVER = (0, 255, 255)
COLOR_WIN_LINE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)

WIN_COMBOS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


# ==========================================================================
# AUDIO HELPERS
# ==========================================================================
def _load_sound(name):
    if not AUDIO_AVAILABLE:
        return None
    path = os.path.join(SOUNDS_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None


class AudioManager:
    def __init__(self):
        self.click_sfx = _load_sound("click.mp3")
        self.win_sfx = _load_sound("win.mp3")
        self.draw_sfx = _load_sound("draw.mp3")
        self.bgm_path = os.path.join(SOUNDS_DIR, "bgm.mp3")
        self._bgm_started = False

    def start_bgm(self):
        if not AUDIO_AVAILABLE or self._bgm_started:
            return
        if os.path.exists(self.bgm_path):
            try:
                pygame.mixer.music.load(self.bgm_path)
                pygame.mixer.music.set_volume(0.35)
                pygame.mixer.music.play(loops=-1)
            except Exception:
                pass
        self._bgm_started = True

    def play_click(self):
        if self.click_sfx:
            self.click_sfx.play()

    def play_win(self):
        if self.win_sfx:
            self.win_sfx.play()

    def play_draw(self):
        if self.draw_sfx:
            self.draw_sfx.play()


# ==========================================================================
# SCORE PERSISTENCE
# ==========================================================================
def load_scores():
    default = {"X": 0, "O": 0, "Draw": 0}
    if os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE, "r") as f:
                data = json.load(f)
            default.update(data)
        except Exception:
            pass
    return default


def save_scores(scores):
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(scores, f)
    except Exception:
        pass


# ==========================================================================
# PLAYER <-> FACE / HAND ASSIGNMENT
# ==========================================================================
# Two people play at once: whoever stands/sits on the LEFT of the frame is
# always "X", whoever is on the RIGHT is always "O". This is re-computed
# every frame from face detections, so it's simple and needs no face
# recognition/identity model — just left/right position.

def assign_faces_to_players(face_results, w, h):
    """
    Return {'X': bbox_px or None, 'O': bbox_px or None} where bbox_px is
    (x1, y1, x2, y2) in pixel coordinates. Leftmost detected face -> X,
    rightmost detected face -> O. If only one face is detected it is
    assigned to whichever side of the frame-midpoint it falls on.
    """
    assignment = {"X": None, "O": None}
    if not face_results.detections:
        return assignment

    boxes = []
    for det in face_results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = min(x1 + bw, w)
        y2 = min(y1 + bh, h)
        cx = (x1 + x2) / 2
        boxes.append((cx, (x1, y1, x2, y2)))

    boxes.sort(key=lambda b: b[0])  # sort left -> right by center x

    if len(boxes) == 1:
        cx = boxes[0][0]
        role = "X" if cx < (w / 2) else "O"
        assignment[role] = boxes[0][1]
    else:
        # leftmost -> X, rightmost -> O (ignore any extra faces beyond 2)
        assignment["X"] = boxes[0][1]
        assignment["O"] = boxes[-1][1]

    return assignment


def player_side_midpoint(player_boxes, w):
    """X-pixel boundary used to decide which player a given hand belongs to."""
    xb = player_boxes.get("X")
    ob = player_boxes.get("O")
    if xb is not None and ob is not None:
        x_center = (xb[0] + xb[2]) / 2
        o_center = (ob[0] + ob[2]) / 2
        return (x_center + o_center) / 2
    return w / 2  # fall back to the frame's horizontal midpoint


def assign_hand_to_player(wrist_x, midpoint_x):
    return "X" if wrist_x < midpoint_x else "O"


# ==========================================================================
# GAME LOGIC
# ==========================================================================
def check_winner(board):
    """Return (winner_symbol, winning_combo) or (None, None)."""
    for combo in WIN_COMBOS:
        a, b, c = combo
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a], combo
    return None, None


def is_draw(board):
    return " " not in board and check_winner(board)[0] is None


def cell_from_pixel(x, y):
    """Return (row, col) for a pixel inside the board, else None."""
    if BOARD_X0 <= x <= BOARD_X0 + BOARD_SIZE and BOARD_Y0 <= y <= BOARD_Y0 + BOARD_SIZE:
        col = (x - BOARD_X0) // CELL
        row = (y - BOARD_Y0) // CELL
        col = min(max(int(col), 0), 2)
        row = min(max(int(row), 0), 2)
        return row, col
    return None


def cell_index(row, col):
    return row * 3 + col


def cell_center_px(row, col):
    cx = BOARD_X0 + col * CELL + CELL // 2
    cy = BOARD_Y0 + row * CELL + CELL // 2
    return cx, cy


# ==========================================================================
# DRAWING HELPERS
# ==========================================================================
def draw_board_grid(frame):
    # panel behind the board
    cv2.rectangle(
        frame,
        (BOARD_X0 - 20, BOARD_Y0 - 20),
        (BOARD_X0 + BOARD_SIZE + 20, BOARD_Y0 + BOARD_SIZE + 20),
        COLOR_BG_PANEL,
        -1,
    )
    for i in range(1, 3):
        x = BOARD_X0 + i * CELL
        cv2.line(frame, (x, BOARD_Y0), (x, BOARD_Y0 + BOARD_SIZE), COLOR_GRID, 3)
        y = BOARD_Y0 + i * CELL
        cv2.line(frame, (BOARD_X0, y), (BOARD_X0 + BOARD_SIZE, y), COLOR_GRID, 3)
    cv2.rectangle(
        frame,
        (BOARD_X0, BOARD_Y0),
        (BOARD_X0 + BOARD_SIZE, BOARD_Y0 + BOARD_SIZE),
        COLOR_GRID,
        3,
    )


def draw_hover_highlight(frame, row, col):
    x0 = BOARD_X0 + col * CELL
    y0 = BOARD_Y0 + row * CELL
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + CELL, y0 + CELL), COLOR_HOVER, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + CELL, y0 + CELL), COLOR_HOVER, 3)


def draw_symbol(frame, row, col, player):
    cx, cy = cell_center_px(row, col)
    pad = int(CELL * 0.28)
    if player == "X":
        cv2.line(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), COLOR_X, 8)
        cv2.line(frame, (cx - pad, cy + pad), (cx + pad, cy - pad), COLOR_X, 8)
    else:
        cv2.circle(frame, (cx, cy), pad, COLOR_O, 8)


def draw_win_line(frame, combo):
    a, c = combo[0], combo[2]
    r1, c1 = a // 3, a % 3
    r2, c2 = c // 3, c % 3
    p1 = cell_center_px(r1, c1)
    p2 = cell_center_px(r2, c2)
    cv2.line(frame, p1, p2, COLOR_WIN_LINE, 10, cv2.LINE_AA)


def put_text_centered(frame, text, y, scale=1.2, color=COLOR_TEXT, thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (FRAME_W - w) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_scoreboard(frame, scores, current_player, status_msg):
    cv2.rectangle(frame, (0, 0), (FRAME_W, 90), (20, 20, 20), -1)
    cv2.putText(frame, f"X Wins: {scores['X']}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, COLOR_X, 2, cv2.LINE_AA)
    cv2.putText(frame, f"O Wins: {scores['O']}", (330, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, COLOR_O, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Draws: {scores['Draw']}", (630, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, COLOR_TEXT, 2, cv2.LINE_AA)

    turn_color = COLOR_X if current_player == "X" else COLOR_O
    cv2.putText(frame, f"Turn: {current_player}", (930, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, turn_color, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, FRAME_H - 50), (FRAME_W, FRAME_H), (20, 20, 20), -1)
    cv2.putText(frame, status_msg, (20, FRAME_H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "R: Restart   C: Clear Scores   Q/ESC: Quit",
                (FRAME_W - 560, FRAME_H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1, cv2.LINE_AA)


def draw_winner_face_overlay(frame, face_img, winner_symbol):
    if face_img is None:
        return
    thumb_size = 160
    thumb = cv2.resize(face_img, (thumb_size, thumb_size))
    x0, y0 = FRAME_W - thumb_size - 30, 110
    # border color matches winner
    color = COLOR_X if winner_symbol == "X" else COLOR_O
    cv2.rectangle(frame, (x0 - 6, y0 - 6), (x0 + thumb_size + 6, y0 + thumb_size + 6), color, 4)
    frame[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb
    cv2.putText(frame, "WINNER", (x0, y0 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def crop_face_from_bbox(frame, bbox_px, w, h):
    """Crop a face out of frame given a (x1, y1, x2, y2) pixel bbox, with padding."""
    if bbox_px is None:
        return None
    x1, y1, x2, y2 = bbox_px
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * 0.25)
    pad_y = int(bh * 0.25)
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w)
    y2 = min(y2 + pad_y, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_face(face_img, winner_symbol):
    if face_img is None or face_img.size == 0:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(FACES_DIR, f"winner_{winner_symbol}_{ts}.jpg")
    try:
        cv2.imwrite(filename, face_img)
    except Exception:
        pass


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    audio = AudioManager()
    scores = load_scores()

    board = [" "] * 9
    current_player = "X"
    winner_symbol = None
    winning_combo = None
    game_over = False
    status_msg = "Player X: stand/sit on the LEFT. Player O: on the RIGHT."

    last_move_time = 0.0
    was_pinching = False

    winner_face_img = None
    winner_face_time = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera index / drivers.")
        return

    audio.start_bgm()

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands, mp_face.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    ) as face_detection:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            face_results = face_detection.process(rgb)
            rgb.flags.writeable = True

            draw_board_grid(frame)

            # ---- face detection -> assign left face = X, right face = O ----
            player_faces = assign_faces_to_players(face_results, w, h)
            midpoint_x = player_side_midpoint(player_faces, w)

            for role, bbox_px in player_faces.items():
                if bbox_px is None:
                    continue
                x1, y1, x2, y2 = bbox_px
                color = COLOR_X if role == "X" else COLOR_O
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Player {role}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # dashed vertical divider showing the X-side / O-side split
            for y in range(0, h, 20):
                cv2.line(frame, (int(midpoint_x), y), (int(midpoint_x), y + 10),
                         (120, 120, 120), 1)

            # ---- hand tracking / pinch / hover (up to 2 hands, 1 per player) ----
            hovered_cell = None
            active_is_pinching = False   # pinch state of the CURRENT player's hand only
            hand_by_player = {"X": None, "O": None}

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    lm = hand_landmarks.landmark
                    wrist_x = lm[0].x * w
                    index_tip = (int(lm[8].x * w), int(lm[8].y * h))
                    thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                    pinch_dist = distance(index_tip, thumb_tip)
                    hand_pinching = pinch_dist < PINCH_THRESHOLD_PX
                    role = assign_hand_to_player(wrist_x, midpoint_x)

                    cv2.circle(frame, index_tip, 10, (0, 255, 255), -1)
                    cv2.circle(frame, thumb_tip, 10, (255, 0, 255), -1)
                    cv2.line(frame, index_tip, thumb_tip,
                             (0, 255, 0) if hand_pinching else (100, 100, 100), 2)

                    # if two hands land on the same side, keep the first one seen
                    if hand_by_player[role] is None:
                        hand_by_player[role] = {
                            "index_tip": index_tip,
                            "pinching": hand_pinching,
                        }

            current_hand = hand_by_player[current_player]
            if current_hand is not None:
                hovered_cell = cell_from_pixel(*current_hand["index_tip"])
                active_is_pinching = current_hand["pinching"]

            if hovered_cell and not game_over:
                draw_hover_highlight(frame, *hovered_cell)

            # ---- move registration (debounced, current player's hand only) ----
            now = time.time()
            pinch_just_started = active_is_pinching and not was_pinching
            cooldown_elapsed = (now - last_move_time) >= MOVE_COOLDOWN

            if (
                not game_over
                and hovered_cell
                and pinch_just_started
                and cooldown_elapsed
            ):
                row, col = hovered_cell
                idx = cell_index(row, col)
                if board[idx] == " ":
                    board[idx] = current_player
                    last_move_time = now
                    audio.play_click()
                    status_msg = f"Player {current_player} placed at ({row + 1},{col + 1})"

                    winner_symbol, winning_combo = check_winner(board)
                    if winner_symbol:
                        game_over = True
                        scores[winner_symbol] += 1
                        save_scores(scores)
                        audio.play_win()
                        status_msg = f"Player {winner_symbol} WINS! Press R to play again"
                        face_img = crop_face_from_bbox(frame, player_faces.get(winner_symbol), w, h)
                        if face_img is not None:
                            save_face(face_img, winner_symbol)
                            winner_face_img = face_img
                            winner_face_time = now
                    elif is_draw(board):
                        game_over = True
                        scores["Draw"] += 1
                        save_scores(scores)
                        audio.play_draw()
                        status_msg = "It's a DRAW! Press R to play again"
                    else:
                        current_player = "O" if current_player == "X" else "X"
                else:
                    status_msg = "That cell is already taken!"

            was_pinching = active_is_pinching

            # ---- draw board marks ----
            for i, mark in enumerate(board):
                if mark != " ":
                    draw_symbol(frame, i // 3, i % 3, mark)

            if winning_combo:
                draw_win_line(frame, winning_combo)

            # ---- winner face overlay (timed) ----
            if winner_face_img is not None:
                if time.time() - winner_face_time < WINNER_FACE_DISPLAY_TIME:
                    draw_winner_face_overlay(frame, winner_face_img, winner_symbol)
                else:
                    winner_face_img = None

            # ---- UI text ----
            draw_scoreboard(frame, scores, current_player, status_msg)
            if game_over:
                put_text_centered(frame, "GAME OVER", FRAME_H - 70, 1.1, (0, 0, 255), 3)

            cv2.imshow("Gesture Tic Tac Toe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("r"):
                board = [" "] * 9
                current_player = "X"
                winner_symbol = None
                winning_combo = None
                game_over = False
                status_msg = "New round! X on the LEFT, O on the RIGHT."
                winner_face_img = None
            elif key == ord("c"):
                scores = {"X": 0, "O": 0, "Draw": 0}
                save_scores(scores)
                status_msg = "Scores cleared"

    cap.release()
    cv2.destroyAllWindows()
    if AUDIO_AVAILABLE:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
