# Gesture-Controlled Tic Tac Toe (OpenCV + MediaPipe)

Play Tic Tac Toe using only a webcam: pinch your thumb and index finger
together over a board cell to place your mark. Faces are detected each
frame, and the winner's face is cropped, saved, and shown on screen.

## Two-Player Setup
This is built for two people sharing one webcam:
- Whoever is on the **LEFT** side of the frame is always **Player X**
- Whoever is on the **RIGHT** side of the frame is always **Player O**

Each frame, up to 2 faces are detected and sorted left-to-right to assign
the X/O role; up to 2 hands are tracked the same way. A hand only counts
as a move if it belongs to the player whose turn it currently is — so O's
hand pinching during X's turn is ignored (and vice versa). A dashed
vertical line on screen shows the current X-side/O-side boundary. This is
purely position-based (no face recognition/identity), so keep to your
side of the frame during play.

## Features
- Webcam capture (OpenCV)
- MediaPipe Face Detection (2 faces -> left = X, right = O)
- MediaPipe Hand Tracking (2 hands -> matched to each player's side)
- Pinch gesture recognition (thumb tip <-> index tip distance)
- 3x3 board rendering with live hover-cell highlighting
- Turn management (X / O alternate automatically)
- Winner checking (rows, columns, diagonals) + draw detection
- Score keeping, persisted to `scores.json` between runs
- Face cropping + saving to `saved_faces/` on every win
- Winner face thumbnail overlay for a few seconds after a win
- Optional music/SFX playback via `pygame` (silent if files are missing)
- Move debounce/cooldown so a single pinch = a single move
- On-screen UI: scoreboard, current turn, status bar, controls

## Windows Setup

Open PowerShell in the project folder and run:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once as admin:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Since your built-in webcam is non-functional, point OpenCV at your
Iriun/phone camera index instead of 0 if needed — change:
```python
cap = cv2.VideoCapture(0)
```
to `cv2.VideoCapture(1)` (or whichever index Iriun registers as; check
with a quick loop over indices 0-3 if unsure).

## Run

```powershell
python tictactoe_cv.py
```

## Controls
- **Pinch** (thumb tip + index fingertip close together) while hovering
  over a board cell -> places the current player's mark
- **R** -> reset the board (keeps the scoreboard)
- **C** -> clear all scores
- **Q** or **ESC** -> quit

## Optional Sound
Drop MP3 files into the `sounds/` folder (all optional — game runs fine
without any of them):
- `click.mp3` — played on every valid move
- `win.mp3` — played when someone wins
- `draw.mp3` — played on a draw
- `bgm.mp3` — looping background music

## Folder Structure
```
tictactoe_cv/
├── tictactoe_cv.py     # main game
├── requirements.txt
├── scores.json          # auto-created, persists scoreboard
├── sounds/               # optional .mp3 files (see above)
└── saved_faces/          # auto-saved winner face crops (JPEG)
```

## Tuning
All key constants live at the top of `tictactoe_cv.py`:
- `PINCH_THRESHOLD_PX` — lower = stricter pinch, higher = more forgiving
- `MOVE_COOLDOWN` — seconds required between two accepted moves
- `WINNER_FACE_DISPLAY_TIME` — how long the winner's face thumbnail stays up
- `BOARD_SIZE`, `BOARD_X0/Y0` — board size/position on screen

## Notes
- Only one hand is tracked at a time (`max_num_hands=1`) to keep pinch
  detection unambiguous; X and O simply take turns pinching.
- If no face is detected at the moment someone wins, the face thumbnail
  is skipped for that round (win/score is still recorded normally).
