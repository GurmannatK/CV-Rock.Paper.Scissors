# ✊✋✌ CV Rock Paper Scissors

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.x-00C800?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**Real-time Rock Paper Scissors using your actual hand — no controllers, just your webcam.**

[Features](#features) · [Installation](#installation) · [How It Works](#how-it-works) · [Architecture](docs/architecture.md)

</div>

---

![Main Menu](screenshots/menu.png)

---

## What is this?

A fully playable Rock Paper Scissors game where MediaPipe tracks your hand landmarks in real time, classifies your gesture each frame, and determines the winner against an adaptive AI or a second local player. Runs at 60 FPS with animated gesture reveals, a particle system, tournament mode, and synthesised audio — all in a single Python file.

---

## Features

- **VS AI** and **Local 2-Player** modes — both players use the same webcam, hands assigned by screen position
- **Tournament system** — Single Round, Best of 3, Best of 5, Best of 7, or Championship (Best of 11)
- **Adaptive AI** — tracks your move history and adjusts strategy the more you win; difficulty shown live in the scoreboard
- **14-frame vote buffer** with majority-vote smoothing so a single bad frame never misfires
- **EMA-smoothed confidence HUD** — separate "hand detected" and "gesture locked" bars
- **Per-gesture idle animations** — Rock slams with shockwaves, Paper floats in a figure-8, Scissors snaps with slash trails
- **Reveal animations** — Rock bounces in from above, Paper springs with elastic easing, Scissors slashes in from the side
- Particle burst system, screen flash on result, winner glow aura, scoreboard scan-beam
- All audio synthesised with NumPy at startup — no audio files needed

---

## Screenshots

| Gameplay | Result |
|---|---|
| ![Gameplay](screenshots/gameplay.png) | ![Result](screenshots/results.png) |

*Left: countdown with live webcam feed, scanner HUD, and AI panel. Right: winner card with glow aura, hand skeleton visible in the background.*

---

## Installation

```bash
git clone https://github.com/GurmannatK/CV-Rock.Paper.Scissors.git
cd CV-Rock.Paper.Scissors
pip install -r requirements.txt
python rock_paper_scissorsVF.py
```

On first launch, the MediaPipe hand landmarker model (~8 MB) downloads automatically and caches as `hand_landmarker.task` next to the script.

**Python 3.9+ required.**

---

## Requirements

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture and frame processing |
| `mediapipe` | 21-point hand landmark detection |
| `numpy` | Frame arrays and audio synthesis |
| `pygame` | Rendering, input, and audio playback |

---

## Controls

| Key | Action |
|---|---|
| `1` | Quick Play / VS AI |
| `2` | Tournament Mode |
| `SPACE` | Next round |
| `R` | Return to main menu |
| `ESC` | Back / Quit |

No key needed to make your move — hold your gesture in frame and wait for **LOCK ACQUIRED**.

---

## How It Works

### Gesture Detection
Each frame goes through: **capture → landmark detection → extension check → classification → vote buffer → EMA smoothing.**

Finger extension is determined geometrically — tip Y vs PIP joint Y for fingers 2–5, and X displacement for the thumb. A gesture is only confirmed when one label holds **8+ votes in a 14-frame window**, making the detection stable even with hand wobble.

Rock is disambiguated by checking if any extended finger's tip is within `2× palm radius` of the wrist — partially curled fingers don't misfire as Scissors.

### Adaptive AI
Starts at 70% random. Tracks the player's last 8 moves and counters the most frequent one. Every player win drops randomness by 5% (floor: 10%), so a strong player faces a near-fully adaptive opponent.

### No ML classifier for gestures
Classification is purely geometric rule-based logic, which makes it fast, interpretable, and hardware-agnostic.

---

## Project Structure

```
CV-Rock.Paper.Scissors/
├── rock_paper_scissorsVF.py   ← entire codebase
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── screenshots/
└── docs/
    └── architecture.md        ← full subsystem breakdown
```

---

## Future Improvements

- Online multiplayer over WebSockets
- Custom gesture training mode
- Round replay export as GIF
- Configurable AI difficulty profiles

---

## Author

Built by **Gurmannat Kaur** — exploring computer vision and real-time game programming.

- GitHub: [@GurmannatK](https://github.com/GurmannatK)
- LinkedIn: [Gurmannat Kaur](https://www.linkedin.com/in/gurmannat-kaur-730841282)

---

## License

MIT — see [LICENSE](LICENSE) for details.  
MediaPipe is licensed under Apache 2.0. The hand landmarker model is downloaded from Google's public model repository.
