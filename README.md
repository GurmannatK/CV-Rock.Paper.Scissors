# ✊✋✌ CV Rock Paper Scissors

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.x-00C800?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**A real-time hand gesture Rock Paper Scissors game powered by computer vision.**

Play with your actual hand in front of a webcam — no controllers, no keyboard for moves, just your fingers.

[Features](#features) · [Demo](#demo) · [Installation](#installation) · [How It Works](#how-it-works) · [Architecture](#architecture)

</div>

---

## What is this?

This started as me wanting to build something that actually *uses* the camera for gameplay rather than just recording. The result is a fully playable Rock Paper Scissors game where MediaPipe tracks your hand landmarks in real time, classifies your gesture each frame, and determines the winner against either an adaptive AI or a second local player.

The game runs at 60 FPS with a full pygame UI — scoreboard, particle effects, animated gesture reveals, per-gesture idle animations, and a tournament mode with up to 11-round Championship format.

---

## Features

### Gameplay
- **VS AI mode** — single player against a computer opponent
- **Local 2-Player mode** — both players use the same webcam, hands assigned by screen position (left = P1, right = P2)
- **Tournament system** — Single Round, Best of 3, Best of 5, Best of 7, or Championship (Best of 11)
- **Adaptive AI** — the opponent adjusts its play style based on your move history. The more you win, the smarter it gets. Randomness drops from 70% down to 10% as your win streak grows.

### Hand Tracking
- Built on **MediaPipe Hand Landmarker** with 21 keypoints per hand
- Custom gesture classifier using finger extension geometry and palm-to-tip distance ratios
- **14-frame vote buffer** with majority-vote smoothing — a single bad frame won't misfire
- EMA-smoothed confidence meter displayed in the HUD (separate "hand detected" and "gesture locked" bars)
- Real-time skeleton overlay drawn directly onto the camera feed

### Visual Polish
- **Per-gesture idle animations** — Rock slams with impact shockwaves, Paper floats in a figure-8, Scissors snaps with slash trails
- **Reveal animations** — each gesture has a unique entrance: Rock slams down from above (bounce easing), Paper unfolds with elastic spring, Scissors slashes in from the side
- Particle burst system with physics (gravity, drag, alpha decay)
- Scoreboard scan-beam animation, camera corner brackets, atmospheric glow blobs
- Screen flash on result reveal, winner glow aura, loser card dim effect
- Drifting star field on menu screens
- Full font fallback chain — works on Windows, macOS, and Linux

### Audio
- Synthesised sound effects (no audio files needed — generated with NumPy at startup)
- Separate sounds for: countdown tick, gesture reveal, win, loss, draw, hover, champion

---

## Demo

> **Screenshots below are placeholders — replace them with your own captures from the running game.**

### Main Menu
![Main Menu](screenshots/menu.png)
*Animated gesture cards with per-gesture idle physics. Rock breathes and slams, Paper drifts, Scissors snaps.*

### Gameplay (VS AI)
![Gameplay](screenshots/gameplay.png)
*Live webcam feed with hand skeleton overlay, confidence HUD, and countdown timer.*

### Round Result
![Result Screen](screenshots/results.png)
*Animated reveal cards. Winner gets a glow aura and crown badge; loser card dims. Particle bursts fire on win.*

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/cv-rock-paper-scissors.git
cd cv-rock-paper-scissors
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python rock_paper_scissorsVF.py
```

On first launch, the MediaPipe hand landmarker model (~8 MB) will download automatically and save as `hand_landmarker.task` next to the script. Subsequent launches use the cached file.

---

## Requirements

| Package | Tested Version | Purpose |
|---|---|---|
| `opencv-python` | 4.9+ | Webcam capture, frame processing |
| `mediapipe` | 0.10+ | Hand landmark detection |
| `numpy` | 1.24+ | Frame arrays, audio synthesis |
| `pygame` | 2.5+ | Window, rendering, input, audio |

Python **3.9 or later** required (tested on 3.10 and 3.11).

---

## Controls

| Key | Action |
|---|---|
| `1` | Quick Play / VS AI |
| `2` | Tournament Mode |
| `SPACE` | Next round / Back to menu from champion screen |
| `R` | Return to main menu from anywhere |
| `ESC` | Go back one screen / Quit from menu |

No key needed to make your move — just hold your gesture in frame and wait for **LOCK ACQUIRED**.

---

## How It Works

### Gesture Detection Pipeline

Each frame from the webcam goes through this chain:

1. **Capture** — OpenCV reads a raw BGR frame and mirrors it horizontally
2. **Landmark detection** — MediaPipe processes the RGB frame in VIDEO mode, returning up to 2 × 21 normalised hand landmarks
3. **Extension check** — For each finger, tip Y vs PIP Y decides if the finger is extended. The thumb uses X-axis displacement from the wrist instead.
4. **Classification** — Rule-based logic on the extension array:
   - `Rock`: all 4 fingers curled, or 1 finger extended but its tip is within `2× palm radius` of the wrist
   - `Paper`: 4+ fingers extended
   - `Scissors`: index + middle extended, ring + pinky curled
5. **Vote buffer** — Results from the last 14 frames are tallied; a gesture is only confirmed when it holds 8+ votes in the window
6. **EMA smoothing** — Confidence values use a slower decay (α=0.08) than rise (α=0.12), so the meter feels organic rather than jerky

### Adaptive AI

The AI opponent maintains a 40-move history deque. At each round:
- With probability `rand_prob` (starts at 0.70), it picks randomly
- Otherwise, it looks at the player's last 8 moves, finds the most common, and plays the gesture that beats it

Every time the player wins, `rand_prob` drops by 0.05 (floor 0.10). This means a player winning 12 consecutive rounds faces an AI that's nearly fully adaptive. The AI difficulty % is shown live in the scoreboard.

### Tournament System

The `Tournament` class tracks wins needed for each format (ceil(total/2)), records round outcomes, and sets a `champion` field when either side reaches the needed wins or all rounds are played. Tournament pip indicators show progress below the camera feed.

---

## Technologies Used

- **Python 3.10** — core language
- **OpenCV** — webcam I/O and frame preprocessing
- **MediaPipe Tasks (Hand Landmarker)** — hand detection and 21-point landmark tracking
- **NumPy** — array operations on frames; also used to synthesise audio waveforms procedurally
- **Pygame** — game loop, 60 FPS rendering, input handling, synthesised audio playback

No pre-trained gesture classifier is used — the classification is purely geometric, which makes it fast and interpretable.

---

## Project Architecture

```
cv-rock-paper-scissors/
├── rock_paper_scissorsVF.py   # entire codebase (single-file)
├── hand_landmarker.task       # auto-downloaded MediaPipe model (gitignored)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── screenshots/               # gameplay screenshots for README
└── docs/
    └── architecture.md        # detailed system design notes
```

For a full breakdown of each subsystem (camera pipeline, gesture classifier, AI, tournament, rendering, audio), see [`docs/architecture.md`](docs/architecture.md).

---

## Future Improvements

- **Online multiplayer** — WebSocket-based remote play so two players can face off over a network
- **Gesture training mode** — let users record custom gestures and re-train the classifier on their own hand shape
- **Replay system** — save rounds as GIFs or short video clips with the result overlay baked in
- **Configurable AI profiles** — different AI personalities (aggressive, defensive, random) selectable from the menu
- **Mobile port** — the core detection logic is platform-agnostic; a Kivy or BeeWare wrapper could make it run on Android/iOS

---

## Author

Built by **[Your Name]** as a personal project to explore computer vision and real-time game programming.

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your Name](https://linkedin.com/in/YOUR_PROFILE)

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

MediaPipe is licensed under Apache 2.0. The hand landmarker model is downloaded from Google's public model repository.
