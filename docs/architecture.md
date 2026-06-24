# Architecture Notes

This document breaks down the major subsystems in `rock_paper_scissorsVF.py`. The entire game lives in a single file (~1300 lines). Here's how it's organised and why each part works the way it does.

---

## Camera Pipeline

**Entry point:** `Game._grab()` → `Game._process()`

Every frame of the game loop calls `_grab()` first, which reads one frame from the OpenCV `VideoCapture` object and mirrors it horizontally (so the view is a natural mirror, not reversed). The buffer size is set to 1 (`CAP_PROP_BUFFERSIZE = 1`) to prevent frame lag accumulating over time.

`_process()` then converts the BGR frame to RGB (MediaPipe expects RGB), passes it to the tracker, and overlays the hand skeleton back onto the original BGR frame before it's rendered.

The camera feed is displayed as a 940×612 surface (`CAM_DW × CAM_DH`) centred in the 1280×720 window, scaled up from the native 640×480 capture resolution using `cv2.INTER_LINEAR`.

---

## MediaPipe Hand Tracking

**Class:** `HandTracker`

The game uses the **MediaPipe Tasks API** with `HandLandmarker` in `VIDEO` mode (not `LIVE_STREAM`). VIDEO mode is synchronous — each call to `detect_for_video()` returns results immediately — which keeps the game loop simple and avoids callback threading issues.

### Landmark indices used
```
_TIPS = (4, 8, 12, 16, 20)   # fingertip landmarks
_PIPS = (3, 7, 11, 15, 19)   # proximal interphalangeal joints
_MCPS = (2, 6, 10, 14, 18)   # metacarpophalangeal joints
```

### Multi-hand slot assignment
When two hands are in frame, they're assigned to slots 0 (P1, left side) and 1 (P2, right side) by sorting on the wrist X coordinate. For one hand, the tracker remembers the last known X position for each slot and assigns the single hand to whichever slot its wrist is closest to. This prevents P1 and P2 flickering when one hand dips out of frame.

### Confidence smoothing
Raw confidence is computed as `votes_for_winner / buffer_length`. This is then passed through an EMA:
- α = 0.12 when confidence is rising (fast response)
- α = 0.08 when confidence is falling (slow decay — meter doesn't drop instantly if hand wavers)

This is what drives the "SEARCHING → HAND DETECTED → LOCKING TARGET → LOCK ACQUIRED" HUD states.

---

## Gesture Classification

**Method:** `HandTracker._classify(lms)`

Classification is purely geometric — no ML model for the gesture itself, just the hand landmark positions.

### Step 1 — finger extension check (`_ext`)
For fingers 2–5 (index through pinky): a finger is "extended" if its tip Y coordinate is above its PIP joint Y coordinate by more than `FIST_MARGIN = 0.018` (in normalised 0–1 space).

For the thumb (finger 1): extension is determined by X displacement from wrist rather than Y, since the thumb extends sideways not upward.

### Step 2 — Rock disambiguation (`_is_rock`)
Rock is the trickiest because a loose fist can look like 0 extended fingers, but sometimes one finger is slightly up. The check:
- 0 extended fingers → Rock immediately
- Exactly 1 extended finger → measure its tip-to-wrist distance; if it's less than `2.0 × palm_radius`, it's still curled enough to count as Rock

Palm radius is the wrist-to-middle-MCP distance, used as a scaling reference.

### Step 3 — Final classification
```
Rock     → _is_rock() returns True
Paper    → 4+ fingers extended
Scissors → index + middle extended, ring + pinky not
```

### Step 4 — Vote buffer (`_vote`)
The last 14 raw classifications are stored in a `deque`. A gesture is only returned as "confirmed" if one label holds at least 8 votes in the current window. This 14/8 ratio tolerates up to 6 bad frames per window without misfiring.

---

## Adaptive AI System

**Class:** `AdaptiveAI`

The AI stores the player's last 40 moves in a deque. Each round it decides:

```python
if random.random() < self.rand_prob:
    return random.choice(GESTURE_LIST)   # pure random
else:
    # find player's most common move in last 8 rounds
    # play the gesture that beats it
```

`rand_prob` starts at 0.70 and decrements by 0.05 on each player win, floored at 0.10. So:

| Player wins in a row | AI rand_prob | AI behaviour |
|---|---|---|
| 0 | 70% | Mostly random |
| 6 | 40% | Balanced |
| 12 | 10% | Nearly fully adaptive |

The AI difficulty percentage shown in the scoreboard is `(1 - rand_prob) * 100`.

---

## Tournament System

**Class:** `Tournament`

Formats: Single (1), Best of 3, 5, 7, Championship (11).

`needed = ceil(total / 2)` — the number of wins required. The tournament tracks `p1_wins`, `p2_wins`, and `played`. A champion is declared as soon as one side reaches `needed` wins, which means a Best of 5 can end in 3 rounds if one player sweeps.

The `champion` property returns `"p1"`, `"p2"`, or `"draw"` (only possible when all rounds are played and scores are tied).

Tournament pip indicators below the camera feed show progress as small filled rectangles, one per needed win per side.

---

## UI Rendering System

**Class:** `Game` — `_draw_*` methods

The render loop follows this order every frame:

1. Blit pre-rendered static background (built once at startup: dark grid + atmospheric colour blobs + vignette)
2. Draw star field particles (menu screens only)
3. Draw explosion/ring particles
4. Dispatch to the current state's draw method
5. Blit scanline overlay (built once: semi-transparent horizontal lines at 4px intervals)
6. Apply screen flash overlay if active
7. Blit FPS counter

### Pre-rendering strategy
Heavy surfaces that don't change are built once in `_Statics` and cached:
- Background (grid + atmospheric glows + vignette)
- Scoreboard chrome (glass bevel, accent borders, tile dividers)
- Camera gradient (bottom fade-out)
- Scanline texture
- Countdown number glyphs at 128px (baked per label/colour)
- Small gesture icon surfaces (46px per gesture, used in live HUD)

Per-frame SRCALPHA surface allocations are kept to a minimum — most effects use direct `lerp_col()` rectangle fills or `set_alpha()` on pre-cached surfaces.

### Easing functions
Four easing functions are implemented:
- `ease_out(t)` — cubic ease-out, used for slide-in animations
- `ease_in_out(t)` — smooth step, used for general transitions
- `ease_elastic(t)` — spring overshoot, used for Paper unfold reveal
- `ease_bounce(t)` — used for Rock slam reveal

### Gesture idle animations
Each gesture has a unique procedural idle (`gesture_idle()`):
- **Rock** — slow vertical breathing + sharp downbeat slam pulse (`sin(t*2.4) ** 9`)
- **Paper** — figure-8 drift pattern (different frequencies on X and Y)
- **Scissors** — rapid snap jitter (`sin(t*4.8) ** 3`) with aggressive rotation

The idle clock for each gesture runs independently (`_idle_t` dict) so the animation phase is preserved even when a gesture isn't on screen.

### Reveal animations
Each gesture has a named reveal style (rock: `"slam"`, paper: `"unfold"`, scissors: `"slash"`) handled in `draw_gesture_animated()`. P1's card reveals immediately; P2 (AI) has a 0.22-second delay for dramatic effect.

---

## Audio System

**Class:** `AudioManager`

No audio files are loaded from disk. All sounds are synthesised at startup using NumPy:

```python
t = np.linspace(0, dur, int(sr * dur))
w = np.sin(2 * np.pi * freq * t)
# optional harmonics: add 2nd and 3rd partial
```

The waveform is normalised, converted to int16, given a 10% linear fade-out tail, then made into a stereo array and passed to `pygame.sndarray.make_sound()`.

Sounds by event:
| Sound | Freq | Dur | Notes |
|---|---|---|---|
| `tick` | 880 Hz | 0.06s | Countdown beat |
| `reveal` | 1046 Hz | 0.15s | With harmonics |
| `win` | 660 Hz | 0.34s | With harmonics |
| `lose` | 200 Hz | 0.38s | Low tone |
| `draw` | 440 Hz | 0.24s | Mid tone |
| `select` | 550 Hz | 0.05s | Menu selection |
| `champ` | 880 Hz | 0.55s | With harmonics |
| `click_hi` | 1320 Hz | 0.04s | Hover enter |
| `click_lo` | 660 Hz | 0.04s | Secondary click |

Audio initialisation is wrapped in a try/except — if the mixer fails (e.g., no audio device), the game continues silently.

---

## Gesture Vector Icon System

**Function:** `draw_gesture_icon(surf, gesture, cx, cy, size, color, alpha)`

Rather than relying on emoji (✊✋✌ render as empty boxes on many systems), all gesture icons are drawn with Pygame primitives at arbitrary scale. Each gesture is constructed from:
- `pygame.draw.rect()` with `border_radius` for palms and finger segments
- `pygame.draw.polygon()` for thumbs and angled shapes
- `pygame.draw.circle()` for fingertip caps
- `pygame.draw.line()` for knuckle highlights and crease details

The drawing uses a `p(v)` helper that converts proportional values (0.0–1.0) to pixels relative to the icon size, so the same function renders correctly at 46px (live HUD) or 130px (result card).

---

## State Machine

```
MENU ──→ MODE_SEL ──→ COUNTDOWN ──→ RESULT ──→ COUNTDOWN (loop)
  └──→ TOURN_SEL ──→ MODE_SEL               └──→ CHAMPION (tournament over)
                                                        └──→ MENU
```

State is stored in `self.state` as a string constant from class `S`. All keyboard and mouse events are routed through `_key()` and `_action()`, which call transition methods like `_begin_cd()`, `_next_round()`, and `_to_menu()`.
