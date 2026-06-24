"""
╔══════════════════════════════════════════════════════════════════════╗
║   ROCK · PAPER · SCISSORS  —  Hand Gesture Game  v1.0               ║
║   Real-time webcam · Single player vs AI · Local 2-Player           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Install:  pip install opencv-python mediapipe numpy pygame          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Controls                                                            ║
║    1 / 2    choose mode            SPACE   next round                ║
║    ESC      back / quit            R       main menu                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pygame
import sys, time, random, math, urllib.request, pathlib
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict

from mediapipe.tasks.python.vision                 import (HandLandmarker,
                                                            HandLandmarkerOptions,
                                                            RunningMode)
from mediapipe.tasks.python.core.base_options      import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe                                      import Image, ImageFormat

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H   = 1280, 720
FPS_TARGET     = 60

CAM_INDEX      = 0
CAM_W, CAM_H   = 640, 480

HISTORY_LEN    = 14
MIN_VOTES      = 8
FIST_MARGIN    = 0.018
ROCK_PALM_MULT = 2.0

COUNTDOWN_DUR  = 1.0

AI_RANDOM_BASE = 0.70
AI_ADAPT_STEP  = 0.05
AI_MIN_RANDOM  = 0.10
AI_HISTORY_WIN = 8

CAM_DW = 940
CAM_DH = 612
CAM_X  = (WIN_W - CAM_DW) // 2   # 170
CAM_Y  = 88

SB_H   = 78
SB_PAD = 8

_FONT_PREF = ["bahnschrift", "segoeui", "calibri", "helvetica", "dejavusans", "freesans", None]
_FONT_MONO = ["consolas", "couriernew", "dejavusansmono", None]

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG        = (  8,  10,  16)
C_PANEL     = ( 16,  20,  32)
C_PANEL_LT  = ( 26,  33,  52)
C_PANEL_MD  = ( 20,  26,  42)
C_BORDER    = ( 36,  46,  70)
C_ACCENT    = (  0, 210, 175)
C_ACCENT2   = (110,  48, 230)
C_WHITE     = (218, 228, 242)
C_GREY      = ( 88, 104, 128)
C_DIM       = ( 40,  50,  68)
C_RED       = (255,  65,  88)
C_GREEN     = ( 48, 230, 130)
C_YELLOW    = (255, 204,  42)
C_DRAW      = (255, 168,  42)
C_P1        = (  0, 188, 255)
C_P2        = (255,  68, 168)

GESTURE_LIST  = ["Rock", "Paper", "Scissors"]
BEATS         = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
GESTURE_COLOR = {"Rock": C_RED, "Paper": C_P1, "Scissors": C_P2}
BEAT_VERB     = {
    ("Rock",     "Scissors"): "crushes",
    ("Paper",    "Rock"):     "covers",
    ("Scissors", "Paper"):    "cuts",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE VISUAL REGISTRY  — single source of truth for all gesture visuals
#  Every screen must resolve icons/labels through this dict, never hardcode.
# ═══════════════════════════════════════════════════════════════════════════════

GESTURE_ICON  = {"Rock": "✊", "Paper": "✋", "Scissors": "✌"}
GESTURE_HINT  = {"Rock": "closed fist", "Paper": "open palm", "Scissors": "two fingers"}
UNKNOWN_ICON  = "?"

# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE VECTOR ICONS  — drawn with pygame primitives, always visible
#  Emoji ✊✋✌ render as blank boxes on many systems — these never fail.
# ═══════════════════════════════════════════════════════════════════════════════

def draw_gesture_icon(surf, gesture, cx, cy, size, color, alpha=255):
    """
    Draw a high-quality vector gesture icon centred at (cx, cy).
    Anatomically correct hands — no emoji font required.
    gesture: "Rock" | "Paper" | "Scissors" | None
    """
    if alpha <= 0:
        return
    tmp  = pygame.Surface((size, size), pygame.SRCALPHA)
    s    = size
    h    = s // 2          # centre
    c    = (color[0], color[1], color[2], alpha)
    # highlight colour — lighter version of c for detail lines
    hi   = (min(255, color[0] + 60), min(255, color[1] + 60),
            min(255, color[2] + 60), max(0, alpha - 60))
    # shadow colour — darker
    sh   = (color[0]//2, color[1]//2, color[2]//2, alpha)

    def _rr(rect, r):
        """Draw filled rounded rect on tmp."""
        pygame.draw.rect(tmp, c, rect, border_radius=r)

    def _poly(pts):
        pygame.draw.polygon(tmp, c, pts)

    def _circ(cx2, cy2, r2):
        pygame.draw.circle(tmp, c, (cx2, cy2), r2)

    # ── Scaled helpers (proportional to `s`) ──────────────────────────────────
    def p(v):   return int(s * v)   # proportion of size
    def ph(v):  return h + int(s * v)  # offset from centre

    if gesture == "Rock":
        # ── Tight closed fist viewed from the front ────────────────────────────
        # Main knuckle row: 4 rounded rectangles side by side, slight arch
        knuckle_offsets = [-0.285, -0.095, 0.095, 0.285]
        knuckle_w = p(0.175)
        knuckle_h = p(0.18)
        knuckle_top = ph(-0.14)

        # Under-palm (lower block — fingers curled inward)
        palm_x = ph(-0.30)
        palm_y = ph(-0.04)
        palm_w = p(0.60)
        palm_h = p(0.30)
        _rr((palm_x, palm_y, palm_w, palm_h), p(0.08))

        # 4 knuckle caps along the top
        for i, koff in enumerate(knuckle_offsets):
            kx = ph(koff) - knuckle_w // 2
            ky = knuckle_top - knuckle_h // 2
            arch = p(0.02) * abs(abs(i - 1.5) - 1.5)  # slight arch: middle higher
            _rr((kx, ky - arch, knuckle_w, knuckle_h + arch), p(0.05))

        # Knuckle highlight lines
        for koff in knuckle_offsets:
            kx = ph(koff)
            pygame.draw.line(tmp, hi,
                             (kx - p(0.05), knuckle_top - p(0.01)),
                             (kx + p(0.05), knuckle_top - p(0.01)), max(1, p(0.015)))

        # Thumb — on the right, angled outward and down
        tx0 = ph(0.27)
        ty0 = ph(0.04)
        thumb_pts = [
            (tx0,          ty0),
            (tx0 + p(0.18), ty0 - p(0.10)),
            (tx0 + p(0.20), ty0 + p(0.04)),
            (tx0 + p(0.10), ty0 + p(0.14)),
            (tx0 - p(0.01), ty0 + p(0.10)),
        ]
        _poly(thumb_pts)
        # Thumb tip highlight
        pygame.draw.arc(tmp, hi,
                        (tx0 + p(0.10), ty0 - p(0.12),
                         p(0.12), p(0.12)), 0, math.pi, max(1, p(0.018)))

        # Finger crease lines across palm
        crease_y = ph(0.08)
        pygame.draw.line(tmp, sh,
                         (ph(-0.28), crease_y), (ph(0.28), crease_y),
                         max(1, p(0.018)))
        crease_y2 = ph(0.18)
        pygame.draw.line(tmp, sh,
                         (ph(-0.22), crease_y2), (ph(0.22), crease_y2),
                         max(1, p(0.012)))

    elif gesture == "Paper":
        # ── Open flat hand — 4 fingers + thumb, viewed from front ─────────────
        # Layout: thumb on left, 4 fingers (index→pinky) across
        # Finger proportions — index slightly tallest, pinky shortest
        fw = p(0.115)   # finger width
        fg = p(0.028)   # gap between fingers

        # X positions of finger centres (index=leftmost, pinky=rightmost)
        # Shifted right a bit to leave room for thumb on left
        f_centres = [ph(-0.15), ph(-0.03), ph(0.09), ph(0.21)]
        # Finger heights (from tip to palm junction)
        f_heights = [p(0.44), p(0.47), p(0.44), p(0.36)]
        # Palm top Y — where fingers emerge
        palm_top_y = ph(0.04)
        # Palm bottom
        palm_bot_y = ph(0.34)

        # Palm body
        palm_l = f_centres[0] - fw // 2 - fg
        palm_r = f_centres[3] + fw // 2 + fg
        palm_w2 = palm_r - palm_l
        _rr((palm_l, palm_top_y, palm_w2, palm_bot_y - palm_top_y), p(0.07))

        # 4 fingers — drawn as rounded rects
        for fi, (fcx, fh2) in enumerate(zip(f_centres, f_heights)):
            fx = fcx - fw // 2
            fy = palm_top_y - fh2 + fw // 2
            _rr((fx, fy, fw, fh2), fw // 2)
            # Knuckle highlight at base of each finger
            pygame.draw.line(tmp, hi,
                             (fcx - fw // 2 + p(0.01), palm_top_y + p(0.01)),
                             (fcx + fw // 2 - p(0.01), palm_top_y + p(0.01)),
                             max(1, p(0.012)))

        # Thumb — exits bottom-left of palm, angled left and slightly up
        thw = p(0.105)
        # Base connects at left side of palm
        th_base_x = palm_l + p(0.02)
        th_base_y = palm_top_y + p(0.08)
        thumb2_pts = [
            (th_base_x,             th_base_y),
            (th_base_x - p(0.16),   th_base_y - p(0.04)),
            (th_base_x - p(0.21),   th_base_y + p(0.08)),
            (th_base_x - p(0.16),   th_base_y + p(0.18)),
            (th_base_x - p(0.02),   th_base_y + p(0.20)),
        ]
        _poly(thumb2_pts)
        # Thumb tip rounded cap
        _circ(th_base_x - p(0.19), th_base_y + p(0.03), thw // 2)

        # Palm crease
        pygame.draw.line(tmp, sh,
                         (palm_l + p(0.04), palm_top_y + p(0.14)),
                         (palm_r - p(0.04), palm_top_y + p(0.14)),
                         max(1, p(0.016)))

    elif gesture == "Scissors":
        # ── Two fingers spread in a V, curled fist body below ─────────────────
        fw = p(0.13)    # finger width

        # Fist body (lower portion — ring+pinky curled into palm)
        fist_top  = ph(0.06)
        fist_bot  = ph(0.40)
        fist_l    = ph(-0.28)
        fist_r    = ph(0.28)
        _rr((fist_l, fist_top, fist_r - fist_l, fist_bot - fist_top), p(0.09))

        # Thumb tucked at right side of fist
        ttx = ph(0.25)
        tty = ph(0.08)
        thumb3_pts = [
            (ttx,          tty),
            (ttx + p(0.14), tty - p(0.08)),
            (ttx + p(0.16), tty + p(0.06)),
            (ttx + p(0.06), tty + p(0.14)),
            (ttx - p(0.01), tty + p(0.10)),
        ]
        _poly(thumb3_pts)

        # Two extended fingers spread in a V shape
        # Index finger — leans LEFT
        idx_tip_x = ph(-0.20)
        idx_tip_y = ph(-0.40)
        idx_base_x = ph(-0.07)
        idx_base_y = fist_top + p(0.04)
        idx_pts = [
            (idx_base_x - fw // 2,  idx_base_y),
            (idx_base_x + fw // 2,  idx_base_y),
            (idx_tip_x  + fw // 2,  idx_tip_y),
            (idx_tip_x  - fw // 2,  idx_tip_y),
        ]
        _poly(idx_pts)
        _circ(idx_tip_x, idx_tip_y, fw // 2)   # rounded tip

        # Middle finger — leans RIGHT
        mid_tip_x = ph(0.20)
        mid_tip_y = ph(-0.40)
        mid_base_x = ph(0.07)
        mid_base_y = fist_top + p(0.04)
        mid_pts = [
            (mid_base_x - fw // 2,  mid_base_y),
            (mid_base_x + fw // 2,  mid_base_y),
            (mid_tip_x  + fw // 2,  mid_tip_y),
            (mid_tip_x  - fw // 2,  mid_tip_y),
        ]
        _poly(mid_pts)
        _circ(mid_tip_x, mid_tip_y, fw // 2)   # rounded tip

        # Finger highlight lines
        for (bx, by, tx, ty) in [
            (idx_base_x, idx_base_y, idx_tip_x, idx_tip_y),
            (mid_base_x, mid_base_y, mid_tip_x, mid_tip_y),
        ]:
            mid_x = (bx + tx) // 2
            mid_y = (by + ty) // 2
            pygame.draw.line(tmp, hi,
                             (mid_x - p(0.02), mid_y + p(0.02)),
                             (mid_x + p(0.02), mid_y - p(0.02)),
                             max(1, p(0.018)))

        # Knuckle detail lines on fist
        for koff in (-0.14, 0.01):
            ky3 = fist_top + p(0.06)
            pygame.draw.line(tmp, hi,
                             (ph(koff) - p(0.04), ky3),
                             (ph(koff) + p(0.04), ky3),
                             max(1, p(0.015)))

    else:
        # Unknown — question mark circle
        pygame.draw.circle(tmp, c, (h, h), h - 2, max(1, p(0.03)))
        q = _font(s // 2).render("?", True, color)
        tmp.blit(q, (h - q.get_width()//2, h - q.get_height()//2))

    surf.blit(tmp, (cx - h, cy - h))


def _bake_gesture_surf(gesture, size, color):
    """Return a pre-rendered SRCALPHA surface with the gesture icon."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    draw_gesture_icon(s, gesture, size//2, size//2, size, color)
    return s


def g_icon(gesture):
    """Kept for API compatibility — returns the gesture name (used as fallback label)."""
    return gesture or "?"


def g_color(gesture, fallback=None):
    """Return gesture accent color, or fallback."""
    return GESTURE_COLOR.get(gesture, fallback or (88, 104, 128))

# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE COMBAT FRAMEWORK
#  Architecture-first metadata for future combat/special-move mechanics.
#  The current game is fully playable; this data is ready for future extensions.
# ═══════════════════════════════════════════════════════════════════════════════

class GestureClass:
    HEAVY   = "heavy"
    CONTROL = "control"
    FAST    = "fast"

class GestureElement:
    EARTH = "earth"
    WIND  = "wind"
    BLADE = "blade"

GESTURE_COMBAT = {
    "Rock": {
        "class":           GestureClass.HEAVY,
        "element":         GestureElement.EARTH,
        "attack_power":    85,
        "speed":           40,
        "defense":         70,
        "combo_potential": 2,
        "crit_chance":     0.12,
        "special_name":    "Meteor Strike",
        "special_hook":    None,
        "anim_style":      "slam",
        "description":     "Slow but devastating. Crushes bladed attacks.",
        "color":           C_RED,
    },
    "Paper": {
        "class":           GestureClass.CONTROL,
        "element":         GestureElement.WIND,
        "attack_power":    55,
        "speed":           65,
        "defense":         90,
        "combo_potential": 3,
        "crit_chance":     0.08,
        "special_name":    "Envelop",
        "special_hook":    None,
        "anim_style":      "wrap",
        "description":     "Defensive and persistent. Covers heavy attacks.",
        "color":           C_P1,
    },
    "Scissors": {
        "class":           GestureClass.FAST,
        "element":         GestureElement.BLADE,
        "attack_power":    65,
        "speed":           90,
        "defense":         45,
        "combo_potential": 4,
        "crit_chance":     0.22,
        "special_name":    "Twin Blades",
        "special_hook":    None,
        "anim_style":      "slash",
        "description":     "Lightning fast with combo potential. Shreds paper.",
        "color":           C_P2,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FONT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_FC: Dict[Tuple, pygame.font.Font] = {}

def _font(size: int, mono: bool = False) -> pygame.font.Font:
    key = (size, mono)
    if key in _FC:
        return _FC[key]
    for name in (_FONT_MONO if mono else _FONT_PREF):
        try:
            _FC[key] = pygame.font.SysFont(name, size)
            return _FC[key]
        except Exception:
            pass
    _FC[key] = pygame.font.Font(None, size)
    return _FC[key]

def _warm_fonts():
    for sz in (14, 15, 16, 18, 20, 22, 24, 28, 30, 32, 36, 44, 52, 58, 60, 66, 72, 88, 128):
        _font(sz)
        _font(sz, mono=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def _synth(freq, dur, vol=0.30, sr=44100, harmonics=False):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    w = np.sin(2 * np.pi * freq * t)
    if harmonics:
        w += 0.28 * np.sin(4 * np.pi * freq * t)
        w += 0.08 * np.sin(6 * np.pi * freq * t)
    peak = np.max(np.abs(w))
    if peak > 0:
        w = w / peak * vol * 32767
    w = w.astype(np.int16)
    fade = max(1, int(len(w) * 0.10))
    w[-fade:] = (w[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.stack([w, w], -1)))


class AudioManager:
    def __init__(self):
        self.ok = False
        self._s  = {}

    def init(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._s = {
                "tick":     _synth(880,  0.06),
                "reveal":   _synth(1046, 0.15, harmonics=True),
                "win":      _synth(660,  0.34, harmonics=True),
                "lose":     _synth(200,  0.38),
                "draw":     _synth(440,  0.24),
                "select":   _synth(550,  0.05),
                "champ":    _synth(880,  0.55, harmonics=True),
                "click_hi": _synth(1320, 0.04, vol=0.18),
                "click_lo": _synth(660,  0.04, vol=0.15),
            }
            self.ok = True
        except Exception:
            pass

    def play(self, name):
        if self.ok:
            try:
                self._s[name].play()
            except Exception:
                pass

audio = AudioManager()

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MODEL_PATH = pathlib.Path(__file__).parent / "hand_landmarker.task"

def ensure_model() -> str:
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 1_000:
        return str(_MODEL_PATH)
    print("[RPS] Downloading hand landmarker model...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[RPS] Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Cannot download MediaPipe model.\n"
            f"Download manually: {_MODEL_URL}\n"
            f"Save as 'hand_landmarker.task' beside this script.\n{e}")
    return str(_MODEL_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
#  HAND TRACKER  — detection logic unchanged (it works perfectly)
# ═══════════════════════════════════════════════════════════════════════════════

_TIPS = (4, 8, 12, 16, 20)
_PIPS = (3, 7, 11, 15, 19)
_MCPS = (2, 6, 10, 14, 18)
_TIPS_SET = frozenset(_TIPS)
_BONES = (
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
)
_BONE_BGR = (0, 195, 155)
_TIP_BGR  = (255, 255, 255)
_JNT_BGR  = (90, 30, 185)


class HandTracker:
    def __init__(self, max_hands=2, model_path=""):
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        self._det      = HandLandmarker.create_from_options(opts)
        self.max_hands = max_hands
        self._bufs     = [deque(maxlen=HISTORY_LEN) for _ in range(max_hands)]
        self._slot_x   : List[Optional[float]] = [None] * max_hands
        self.conf      : List[float]            = [0.0]  * max_hands
        # Smoothed confidence — gradual EMA, no instant jumps
        self.conf_smooth : List[float]          = [0.0]  * max_hands
        self._t0       = int(time.perf_counter() * 1000)

    def close(self):
        self._det.close()

    def _ext(self, lms):
        w = lms[0]
        out = [abs(lms[_TIPS[0]].x - w.x) > abs(lms[_MCPS[0]].x - w.x) + FIST_MARGIN]
        for i in range(1, 5):
            out.append(lms[_TIPS[i]].y < lms[_PIPS[i]].y - FIST_MARGIN)
        return out

    def _palm_r(self, lms):
        return math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)

    def _is_rock(self, lms, ext):
        n = sum(ext[1:])
        if n == 0:
            return True
        if n == 1:
            pr = self._palm_r(lms)
            for i in range(1, 5):
                if ext[i]:
                    d = math.hypot(lms[_TIPS[i]].x - lms[0].x,
                                   lms[_TIPS[i]].y - lms[0].y)
                    if d < pr * ROCK_PALM_MULT:
                        return True
        return False

    def _classify(self, lms):
        ext = self._ext(lms)
        _, idx, mid, rng, pky = ext
        n = idx + mid + rng + pky
        if self._is_rock(lms, ext):                  return "Rock"
        if n >= 4:                                    return "Paper"
        if idx and mid and not rng and not pky:       return "Scissors"
        if idx and not mid and not rng and not pky:   return "Scissors"
        return None

    def _vote(self, buf, raw, slot):
        buf.append(raw)
        counts = Counter(g for g in buf if g is not None)
        # Raw confidence from vote majority
        if not counts:
            raw_conf = 0.0
        else:
            best, n = counts.most_common(1)[0]
            raw_conf = n / len(buf)
        self.conf[slot] = raw_conf
        # Smooth confidence: EMA — α=0.12 rise, α=0.08 fall
        # Prevents instant 0→100% jumps; meter feels realistic and gradual
        alpha = 0.12 if raw_conf > self.conf_smooth[slot] else 0.08
        self.conf_smooth[slot] += alpha * (raw_conf - self.conf_smooth[slot])
        if not counts:
            return None
        best, n = counts.most_common(1)[0]
        return best if n >= MIN_VOTES else None

    def _assign_slots(self, hands):
        if self.max_hands == 1:
            return [0] if hands else []
        n = len(hands)
        if n == 0:
            return []
        if n == 1:
            wx = hands[0][0].x
            s0, s1 = self._slot_x
            if s0 is None and s1 is None:
                return [0 if wx < 0.5 else 1]
            if s0 is None:
                return [1]
            if s1 is None:
                return [0]
            return [0 if abs(wx - s0) <= abs(wx - s1) else 1]
        order = sorted(range(n), key=lambda i: hands[i][0].x)
        return [order[0], order[1]]

    def process(self, rgb):
        ts  = int(time.perf_counter() * 1000) - self._t0
        res = self._det.detect_for_video(
            Image(image_format=ImageFormat.SRGB, data=rgb), ts)
        hands = res.hand_landmarks
        slots = self._assign_slots(hands)
        g_list = [None] * self.max_hands
        for di, slot in enumerate(slots):
            lms = hands[di]
            self._slot_x[slot] = lms[0].x
            g_list[slot] = self._vote(self._bufs[slot], self._classify(lms), slot)
        active = set(slots)
        for slot in range(self.max_hands):
            if slot not in active:
                self._slot_x[slot] = None
                self.conf[slot]     = max(0.0, self.conf[slot] - 0.10)
                # Smooth decay too
                self.conf_smooth[slot] = max(0.0, self.conf_smooth[slot] - 0.04)
                self._bufs[slot].append(None)
        return res, g_list

    def draw_hands(self, bgr, res):
        h, w = bgr.shape[:2]
        for lms in res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in _BONES:
                cv2.line(bgr, pts[a], pts[b], _BONE_BGR, 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                tip = i in _TIPS_SET
                cv2.circle(bgr, pt, 5 if tip else 3,
                           _TIP_BGR if tip else _JNT_BGR, -1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE AI
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveAI:
    def __init__(self):
        self.history   = deque(maxlen=40)
        self.rand_prob = AI_RANDOM_BASE
        self.wins      = 0

    def record(self, g):
        self.history.append(g)

    def on_player_win(self):
        self.wins     += 1
        self.rand_prob = max(AI_MIN_RANDOM,
                             AI_RANDOM_BASE - self.wins * AI_ADAPT_STEP)

    def choose(self):
        if random.random() < self.rand_prob or len(self.history) < AI_HISTORY_WIN:
            return random.choice(GESTURE_LIST)
        likely   = Counter(list(self.history)[-AI_HISTORY_WIN:]).most_common(1)[0][0]
        counters = [g for g, b in BEATS.items() if b == likely]
        return counters[0] if counters else random.choice(GESTURE_LIST)

# ═══════════════════════════════════════════════════════════════════════════════
#  TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════════

class Tournament:
    FORMATS = [
        ("Single Round",  1),
        ("Best of 3",     3),
        ("Best of 5",     5),
        ("Best of 7",     7),
        ("Championship", 11),
    ]

    def __init__(self, total=1):
        self.total   = total
        self.needed  = (total + 1) // 2
        self.p1_wins = 0
        self.p2_wins = 0
        self.played  = 0
        self.champion = None

    def record(self, winner):
        self.played += 1
        if winner == "p1":
            self.p1_wins += 1
        elif winner == "p2":
            self.p2_wins += 1
        if self.p1_wins >= self.needed:
            self.champion = "p1"
        elif self.p2_wins >= self.needed:
            self.champion = "p2"
        elif self.played >= self.total and self.p1_wins == self.p2_wins:
            self.champion = "draw"

    @property
    def over(self):
        return self.champion is not None

    @property
    def score_str(self):
        return f"{self.p1_wins}  —  {self.p2_wins}"

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

_PSURF: Dict[int, pygame.Surface] = {}

def _psurf(r):
    if r not in _PSURF:
        _PSURF[r] = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
    return _PSURF[r]


def panel(surf, fill, rect, r=12, bw=0, bc=None, alpha=255):
    """Rounded rect. Direct draw when opaque; SRCALPHA surface only when alpha<255."""
    if alpha == 255:
        pygame.draw.rect(surf, fill, rect, border_radius=r)
    else:
        x, y, w, h = rect
        tmp = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*fill, alpha), (0, 0, w, h), border_radius=r)
        surf.blit(tmp, (x, y))
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, width=bw, border_radius=r)


def txt(surf, text, fnt, color, pos, center=True,
        glow=False, glow_col=None, glow_r=5):
    """Crisp text with optional single-pass glow (4 cardinal offsets, alpha 38)."""
    base = fnt.render(text, True, color)
    bw, bh = base.get_size()
    rx = pos[0] - bw // 2 if center else pos[0]
    ry = pos[1] - bh // 2 if center else pos[1]
    if glow and glow_col:
        gr = glow_r
        gs = pygame.Surface((bw + gr * 2, bh + gr * 2), pygame.SRCALPHA)
        gs.blit(fnt.render(text, True, glow_col), (gr, gr))
        gs.set_alpha(38)
        for dx, dy in ((gr, 0), (-gr, 0), (0, gr), (0, -gr)):
            surf.blit(gs, (rx - gr + dx, ry - gr + dy))
    surf.blit(base, (rx, ry))
    return pygame.Rect(rx, ry, bw, bh)


def cam_to_surf(bgr, w, h):
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))


def lerp_col(a, b, t):
    return (int(a[0] + (b[0]-a[0])*t),
            int(a[1] + (b[1]-a[1])*t),
            int(a[2] + (b[2]-a[2])*t))


def combat_stat_bar(surf, x, y, w, h, value, max_val, color, label=""):
    """Utility: draw a filled stat bar. Ready for future combat HUD extensions."""
    fill = max(0, int(w * value / max_val))
    pygame.draw.rect(surf, (20, 26, 40), (x, y, w, h), border_radius=h//2)
    if fill > 0:
        pygame.draw.rect(surf, color, (x, y, fill, h), border_radius=h//2)
        pygame.draw.rect(surf, lerp_col(color, (255, 255, 255), 0.35),
                         (x, y, fill, max(1, h//3)), border_radius=h//2)


def conf_color(c):
    if c < 0.5:
        return (255, int(c * 2 * 210), 40)
    return (int((1 - c) * 2 * 255), 210, 40)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def ease_elastic(t):
    """Overshoot spring — good for reveal pops."""
    t = max(0.0, min(1.0, t))
    c = 2 * math.pi / 3
    if t == 0: return 0.0
    if t == 1: return 1.0
    return (2 ** (-10 * t)) * math.sin((t * 10 - 0.75) * c) + 1.0

def ease_bounce(t):
    n1, d1 = 7.5625, 2.75
    if t < 1/d1:       return n1 * t * t
    elif t < 2/d1:     t -= 1.5/d1;   return n1*t*t + 0.75
    elif t < 2.5/d1:   t -= 2.25/d1;  return n1*t*t + 0.9375
    else:              t -= 2.625/d1; return n1*t*t + 0.984375

# ── Per-gesture idle transform helper ─────────────────────────────────────────
# Returns (dx, dy, angle_deg, scale) relative to centre at time t

def gesture_idle(gesture, t, intensity=1.0):
    """Procedural idle animation per gesture. Returns (dx, dy, scale, angle)."""
    if gesture == "Rock":
        # Heavy breathing: slow vertical, slight lateral sway, impact slam pulse
        slam_raw  = math.sin(t * 2.4)
        slam      = max(0.0, slam_raw ** 9) * intensity   # only on downbeat
        dy        = math.sin(t * 0.95) * 3.5 * intensity + slam * 10
        dx        = math.sin(t * 0.48) * 2.0 * intensity
        scale     = 1.0 + 0.028 * math.sin(t * 0.95) * intensity + 0.06 * slam
        angle     = math.sin(t * 0.52) * 1.2 * intensity
    elif gesture == "Paper":
        # Graceful float: figure-8 drift, gentle rotation, slow breathe
        dy        = math.sin(t * 0.68) * 6.0 * intensity
        dx        = math.sin(t * 0.34) * 3.5 * intensity
        scale     = 1.0 + 0.018 * math.sin(t * 0.72) * intensity
        angle     = math.sin(t * 0.58) * 2.2 * intensity
    else:  # Scissors
        # Aggressive: quick snappy jitter, sharper rotation, slash twitch
        snap      = math.sin(t * 4.8) ** 3 * intensity
        dy        = math.sin(t * 1.8) * 3.0 * intensity + snap * 4
        dx        = math.cos(t * 3.6) * 2.5 * intensity
        scale     = 1.0 + 0.035 * abs(math.sin(t * 2.4)) * intensity
        angle     = math.sin(t * 2.2) * 3.5 * intensity + snap * 6
    return dx, dy, scale, angle


def draw_gesture_animated(surf, gesture, cx, cy, size, color, t,
                           intensity=1.0, alpha=255, reveal_t=1.0,
                           reveal_style="pop"):
    """
    Draw gesture icon with full procedural idle animation.
    reveal_t: 0→1, drives the entrance animation.
    reveal_style: "pop" | "slam" | "unfold" | "slash"
    """
    if alpha <= 0:
        return

    dx, dy, idle_scale, angle = gesture_idle(gesture, t, intensity)

    # Reveal animation layer on top of idle
    if reveal_t < 1.0:
        if reveal_style == "slam":          # Rock: falls from above
            drop = (1 - ease_bounce(reveal_t)) * size * 1.8
            dy  -= drop
            alpha = int(alpha * min(1.0, reveal_t * 3))
        elif reveal_style == "unfold":      # Paper: scales up from 0
            idle_scale *= ease_elastic(reveal_t)
            alpha = int(alpha * min(1.0, reveal_t * 2.5))
        elif reveal_style == "slash":       # Scissors: slashes in from side
            dx  += (1 - ease_out(reveal_t)) * size * 2.2
            angle += (1 - reveal_t) * 45
            alpha = int(alpha * min(1.0, reveal_t * 2))
        else:                               # Generic pop
            idle_scale *= ease_elastic(reveal_t)
            alpha = int(alpha * min(1.0, reveal_t * 2.5))

    alpha = max(0, min(255, alpha))
    eff_size = max(4, int(size * idle_scale))
    final_cx = cx + int(dx)
    final_cy = cy + int(dy)

    # Draw to temp surface
    tmp = pygame.Surface((eff_size, eff_size), pygame.SRCALPHA)
    draw_gesture_icon(tmp, gesture, eff_size//2, eff_size//2, eff_size, color)

    # Rotate if needed
    if abs(angle) > 0.3:
        tmp = pygame.transform.rotate(tmp, -angle)

    tmp.set_alpha(alpha)
    surf.blit(tmp, (final_cx - tmp.get_width()//2,
                    final_cy - tmp.get_height()//2))

# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Anim:
    """Exponential-lerp animator. Set .target, call .update(dt) each frame."""
    def __init__(self, v=0.0, speed=8.0):
        self.v      = float(v)
        self.target = float(v)
        self.speed  = speed

    def update(self, dt):
        self.v += (self.target - self.v) * min(1.0, self.speed * dt)

    def set(self, target):
        self.target = float(target)

    def snap(self, v):
        self.v = float(v)
        self.target = float(v)

# ═══════════════════════════════════════════════════════════════════════════════
#  STAR FIELD  — drifting micro-dots on menu screens
# ═══════════════════════════════════════════════════════════════════════════════

class StarField:
    N = 60

    def __init__(self):
        self.stars = [self._mk() for _ in range(self.N)]

    def _mk(self, start_y=None):
        return {
            "x":  random.uniform(0, WIN_W),
            "y":  (random.uniform(0, WIN_H) if start_y is None else start_y),
            "vy": random.uniform(-0.12, -0.44),
            "r":  random.uniform(0.4, 1.7),
            "a":  random.randint(28, 85),
            "c":  random.choice([C_ACCENT, C_ACCENT2, C_WHITE, C_P1, C_P2]),
        }

    def update(self):
        for st in self.stars:
            st["y"] += st["vy"]
            if st["y"] < -4:
                st.update(self._mk(start_y=WIN_H + 4))

    def draw(self, surf):
        for st in self.stars:
            r = max(1, int(st["r"] + 0.5))
            s = _psurf(r)
            s.fill((0, 0, 0, 0))
            cr, cg, cb = st["c"]
            pygame.draw.circle(s, (cr, cg, cb, st["a"]), (r, r), r)
            surf.blit(s, (int(st["x"]) - r, int(st["y"]) - r))

# ── Slash trail system — reusable for Scissors effects ────────────────────────

class SlashTrail:
    """Maintains a short trail of slash segments that fade out."""
    def __init__(self, max_segs=6):
        self.segs = []   # each: {x1,y1,x2,y2, life, col, width}
        self.max  = max_segs

    def add(self, x1, y1, x2, y2, col, width=3):
        self.segs.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                           "life":1.0,"col":col,"width":width})
        if len(self.segs) > self.max:
            self.segs.pop(0)

    def update(self, dt, decay=3.5):
        for s in self.segs:
            s["life"] = max(0.0, s["life"] - decay * dt)
        self.segs = [s for s in self.segs if s["life"] > 0]

    def draw(self, surf):
        for s in self.segs:
            a = int(s["life"] ** 0.5 * 200)
            if a < 4: continue
            cr,cg,cb = s["col"]
            col_a = (min(255,cr+60), min(255,cg+60), min(255,cb+60), a)
            tmp = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.line(tmp, col_a,
                             (int(s["x1"]),int(s["y1"])),
                             (int(s["x2"]),int(s["y2"])), s["width"])
            surf.blit(tmp, (0,0))

# ═══════════════════════════════════════════════════════════════════════════════
#  STATIC PRE-RENDERED SURFACES  — built once at startup
# ═══════════════════════════════════════════════════════════════════════════════

class _Statics:

    def build_bg(self):
        s = pygame.Surface((WIN_W, WIN_H))
        s.fill(C_BG)
        # Fine grid
        gc = (13, 17, 26)
        for x in range(0, WIN_W, 40):
            pygame.draw.line(s, gc, (x, 0), (x, WIN_H))
        for y in range(0, WIN_H, 40):
            pygame.draw.line(s, gc, (0, y), (WIN_W, y))
        # Atmospheric colour blobs
        blob = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for rad in range(340, 0, -8):
            a = max(0, int(7 * (1 - rad / 340)))
            pygame.draw.circle(blob, (*C_ACCENT, a), (150, 72), rad)
        for rad in range(300, 0, -8):
            a = max(0, int(6 * (1 - rad / 300)))
            pygame.draw.circle(blob, (*C_ACCENT2, a), (WIN_W - 130, WIN_H - 72), rad)
        s.blit(blob, (0, 0))
        # Corner vignette
        vig = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for i in range(9):
            px, py = i * 22, i * 14
            pygame.draw.rect(vig, (0, 0, 0, 8 - i),
                             (px, py, WIN_W - px*2, WIN_H - py*2), width=24)
        s.blit(vig, (0, 0))
        return s

    def build_sb(self):
        """Hi-fi scoreboard chrome: glass bevel, accent border, tile dividers."""
        W = WIN_W - SB_PAD * 2
        s = pygame.Surface((WIN_W, SB_H + 6), pygame.SRCALPHA)
        pygame.draw.rect(s, (*C_PANEL, 255), (SB_PAD, 2, W, SB_H), border_radius=13)
        # Inner lighter strip (number zone)
        ih = int(SB_H * 0.60)
        inner = pygame.Surface((W, ih), pygame.SRCALPHA)
        pygame.draw.rect(inner, (255, 255, 255, 9), (0, 0, W, ih), border_radius=10)
        s.blit(inner, (SB_PAD, 2))
        # Glass bevel
        pygame.draw.line(s, (255, 255, 255, 32), (SB_PAD + 14, 3), (SB_PAD + W - 14, 3))
        # Accent border
        pygame.draw.rect(s, (*C_ACCENT, 200), (SB_PAD, 2, W, SB_H), width=1, border_radius=13)
        # Bottom accent hairline
        pygame.draw.line(s, (*C_ACCENT2, 65), (SB_PAD, SB_H + 2), (SB_PAD + W, SB_H + 2))
        # Tile dividers — aligned with new xf positions
        # Primary section dividers: after YOU, after DRAW, after AI; then secondary dividers
        for f in (0.230, 0.410, 0.580, 0.720, 0.855):
            dx = int(WIN_W * f)
            pygame.draw.line(s, (*C_DIM, 160), (dx, 10), (dx, SB_H - 4), 1)
        # Visual separator between primary and secondary score groups
        sep_x = int(WIN_W * 0.580)
        pygame.draw.line(s, (*C_ACCENT, 80), (sep_x, 8), (sep_x, SB_H - 4), 2)
        return s

    def build_cam_grad(self, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        for row in range(h):
            a = int((row / h) ** 0.7 * 200)
            pygame.draw.line(s, (0, 0, 0, a), (0, row), (w, row))
        return s

    def build_scanlines(self):
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for y in range(0, WIN_H, 4):
            pygame.draw.line(s, (0, 0, 0, 16), (0, y), (WIN_W, y))
        return s

    def build_veil(self):
        return pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)


_st = _Statics()

# ═══════════════════════════════════════════════════════════════════════════════
#  GAME STATES
# ═══════════════════════════════════════════════════════════════════════════════

class S:
    MENU      = "menu"
    MODE_SEL  = "mode_sel"
    TOURN_SEL = "tourn_sel"
    COUNTDOWN = "countdown"
    RESULT    = "result"
    CHAMPION  = "champion"

# ═══════════════════════════════════════════════════════════════════════════════
#  GAME
# ═══════════════════════════════════════════════════════════════════════════════

class Game:

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        pygame.init()
        audio.init()
        _warm_fonts()

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Rock · Paper · Scissors  |  Hand Gesture")
        self.clock  = pygame.time.Clock()

        self._load_fonts()

        # Static surfaces (built once, blitted every frame)
        self._bg        = _st.build_bg()
        self._sb        = _st.build_sb()
        self._veil      = _st.build_veil()
        self._cam_grad  = _st.build_cam_grad(CAM_DW, 64)
        self._scanlines = _st.build_scanlines()
        # Cache scoreboard scan beam (6×(SB_H-12)) — built once
        _sb_beam = pygame.Surface((6, SB_H - 12), pygame.SRCALPHA)
        pygame.draw.rect(_sb_beam, (*C_ACCENT, 14), (0, 0, 6, SB_H - 12), border_radius=3)
        self._sb_beam = _sb_beam
        # Cache camera scan line (4×CAM_DH) — built once, alpha varies via set_alpha
        _cam_scan = pygame.Surface((4, CAM_DH), pygame.SRCALPHA)
        pygame.draw.rect(_cam_scan, (*C_ACCENT, 255), (0, 0, 4, CAM_DH))
        self._cam_scan = _cam_scan
        # Pre-bake small gesture icon surfaces (46px) for live HUD overlays
        self._gicon_sm   = {g: _bake_gesture_surf(g, 46, GESTURE_COLOR[g])
                            for g in GESTURE_LIST}

        # Pre-rendered countdown glyphs at 128 px
        self._cd_surfs: Dict[str, pygame.Surface] = {}
        self._bake_cd_glyphs()

        self._model_path = ensure_model()

        # Camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            print(f"[RPS] WARNING: camera {CAM_INDEX} not found.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tracker = HandTracker(max_hands=2, model_path=self._model_path)
        self.ai      = AdaptiveAI()
        self.stars   = StarField()

        self.tourn_fmt  = 0
        self.tournament : Optional[Tournament] = None

        # (No background music — sound effects only)

        self.state  = S.MENU
        self.mode   = None
        self._reset_scores()

        self.cd_start  = 0.0
        self.cd_phase  = 3   # 3→2→1→0=GO; -1=resolved sentinel

        self.locked     : List[Optional[str]] = []
        self.res_txt    = ""
        self.res_col    = C_WHITE
        self.res_detail = ""
        self.res_start  = 0.0

        self.live_g    : List[Optional[str]] = []
        self.last_bgr  : Optional[np.ndarray] = None
        self.fps        = 0.0
        self.t          = 0.0
        self.dt         = 0.016

        self.parts : List[dict]    = []
        self.hover : Optional[str] = None
        self._btns : dict          = {}

        # ── Animators ─────────────────────────────────────────────────────────
        self._menu_in  = Anim(0.0, speed=4.5)   # menu slide-in 0→1
        self._menu_in.set(1.0)

        self._card_a   = [Anim(0.0, speed=9.0),   # result card slide-in
                          Anim(0.0, speed=9.0)]

        self._tile_fl  = {k: Anim(0.0, speed=5.0)  # scoreboard flash 1→0
                          for k in ("p1", "p2", "draw")}

        self._glabel_a = [Anim(0.0, speed=6.0),   # gesture label fade-in
                          Anim(0.0, speed=6.0)]

        self._cd_ring  = Anim(110.0, speed=3.0)   # countdown ring radius

        self._prev_live_g: List[Optional[str]] = [None, None]
        self._prev_status: List[str]           = ["SEARCHING", "SEARCHING"]

        # ── Character / gesture animation state ───────────────────────────────
        # Idle time accumulators — each gesture has its own clock
        self._idle_t   = {"Rock": 0.0, "Paper": 0.0, "Scissors": 0.0}
        # Reveal animation progress per result card slot (0=player, 1=AI)
        self._reveal_t = [0.0, 0.0]   # 0→1 over ~0.5s
        # AI avatar display state: "thinking" | "locked" | None
        self._ai_avatar_state  = None
        self._ai_avatar_gest   = None   # gesture AI is "considering"
        self._ai_cycle_t       = 0.0   # time inside current cycle
        # Slash trail for Scissors animation
        self._slash_trail      = SlashTrail(max_segs=8)
        # Screen flash (result reveal) — alpha 0→1→0
        self._screen_flash_a   = 0.0
        self._screen_flash_col = C_WHITE
        # Per-gesture idle energy crack positions (Rock)
        self._crack_offsets    = [(random.uniform(-30,30), random.uniform(-30,30))
                                  for _ in range(6)]
        # Energy wave rings for Paper — list of {r, life}
        self._paper_waves: List[dict] = []
        self._paper_wave_t = 0.0

    def _reset_scores(self):
        self.score  = {"p1": 0, "p2": 0, "draw": 0, "rounds": 0}
        self.streak = 0
        self.best   = 0

    def _load_fonts(self):
        self.f_title = _font(52)
        self.f_big   = _font(44)
        self.f_res   = _font(66)
        self.f_mid   = _font(32)
        self.f_sm    = _font(24)
        self.f_xs    = _font(18)
        self.f_2xs   = _font(14)
        self.f_num   = _font(44, mono=True)

    def _bake_cd_glyphs(self):
        f = _font(128)
        for label, col in [("3", C_ACCENT), ("2", C_ACCENT),
                            ("1", C_YELLOW), ("GO!", C_GREEN)]:
            self._cd_surfs[label] = f.render(label, True, col)

    # ── particles ─────────────────────────────────────────────────────────────

    def _burst(self, x, y, color, n=28):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            v = random.uniform(1.5, 7.5)
            self.parts.append({
                "x": float(x), "y": float(y),
                "vx": math.cos(a)*v, "vy": math.sin(a)*v,
                "life": 1.0, "decay": random.uniform(0.016, 0.042),
                "r": random.randint(2, 6), "c": color,
            })

    def _ring_burst(self, x, y, color, n=14, spd=3.5):
        for i in range(n):
            a = math.tau * i / n
            self.parts.append({
                "x": float(x), "y": float(y),
                "vx": math.cos(a)*spd, "vy": math.sin(a)*spd,
                "life": 1.0, "decay": 0.038, "r": 3, "c": color,
            })

    def _tick_parts(self):
        keep = []
        for p in self.parts:
            p["x"] += p["vx"];  p["y"] += p["vy"]
            p["vy"] += 0.13;    p["vx"] *= 0.97
            p["life"] -= p["decay"]
            if p["life"] > 0:
                keep.append(p)
        self.parts = keep

    def _draw_parts(self):
        for p in self.parts:
            r = p["r"]
            s = _psurf(r)
            s.fill((0, 0, 0, 0))
            cr, cg, cb = p["c"]
            pygame.draw.circle(s, (cr, cg, cb, int(p["life"] * 255)), (r, r), r)
            self.screen.blit(s, (int(p["x"]) - r, int(p["y"]) - r))

    # ── webcam ────────────────────────────────────────────────────────────────

    def _grab(self):
        ok, frame = self.cap.read()
        if ok:
            self.last_bgr = cv2.flip(frame, 1)

    def _process(self):
        if self.last_bgr is None:
            return
        rgb = cv2.cvtColor(self.last_bgr, cv2.COLOR_BGR2RGB)
        res, gestures = self.tracker.process(rgb)
        self.tracker.draw_hands(self.last_bgr, res)
        # Trigger label fade-in when gesture first confirmed
        for i, g in enumerate(gestures):
            if i < 2:
                if g != self._prev_live_g[i] and g is not None:
                    self._glabel_a[i].snap(0.0)
                    self._glabel_a[i].set(1.0)
                # Fire lock sound when status first reaches LOCK ACQUIRED
                new_status = self._tracking_status(i)
                if (new_status == "LOCK ACQUIRED" and
                        self._prev_status[i] != "LOCK ACQUIRED" and
                        self.state == S.COUNTDOWN):
                    audio.play("click_hi")
                self._prev_status[i] = new_status
                self._prev_live_g[i] = g
        self.live_g = gestures

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        prev = time.perf_counter()
        while True:
            now      = time.perf_counter()
            self.dt  = max(now - prev, 1e-6)
            prev     = now
            self.fps = 0.9 * self.fps + 0.1 / self.dt
            self.t  += self.dt

            self._grab()
            self._process()
            self._tick_parts()
            self.stars.update()
            self._slash_trail.update(self.dt)

            # Tick idle gesture clocks
            for g in GESTURE_LIST:
                self._idle_t[g] += self.dt

            # Tick reveal animations (advance toward 1.0 at different rates per slot)
            if self.state == S.RESULT:
                el_res = time.perf_counter() - self.res_start
                self._reveal_t[0] = min(1.0, el_res / 0.45)
                self._reveal_t[1] = min(1.0, max(0.0, el_res - 0.22) / 0.45)

            # Tick AI cycle clock (for changing gesture during thinking)
            if self._ai_avatar_state == "thinking":
                self._ai_cycle_t += self.dt
                if self._ai_cycle_t > 0.55:
                    self._ai_cycle_t = 0.0
                    self._ai_avatar_gest = random.choice(GESTURE_LIST)

            # Tick screen flash
            if self._screen_flash_a > 0:
                self._screen_flash_a = max(0.0, self._screen_flash_a - self.dt * 4.5)

            # Spawn Paper energy waves
            if self.state == S.MENU:
                self._paper_wave_t += self.dt
                if self._paper_wave_t > 1.2:
                    self._paper_wave_t = 0.0
                    self._paper_waves.append({"r": 30.0, "life": 1.0})
                for w in self._paper_waves:
                    w["r"]    += self.dt * 90
                    w["life"] -= self.dt * 1.1
                self._paper_waves = [w for w in self._paper_waves if w["life"] > 0]

            # Tick all animators
            self._menu_in.update(self.dt)
            for a in self._card_a:       a.update(self.dt)
            for a in self._tile_fl.values(): a.update(self.dt)
            for a in self._glabel_a:     a.update(self.dt)
            self._cd_ring.update(self.dt)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit()
                self._handle(ev)

            self._update()
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS_TARGET)

    def _quit(self):
        self.tracker.close()
        self.cap.release()
        pygame.quit()
        sys.exit()

    # ── events ────────────────────────────────────────────────────────────────

    def _handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            self._key(ev.key)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for name, rect in self._btns.items():
                if rect.collidepoint(ev.pos):
                    audio.play("select")
                    self._action(name)
                    break
        elif ev.type == pygame.MOUSEMOTION:
            new_hover = next(
                (n for n, r in self._btns.items() if r.collidepoint(ev.pos)), None)
            if new_hover != self.hover and new_hover is not None:
                audio.play("tick")   # subtle hover enter tick
            self.hover = new_hover

    def _key(self, k):
        if k == pygame.K_ESCAPE:
            if self.state in (S.COUNTDOWN, S.RESULT, S.CHAMPION):
                self._to_menu()
            elif self.state in (S.MODE_SEL, S.TOURN_SEL):
                self.state = S.MENU
            else:
                self._quit()
            return
        if k == pygame.K_r:
            self._to_menu()
            return

        if self.state == S.MENU:
            if k == pygame.K_1:   self.state = S.MODE_SEL
            elif k == pygame.K_2: self.state = S.TOURN_SEL

        elif self.state == S.MODE_SEL:
            if k == pygame.K_1:   self._start("single")
            elif k == pygame.K_2: self._start("two")

        elif self.state == S.TOURN_SEL:
            for i in range(len(Tournament.FORMATS)):
                if k == getattr(pygame, f"K_{i+1}", None):
                    self.tourn_fmt = i
                    self.state = S.MODE_SEL
                    break

        elif self.state == S.RESULT:
            if k == pygame.K_SPACE:
                self._next_round()

        elif self.state == S.CHAMPION:
            if k == pygame.K_SPACE:
                self._to_menu()

    def _action(self, name):
        dispatch = {
            "play":       lambda: setattr(self, "state", S.MODE_SEL),
            "tournament": lambda: setattr(self, "state", S.TOURN_SEL),
            "single":     lambda: self._start("single"),
            "two":        lambda: self._start("two"),
            "next":       self._next_round,
            "to_menu":    self._to_menu,
        }
        if name in dispatch:
            dispatch[name]()
            return
        if name.startswith("tf_"):
            self.tourn_fmt = int(name[3:])
            self.state = S.MODE_SEL

    # ── transitions ───────────────────────────────────────────────────────────

    def _to_menu(self):
        self.state      = S.MENU
        self.mode       = None
        self.tournament = None
        self._reset_scores()
        self.ai = AdaptiveAI()
        self.parts.clear()
        self._menu_in.snap(0.0)
        self._menu_in.set(1.0)

    def _start(self, mode):
        self.mode = mode
        self.tracker.close()
        self.tracker = HandTracker(
            max_hands=2 if mode == "two" else 1,
            model_path=self._model_path)
        self.live_g          = []
        self._prev_live_g    = [None, None]
        _, n = Tournament.FORMATS[self.tourn_fmt]
        self.tournament = Tournament(n) if n > 1 else None
        self._reset_scores()
        self._begin_cd()

    def _begin_cd(self):
        self.state    = S.COUNTDOWN
        self.cd_start = time.perf_counter()
        self.cd_phase = 3
        self.locked   = []
        audio.play("tick")
        self._cd_ring.snap(110.0)
        # Reset AI avatar to thinking mode
        self._ai_avatar_state = "thinking"
        self._ai_avatar_gest  = random.choice(GESTURE_LIST)
        self._ai_cycle_t      = 0.0
        self._reveal_t        = [0.0, 0.0]

    def _next_round(self):
        if self.tournament and self.tournament.over:
            self.state = S.CHAMPION
            audio.play("champ")
        else:
            self._begin_cd()

    # ── update ────────────────────────────────────────────────────────────────

    def _update(self):
        """Countdown 3→2→1→GO. Sentinel cd_phase==-1 prevents double-fire."""
        if self.state != S.COUNTDOWN or self.cd_phase == -1:
            return
        new = max(0, 3 - int((time.perf_counter() - self.cd_start) / COUNTDOWN_DUR))
        if new < self.cd_phase:
            self.cd_phase = new
            if new >= 1:
                audio.play("tick")
                self._cd_ring.snap(110.0)
            else:
                audio.play("reveal")
                self._resolve()
                self.cd_phase = -1   # sentinel

    # ── resolve ───────────────────────────────────────────────────────────────

    def _resolve(self):
        live = list(self.live_g)
        if self.mode == "single":
            p1 = live[0] if live else None
            ai = self.ai.choose()
            if p1:
                self.ai.record(p1)
            self.locked = [p1, ai]
            self._outcome(p1, ai, ("YOU", "AI"))
        else:
            p1 = live[0] if len(live) > 0 else None
            p2 = live[1] if len(live) > 1 else None
            self.locked = [p1, p2]
            self._outcome(p1, p2, ("P1", "P2"))
        self.state     = S.RESULT
        self.res_start = time.perf_counter()
        for a in self._card_a:
            a.snap(0.0)
            a.set(1.0)
        # Trigger reveal animations
        self._reveal_t = [0.0, 0.0]
        # Screen flash on reveal
        self._screen_flash_a   = 1.0
        self._screen_flash_col = (C_GREEN if (
            self.mode == "single" and len(self.locked) >= 2 and
            self.locked[0] and self.locked[1] and
            BEATS.get(self.locked[0]) == self.locked[1]) else C_WHITE)
        # Lock AI avatar
        self._ai_avatar_state = "locked"
        # Extra burst at player card position if player won
        if self.mode == "single" and len(self.locked) >= 2:
            p1g, aig = self.locked[0], self.locked[1]
            if p1g and aig and BEATS.get(p1g) == aig:
                self._burst(WIN_W//2 - 230, WIN_H//2 + 50, C_GREEN, 22)

    def _outcome(self, p1, p2, labels):
        self.score["rounds"] += 1
        l1, l2 = labels

        if p1 is None or p2 is None:
            who = l1 if p1 is None else l2
            self.res_txt    = f"{who}: No Hand"
            self.res_col    = C_GREY
            self.res_detail = ""
            return

        if p1 == p2:
            self.res_txt    = "DRAW"
            self.res_col    = C_DRAW
            self.res_detail = f"Both played {p1}"
            self.score["draw"] += 1
            self.streak = 0
            audio.play("draw")
            self._burst(WIN_W // 2, WIN_H // 2, C_DRAW)
            self._flash("draw", C_DRAW)
            if self.tournament:
                self.tournament.record("draw")

        elif BEATS.get(p1) == p2:
            verb = BEAT_VERB.get((p1, p2), "beats")
            col  = C_GREEN if l1 == "YOU" else C_P1
            self.res_txt    = "YOU WIN!" if l1 == "YOU" else f"{l1} WINS!"
            self.res_col    = col
            self.res_detail = f"{p1} {verb} {p2}"
            self.score["p1"] += 1
            self.streak += 1
            self.best = max(self.best, self.streak)
            if self.mode == "single":
                self.ai.on_player_win()
            audio.play("win")
            self._burst(WIN_W // 2, WIN_H // 2, col, 40)
            self._flash("p1", col)
            if self.tournament:
                self.tournament.record("p1")

        else:
            verb = BEAT_VERB.get((p2, p1), "beats")
            col  = C_RED if l2 == "AI" else C_P2
            self.res_txt    = f"{l2} WINS!"
            self.res_col    = col
            self.res_detail = f"{p2} {verb} {p1}"
            self.score["p2"] += 1
            self.streak = 0
            audio.play("lose" if l2 == "AI" else "win")
            self._burst(WIN_W // 2, WIN_H // 2, col)
            self._flash("p2", col)
            if self.tournament:
                self.tournament.record("p2")

    def _flash(self, key, color):
        """Flash a scoreboard tile and emit a small ring burst."""
        a = self._tile_fl[key]
        a.snap(1.0)
        a.set(0.0)
        # x-fractions match the primary tile positions in _draw_scoreboard
        tx = {"p1":   int(WIN_W * 0.140),
              "draw": int(WIN_W * 0.320),
              "p2":   int(WIN_W * 0.500)}.get(key, WIN_W // 2)
        self._ring_burst(tx, SB_H // 2 + 4, color, n=10, spd=3.0)

    # ═══════════════════════════════════════════════════════════════════════════
    #  DRAW DISPATCH
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw(self):
        self._btns = {}
        self.screen.blit(self._bg, (0, 0))

        if self.state in (S.MENU, S.MODE_SEL, S.TOURN_SEL):
            self.stars.draw(self.screen)

        self._draw_parts()

        if   self.state == S.MENU:      self._draw_menu()
        elif self.state == S.MODE_SEL:  self._draw_mode_sel()
        elif self.state == S.TOURN_SEL: self._draw_tourn_sel()
        elif self.state == S.COUNTDOWN: self._draw_game(); self._draw_countdown()
        elif self.state == S.RESULT:    self._draw_game(); self._draw_result()
        elif self.state == S.CHAMPION:  self._draw_champion()

        self.screen.blit(self._scanlines, (0, 0))

        # ── Screen flash overlay ──────────────────────────────────────────────
        if self._screen_flash_a > 0.01:
            fa = int(self._screen_flash_a ** 2 * 160)
            if fa > 2:
                cr2,cg2,cb2 = self._screen_flash_col
                fveil = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                fveil.fill((cr2, cg2, cb2, fa))
                self.screen.blit(fveil, (0,0))

        fps_s = self.f_2xs.render(f"FPS {self.fps:>3.0f}", True, C_DIM)
        self.screen.blit(fps_s, (WIN_W - fps_s.get_width() - 10, 6))

    # ── shared helpers ────────────────────────────────────────────────────────

    def _btn(self, label, name, x, y, w, h, col, fnt=None):
        f   = fnt or self.f_sm
        hov = self.hover == name
        # Body
        bg_fill = C_PANEL_LT if hov else C_PANEL
        panel(self.screen, bg_fill, (x, y, w, h), r=12, bw=2, bc=col)
        # Left accent bar
        bh2 = int(h * 0.52)
        by2 = y + (h - bh2) // 2
        pygame.draw.rect(self.screen, col, (x + 1, by2, 3, bh2), border_radius=2)
        # Hover outer ring
        if hov:
            pygame.draw.rect(self.screen, (*col, 32),
                             (x - 3, y - 3, w + 6, h + 6), width=2, border_radius=14)
        txt(self.screen, label, f, col, (x + w // 2, y + h // 2),
            glow=hov, glow_col=col, glow_r=4)
        r = pygame.Rect(x, y, w, h)
        self._btns[name] = r
        return r

    def _heading(self, text, y, col=C_ACCENT, fnt=None):
        f = fnt or self.f_title
        txt(self.screen, text, f, col, (WIN_W // 2, y),
            glow=True, glow_col=col, glow_r=6)

    def _hsep(self, y, hw=280, col=C_DIM):
        pygame.draw.line(self.screen, col,
                         (WIN_W // 2 - hw, y), (WIN_W // 2 + hw, y), 1)

    def _dot_div(self, y, n=7, col=C_DIM):
        """Row of dots — elegant section divider."""
        sp = 22
        ox = WIN_W // 2 - (n // 2) * sp
        for i in range(n):
            pygame.draw.circle(self.screen, col, (ox + i * sp, y), 2)

    # ═══════════════════════════════════════════════════════════════════════════
    #  MENU
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_menu(self):
        t   = self.t
        ain = ease_out(self._menu_in.v)

        # ── Decorative corner brackets (full-screen HUD feel) ─────────────────
        CL_M = 32
        brt_m = lerp_col(C_DIM, C_ACCENT, 0.25 * ain)
        for ox2, oy2, sx, sy in (
                (18, 18, 1, 1), (WIN_W - 18, 18, -1, 1),
                (18, WIN_H - 18, 1, -1), (WIN_W - 18, WIN_H - 18, -1, -1)):
            pygame.draw.line(self.screen, brt_m,
                             (ox2, oy2), (ox2 + CL_M * sx, oy2), 1)
            pygame.draw.line(self.screen, brt_m,
                             (ox2, oy2), (ox2, oy2 + CL_M * sy), 1)

        # Horizontal decorative rules — direct draw (no SRCALPHA allocation)
        if ain > 0.01:
            rule_col = lerp_col(C_BG, C_ACCENT, ain * 0.16)
            pygame.draw.line(self.screen, rule_col, (40, 36), (WIN_W - 40, 36))
            pygame.draw.line(self.screen, rule_col, (40, WIN_H - 36), (WIN_W - 40, WIN_H - 36))

        # ── Title block (slides down from above) ───────────────────────────────
        title_oy = int((1 - ain) * -60)

        # Glow halo behind title — direct lerp rect (no SRCALPHA allocation)
        if ain > 0.01:
            hw_h, hh_h = 560, 72
            halo_col = lerp_col(C_BG, C_ACCENT, ain * 0.11)
            pygame.draw.rect(self.screen, halo_col,
                             (WIN_W // 2 - hw_h // 2, 76 + title_oy - hh_h // 2,
                              hw_h, hh_h), border_radius=36)

        # Title text — large, bold, glowing
        title_f = _font(58)
        title   = "ROCK · PAPER · SCISSORS"
        ts = title_f.render(title, True, C_ACCENT)
        ts.set_alpha(int(ain * 255))
        self.screen.blit(ts, (WIN_W // 2 - ts.get_width() // 2,
                               76 + title_oy - ts.get_height() // 2))

        # Subtitle pill — direct lerp rect (no SRCALPHA allocation)
        badge_y = 76 + title_oy + 44
        bw2, bh2 = 270, 26
        pill_col = lerp_col(C_BG, C_ACCENT2, ain * 0.22)
        pygame.draw.rect(self.screen, pill_col,
                         (WIN_W // 2 - bw2 // 2, badge_y - bh2 // 2, bw2, bh2),
                         border_radius=13)
        pygame.draw.rect(self.screen, lerp_col(C_DIM, C_ACCENT2, ain * 0.5),
                         (WIN_W // 2 - bw2 // 2, badge_y - bh2 // 2, bw2, bh2),
                         width=1, border_radius=13)
        sub = self.f_xs.render("HAND GESTURE EDITION", True,
                                lerp_col(C_GREY, C_WHITE, 0.3))
        sub.set_alpha(int(ain * 255))
        self.screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2,
                                badge_y - sub.get_height() // 2))

        sep_y = badge_y + 28
        self._hsep(sep_y, hw=360, col=lerp_col(C_DIM, C_ACCENT, 0.2))

        # ── Gesture character showcase — animated fighters ─────────────────────
        card_cy  = sep_y + 82
        cw, ch   = 240, 220
        spacing  = 268
        MENU_ICON_SIZE = 88

        for i, g in enumerate(GESTURE_LIST):
            gc  = GESTURE_COLOR[g]
            ox  = WIN_W // 2 + (i - 1) * spacing

            idle_dx, idle_dy, idle_sc, idle_ang = gesture_idle(g, self._idle_t[g])
            bob        = int(idle_dy * 0.8)
            slide      = ease_out(max(0.0, min(1.0, ain * 3.2 - i * 0.6)))
            draw_y     = card_cy + bob + int((1 - slide) * 60)
            card_alpha = int(slide * 255)

            # ── Outer character aura ────────────────────────────────────────────
            if slide > 0.2:
                if g == "Rock":
                    slam = max(0.0, math.sin(self._idle_t[g] * 2.4) ** 9)
                    aura_r = int((cw//2 + 14) * (1.0 + 0.12 * slam))
                    aura_col = lerp_col(C_BG, gc, (0.10 + 0.18*slam) * slide)
                    pygame.draw.ellipse(self.screen, aura_col,
                                        (ox - aura_r, draw_y - ch//2 - 4,
                                         aura_r*2, int(ch*1.1)))
                    if slam > 0.5:
                        for ck_dx, ck_dy in self._crack_offsets[:4]:
                            c1x = ox + int(ck_dx * 0.3)
                            c1y = draw_y + int(ck_dy * 0.3) - ch//4
                            c2x = c1x + int(ck_dx * slam)
                            c2y = c1y + int(ck_dy * slam)
                            pygame.draw.line(self.screen,
                                             lerp_col(C_BG, gc, 0.35*slam*slide),
                                             (c1x, c1y), (c2x, c2y),
                                             max(1, int(slam * 2)))
                elif g == "Paper":
                    for w in self._paper_waves:
                        wr = int(w["r"] * 0.65)
                        if wr < 5: continue
                        wave_col = lerp_col(C_BG, gc, w["life"] * 0.18 * slide)
                        pygame.draw.circle(self.screen, wave_col,
                                           (ox, draw_y - ch//4), wr, 2)
                else:  # Scissors
                    snap = math.sin(self._idle_t[g] * 4.8) ** 3
                    if snap > 0.5 and slide > 0.4:
                        x1s = ox + int(idle_dx * 2)
                        y1s = draw_y - ch//3 + int(idle_dy)
                        self._slash_trail.add(x1s - 50, y1s - 25,
                                              x1s + 50, y1s + 25, gc, width=2)
                    self._slash_trail.draw(self.screen)

            # Pulsing border
            pulse_b  = 0.40 + 0.60 * ((math.sin(t * 1.85 + i * 0.92) + 1) / 2)
            border_c = lerp_col(C_PANEL_LT, gc, pulse_b * 0.80)

            # Drop shadow
            pygame.draw.rect(self.screen, lerp_col(C_BG, (0,0,0), 0.75),
                             (ox - cw//2 + 4, draw_y - ch//2 + 7, cw, ch),
                             border_radius=20)

            # Card body
            panel(self.screen, C_PANEL_MD, (ox - cw//2, draw_y - ch//2, cw, ch),
                  r=20, bw=2, bc=border_c)

            # Top accent strip — pulses with idle
            strip_bright = 0.2 + 0.4 * ((math.sin(self._idle_t[g] * 2.1) + 1) / 2)
            strip_col = lerp_col(gc, C_WHITE, strip_bright)
            pygame.draw.rect(self.screen, strip_col,
                             (ox - cw//2 + 16, draw_y - ch//2, cw - 32, 3),
                             border_radius=2)
            pygame.draw.rect(self.screen, lerp_col(strip_col, C_WHITE, 0.5),
                             (ox - cw//2 + 16, draw_y - ch//2, cw - 32, 1),
                             border_radius=2)

            if card_alpha > 10:
                icon_cy_pos = draw_y - ch//2 + 88
                icon_col    = lerp_col(gc, C_WHITE, 0.22)

                # Glow layer
                g_sz = int(MENU_ICON_SIZE * 1.3 * idle_sc)
                glow_str = 0.12 + 0.14 * ((math.sin(self._idle_t[g]*1.8+i)+1)/2)
                if g == "Rock":
                    slam_v = max(0.0, math.sin(self._idle_t[g]*2.4)**9)
                    glow_str += slam_v * 0.30
                glow_tmp = pygame.Surface((g_sz + 4, g_sz + 4), pygame.SRCALPHA)
                draw_gesture_icon(glow_tmp, g, (g_sz+4)//2, (g_sz+4)//2, g_sz, gc)
                glow_tmp.set_alpha(int(glow_str * 90 * slide))
                self.screen.blit(glow_tmp,
                                 (ox - (g_sz+4)//2 + int(idle_dx),
                                  icon_cy_pos - (g_sz+4)//2))

                # Main animated character icon
                draw_gesture_animated(self.screen, g,
                                      ox + int(idle_dx * 0.5), icon_cy_pos,
                                      MENU_ICON_SIZE, icon_col,
                                      self._idle_t[g], intensity=1.0,
                                      alpha=card_alpha)

                # Name
                name_col = lerp_col(gc, C_WHITE, 0.45)
                gs = _font(22).render(g, True, name_col)
                gs.set_alpha(card_alpha)
                self.screen.blit(gs, (ox - gs.get_width()//2, draw_y + 50))

                # Tagline
                taglines = {"Rock": "CRUSHING FORCE",
                            "Paper": "ELEGANT GRACE",
                            "Scissors": "RAZOR SPEED"}
                hs2 = self.f_2xs.render(taglines[g], True,
                                         lerp_col(C_DIM, gc, 0.45))
                hs2.set_alpha(int(card_alpha * 0.8))
                self.screen.blit(hs2, (ox - hs2.get_width()//2, draw_y + 72))

        self._dot_div(card_cy + ch // 2 + 16, n=9)

        # ── Menu buttons ───────────────────────────────────────────────────────
        bw, bh, gap = 370, 62, 12
        bx_final    = WIN_W // 2 - bw // 2
        specs = [
            ("1  ·  Quick Play",      "play",       C_ACCENT),
            ("2  ·  Tournament Mode", "tournament", C_YELLOW),
        ]
        for i, (lbl, name, col) in enumerate(specs):
            by     = card_cy + ch // 2 + 40 + i * (bh + gap)
            slide_b = ease_out(max(0.0, min(1.0, ain * 2.5 - 0.4 - i * 0.25)))
            bx_anim = int(bx_final + (1 - slide_b) * 140)
            btn_s = pygame.Surface((bw, bh), pygame.SRCALPHA)
            bg_c  = C_PANEL_LT if self.hover == name else C_PANEL
            pygame.draw.rect(btn_s, (*bg_c, 255), (0, 0, bw, bh), border_radius=14)
            pygame.draw.rect(btn_s, (*col, 255),  (0, 0, bw, bh), width=2, border_radius=14)
            # Left accent bar
            bar_h = int(bh * 0.52)
            bar_y2 = (bh - bar_h) // 2
            pygame.draw.rect(btn_s, (*col, 255), (1, bar_y2, 4, bar_h), border_radius=2)
            if self.hover == name:
                pygame.draw.rect(btn_s, (*col, 28),
                                 (-3, -3, bw + 6, bh + 6), width=2, border_radius=16)
            # Bevel highlight
            pygame.draw.line(btn_s, lerp_col(bg_c, C_WHITE, 0.12),
                             (14, 1), (bw - 14, 1))
            lbl_s = _font(22).render(lbl, True, col)
            btn_s.blit(lbl_s, (bw // 2 - lbl_s.get_width() // 2,
                                bh // 2 - lbl_s.get_height() // 2))
            btn_s.set_alpha(int(slide_b * 255))
            self.screen.blit(btn_s, (bx_anim, by))
            self._btns[name] = pygame.Rect(bx_anim, by, bw, bh)

        # Format badge
        fmt = Tournament.FORMATS[self.tourn_fmt][0]
        fmt_y = card_cy + ch // 2 + 40 + (bh + gap) * 2 + 38
        txt(self.screen, f"Format: {fmt}", self.f_xs,
            lerp_col(C_DIM, C_YELLOW, 0.3), (WIN_W // 2, fmt_y))

        # ── Mode badges row ────────────────────────────────────────────────────
        badge_specs = [
            ("VS AI",       C_ACCENT),
            ("2-PLAYER",    C_ACCENT2),
            ("TOURNAMENT",  C_YELLOW),
            ("HAND GESTURE", C_P1),
        ]
        badge_row_y = fmt_y + 22
        total_bw = sum(len(b[0]) * 8 + 22 for b in badge_specs) + 10 * (len(badge_specs)-1)
        bx_off = WIN_W // 2 - total_bw // 2
        for blbl, bcol in badge_specs:
            bpw = len(blbl) * 8 + 22
            bph = 18
            bs  = pygame.Surface((bpw, bph), pygame.SRCALPHA)
            cr_b, cg_b, cb_b = bcol
            pygame.draw.rect(bs, (cr_b, cg_b, cb_b, int(ain * 28)), (0, 0, bpw, bph), border_radius=9)
            pygame.draw.rect(bs, (cr_b, cg_b, cb_b, int(ain * 80)), (0, 0, bpw, bph), width=1, border_radius=9)
            bs_label = self.f_2xs.render(blbl, True, bcol)
            bs.blit(bs_label, (bpw//2 - bs_label.get_width()//2, bph//2 - bs_label.get_height()//2))
            bs.set_alpha(int(ain * 220))
            self.screen.blit(bs, (bx_off, badge_row_y))
            bx_off += bpw + 10

        # Footer
        self._hsep(WIN_H - 42, hw=360, col=lerp_col(C_DIM, C_ACCENT, 0.15))
        txt(self.screen, "1  Quick Play    ·    2  Tournament    ·    ESC  Quit    ·    R  Reset",
            self.f_2xs, C_DIM, (WIN_W // 2, WIN_H - 26))

    # ═══════════════════════════════════════════════════════════════════════════
    #  MODE SELECT
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_mode_sel(self):
        # Background panel for focus
        bg_s = pygame.Surface((500, 320), pygame.SRCALPHA)
        pygame.draw.rect(bg_s, (*C_PANEL, 200), (0, 0, 500, 320), border_radius=22)
        pygame.draw.rect(bg_s, (*C_ACCENT, 60), (0, 0, 500, 320), width=1, border_radius=22)
        self.screen.blit(bg_s, (WIN_W//2 - 250, WIN_H//2 - 176))

        self._heading("SELECT MODE", WIN_H // 2 - 138, col=C_ACCENT)
        self._hsep(WIN_H // 2 - 104, hw=200, col=lerp_col(C_DIM, C_ACCENT, 0.3))
        txt(self.screen, f"Format:  {Tournament.FORMATS[self.tourn_fmt][0]}",
            self.f_xs, lerp_col(C_GREY, C_YELLOW, 0.4), (WIN_W // 2, WIN_H // 2 - 80))
        bw, bh = 360, 66
        bx = WIN_W // 2 - bw // 2
        self._btn("1  ·  VS  AI",         "single", bx, WIN_H//2 - 40, bw, bh, C_ACCENT)
        self._btn("2  ·  Local 2-Player", "two",    bx, WIN_H//2 + bh + 8, bw, bh, C_ACCENT2)
        self._hsep(WIN_H // 2 + bh * 2 + 44, hw=200, col=C_DIM)
        txt(self.screen, "ESC  back", self.f_2xs, C_DIM,
            (WIN_W // 2, WIN_H // 2 + bh * 2 + 60))

    # ═══════════════════════════════════════════════════════════════════════════
    #  TOURNAMENT SELECT
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_tourn_sel(self):
        # Background panel
        bg_s = pygame.Surface((440, 480), pygame.SRCALPHA)
        pygame.draw.rect(bg_s, (*C_PANEL, 210), (0, 0, 440, 480), border_radius=22)
        pygame.draw.rect(bg_s, (*C_YELLOW, 50), (0, 0, 440, 480), width=1, border_radius=22)
        self.screen.blit(bg_s, (WIN_W//2 - 220, 88))

        self._heading("TOURNAMENT FORMAT", 118, col=C_YELLOW)
        self._hsep(152, hw=220, col=lerp_col(C_DIM, C_YELLOW, 0.25))
        bw, bh, gap = 380, 54, 10
        bx = WIN_W // 2 - bw // 2
        for i, (name, _) in enumerate(Tournament.FORMATS):
            by  = 172 + i * (bh + gap)
            sel = (i == self.tourn_fmt)
            col = C_YELLOW if sel else C_GREY
            self._btn(f"{i+1}  ·  {name}" + ("  ✓" if sel else ""),
                      f"tf_{i}", bx, by, bw, bh, col)
        bot = 172 + len(Tournament.FORMATS) * (bh + gap)
        self._hsep(bot + 10, hw=220, col=C_DIM)
        txt(self.screen, "ESC  back", self.f_2xs, C_DIM, (WIN_W // 2, bot + 28))

    # ═══════════════════════════════════════════════════════════════════════════
    #  GAME BASE  (camera + scoreboard + HUD)
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_game(self):
        self._draw_scoreboard()
        cx, cy, cw, ch = CAM_X, CAM_Y, CAM_DW, CAM_DH

        if self.last_bgr is not None:
            self.screen.blit(cam_to_surf(self.last_bgr, cw, ch), (cx, cy))
            self.screen.blit(self._cam_grad, (cx, cy + ch - 64))
        else:
            panel(self.screen, C_PANEL_MD, (cx, cy, cw, ch), r=8)
            txt(self.screen, "Camera not found — check CAM_INDEX",
                self.f_sm, C_GREY, (cx + cw // 2, cy + ch // 2))

        # ── Premium camera frame ───────────────────────────────────────────────
        # Outer atmosphere ring — direct rect, no per-frame surface
        atm_col = lerp_col(C_BG, C_ACCENT2,
                           0.06 + 0.03 * math.sin(self.t * 1.4))
        pygame.draw.rect(self.screen, atm_col,
                         (cx - 8, cy - 8, cw + 16, ch + 16),
                         width=8, border_radius=18)

        # Main accent border (teal, 2px, rounded)
        pygame.draw.rect(self.screen, C_ACCENT,
                         (cx-2, cy-2, cw+4, ch+4), width=2, border_radius=10)
        # Inner highlight line
        pygame.draw.rect(self.screen, lerp_col(C_ACCENT, C_WHITE, 0.25),
                         (cx+1, cy+1, cw-2, 1))

        # ── Corner bracket accents ─────────────────────────────────────────────
        CL   = 28
        CL2  = 14
        ext  = int(1.5 + 1.5 * math.sin(self.t * 2.0))   # gentle breathing
        brt  = lerp_col(C_YELLOW, C_WHITE, 0.25 + 0.25 * math.sin(self.t * 1.6))
        for ox2, oy2, sx, sy in (
                (cx - 2, cy - 2,  1,  1),
                (cx + cw - CL + 2, cy - 2, -1,  1),
                (cx - 2, cy + ch - CL + 2,  1, -1),
                (cx + cw - CL + 2, cy + ch - CL + 2, -1, -1)):
            # Outer L (longer)
            pygame.draw.line(self.screen, brt, (ox2, oy2), (ox2 + CL*sx, oy2), 2)
            pygame.draw.line(self.screen, brt, (ox2, oy2), (ox2, oy2 + (CL+ext)*sy), 2)
            # Inner dot accent
            pygame.draw.circle(self.screen, C_ACCENT, (ox2 + 6*sx, oy2 + 6*sy), 2)

        # ── HUD labels in frame corners ────────────────────────────────────────
        hud_col = lerp_col(C_DIM, C_ACCENT, 0.35)
        hud_f   = self.f_2xs
        # Top-left: VISION
        hvs = hud_f.render("● VISION ACTIVE", True, hud_col)
        self.screen.blit(hvs, (cx + 10, cy + 8))
        # Top-right: resolution tag
        hres = hud_f.render(f"{CAM_W}×{CAM_H}", True, hud_col)
        self.screen.blit(hres, (cx + cw - hres.get_width() - 10, cy + 8))

        # Animated scan line (uses pre-cached surface)
        scan_x = int((self.t * 0.22 % 1.0) * cw)
        scan_a = 10 + int(4 * math.sin(self.t * 3.0))
        self._cam_scan.set_alpha(scan_a)
        self.screen.blit(self._cam_scan, (cx + scan_x - 2, cy))

        # ── Gesture detection HUD ──────────────────────────────────────────────
        ly = cy + ch - 44
        if self.mode == "single":
            g   = self.live_g[0] if self.live_g else None
            gc  = g_color(g, C_P1)
            lbl = g if g else "— show your hand —"
            alpha = int(self._glabel_a[0].v * 255) if g else 120

            # Live gesture icon (small, right of camera) — set_alpha in-place, no copy()
            if g and g in self._gicon_sm:
                icon_x = cx + cw - 54
                icon_y = ly
                self._gicon_sm[g].set_alpha(alpha)
                self.screen.blit(self._gicon_sm[g], (icon_x - 23, icon_y - 23))
                self._gicon_sm[g].set_alpha(255)

            # Detection pill — direct draw (no SRCALPHA allocation)
            pill_s = self.f_sm.render(lbl, True, gc)
            pw, ph = pill_s.get_width() + 28, pill_s.get_height() + 8
            pill_x = cx + cw//2 - pw//2
            pill_y = ly - ph//2
            if g:
                pill_bg_col = lerp_col(C_BG, gc, 0.12)
                pill_bdr    = lerp_col(C_DIM, gc, 0.32)
            else:
                pill_bg_col = lerp_col(C_BG, C_DIM, 0.3)
                pill_bdr    = C_DIM
            pygame.draw.rect(self.screen, pill_bg_col,
                             (pill_x, pill_y, pw, ph), border_radius=ph//2)
            pygame.draw.rect(self.screen, pill_bdr,
                             (pill_x, pill_y, pw, ph), width=1, border_radius=ph//2)
            pill_s.set_alpha(alpha)
            self.screen.blit(pill_s, (cx + cw//2 - pill_s.get_width()//2,
                                      ly - pill_s.get_height()//2))

            # Tracking status label — futuristic scanner HUD
            status     = self._tracking_status(0)
            status_col = {
                "LOCK ACQUIRED": C_GREEN,
                "LOCKING TARGET": C_YELLOW,
                "HAND DETECTED":  C_ACCENT,
                "SEARCHING":      C_GREY,
            }.get(status, C_GREY)
            # Animated blink dot for SEARCHING
            dot = ""
            if status == "SEARCHING":
                dot = " " + ("·" * (int(self.t * 2) % 4))
            elif status == "LOCKING TARGET":
                dot = " " + ("▪" * (int(self.t * 3) % 3 + 1))
            status_label = status + dot
            status_s = _font(15).render(status_label, True, status_col)
            sx_pos = cx + cw - status_s.get_width() - 14
            sy_pos = cy + ch - 52
            # Subtle background pill
            spl_w = status_s.get_width() + 16
            spl_h = status_s.get_height() + 6
            spl_col = lerp_col(C_BG, status_col, 0.08)
            pygame.draw.rect(self.screen, spl_col,
                             (sx_pos - 8, sy_pos - 3, spl_w, spl_h),
                             border_radius=5)
            pygame.draw.rect(self.screen, lerp_col(C_DIM, status_col, 0.35),
                             (sx_pos - 8, sy_pos - 3, spl_w, spl_h),
                             width=1, border_radius=5)
            self.screen.blit(status_s, (sx_pos, sy_pos))

            # Dual confidence panel — left side of camera bottom
            hand_conf  = self.tracker.conf_smooth[0]
            gest_conf  = min(1.0, self.tracker.conf_smooth[0] * (1.15 if g else 0.0))
            self._conf_panel(hand_conf, gest_conf, cx + 112, cy + ch - 32)
        else:
            for i, (pfx, col) in enumerate([("P1", C_P1), ("P2", C_P2)]):
                g   = self.live_g[i] if i < len(self.live_g) else None
                gc  = g_color(g, col)
                lbl = f"{pfx}:  {g or 'no hand'}"
                ox  = cx + cw // 4 + i * (cw // 2)
                alpha = int(self._glabel_a[i].v * 255) if g else 120
                # Small live gesture icon — set_alpha in-place, no .copy()
                if g and g in self._gicon_sm:
                    self._gicon_sm[g].set_alpha(alpha)
                    self.screen.blit(self._gicon_sm[g], (ox - 23, ly - 54))
                    self._gicon_sm[g].set_alpha(255)
                pill_s = self.f_sm.render(lbl, True, gc)
                pill_s.set_alpha(alpha)
                self.screen.blit(pill_s, (ox - pill_s.get_width()//2,
                                          ly - pill_s.get_height()//2))
                conf_v = self.tracker.conf_smooth[i]
                self._conf_bar(conf_v, ox - 41, ly + 14)
                pct_s = self.f_2xs.render(f"{int(conf_v*100)}%",
                                          True, conf_color(conf_v))
                self.screen.blit(pct_s, (ox + 44, ly + 11))
                # Per-player status
                st   = self._tracking_status(i)
                stc  = {
                    "LOCK ACQUIRED": C_GREEN,
                    "LOCKING TARGET": C_YELLOW,
                    "HAND DETECTED":  C_ACCENT,
                    "SEARCHING":      C_GREY,
                }.get(st, C_GREY)
                sts  = self.f_2xs.render(st, True, stc)
                self.screen.blit(sts, (ox - sts.get_width()//2, ly + 26))

        if self.tournament:
            self._tourn_pips(cy + ch + 10)

    def _tracking_status(self, slot=0):
        """Return a futuristic scanner state string based on smoothed confidence."""
        c = self.tracker.conf_smooth[slot] if slot < len(self.tracker.conf_smooth) else 0.0
        g = self.live_g[slot] if slot < len(self.live_g) else None
        if c < 0.15:
            return "SEARCHING"
        if c < 0.45:
            return "HAND DETECTED"
        if g is None:
            return "LOCKING TARGET"
        if c < 0.80:
            return "LOCKING TARGET"
        return "LOCK ACQUIRED"

    def _conf_bar(self, conf, x, y):
        """Compact confidence bar for 2-player layout."""
        bw, bh = 82, 5
        pygame.draw.rect(self.screen, C_DIM, (x, y, bw, bh), border_radius=2)
        fw = max(0, int(bw * conf))
        if fw > 0:
            pygame.draw.rect(self.screen, conf_color(conf),
                             (x, y, fw, bh), border_radius=2)

    def _conf_panel(self, hand_conf, gest_conf, cx, cy, label=""):
        """Futuristic scanner confidence display."""
        PW, PH = 210, 66
        px     = cx - PW // 2
        py     = cy - PH // 2

        # Panel background
        panel_col = lerp_col(C_BG, C_PANEL_MD, 0.9)
        pygame.draw.rect(self.screen, panel_col,
                         (px, py, PW, PH), border_radius=8)
        # Border — accent when confident, dim when not
        conf_level = max(hand_conf, gest_conf)
        border_col = lerp_col(C_DIM, C_ACCENT, conf_level * 0.6)
        pygame.draw.rect(self.screen, border_col,
                         (px, py, PW, PH), width=1, border_radius=8)
        # Top bevel
        pygame.draw.line(self.screen,
                         lerp_col(panel_col, C_WHITE, 0.08),
                         (px + 8, py + 1), (px + PW - 8, py + 1))

        # Header label
        hdr = self.f_2xs.render("■  SCANNER", True,
                                 lerp_col(C_DIM, C_ACCENT, 0.55))
        self.screen.blit(hdr, (px + 8, py + 5))

        # Separator
        pygame.draw.line(self.screen, lerp_col(C_BG, C_ACCENT, 0.18),
                         (px + 8, py + 19), (px + PW - 8, py + 19))

        BAR_W = 100
        BAR_H = 4

        def _row(label_txt, conf, ry):
            col = conf_color(conf)
            # Row label
            ls = self.f_2xs.render(label_txt, True,
                                    lerp_col(C_DIM, C_GREY, 0.7))
            self.screen.blit(ls, (px + 8, ry))
            # Percentage value
            pct_s = self.f_2xs.render(f"{int(conf * 100):>3}%", True, col)
            bar_x = px + PW - BAR_W - 8
            self.screen.blit(pct_s,
                             (bar_x - pct_s.get_width() - 6, ry))
            # Bar track
            pygame.draw.rect(self.screen,
                             lerp_col(C_BG, C_PANEL_LT, 0.4),
                             (bar_x, ry + 3, BAR_W, BAR_H), border_radius=2)
            # Bar fill
            fw = max(0, int(BAR_W * conf))
            if fw:
                pygame.draw.rect(self.screen, col,
                                 (bar_x, ry + 3, fw, BAR_H), border_radius=2)
                # Shine
                pygame.draw.rect(self.screen,
                                 lerp_col(col, C_WHITE, 0.5),
                                 (bar_x, ry + 3, fw, max(1, BAR_H // 2)),
                                 border_radius=2)
            # Tick marks at 25/50/75
            for tick_f in (0.25, 0.5, 0.75):
                tx2 = bar_x + int(BAR_W * tick_f)
                pygame.draw.line(self.screen,
                                 lerp_col(C_BG, C_DIM, 0.6),
                                 (tx2, ry + 2), (tx2, ry + BAR_H + 4))

        _row("HAND", hand_conf, py + 24)
        _row("LOCK", gest_conf, py + 44)

    def _tourn_pips(self, y):
        if not self.tournament:
            return
        needed = self.tournament.needed
        pw, gap = 12, 5
        cx = WIN_W // 2
        for side, (wins, col) in enumerate([
                (self.tournament.p1_wins, C_P1),
                (self.tournament.p2_wins, C_P2)]):
            for pip in range(needed):
                if side == 0:
                    px = cx - 28 - (needed - pip) * (pw + gap) + gap
                else:
                    px = cx + 28 + pip * (pw + gap)
                filled = pip < wins
                pygame.draw.rect(self.screen, col if filled else C_DIM,
                                 (px, y, pw, 8), border_radius=3)
                if filled:
                    pygame.draw.rect(self.screen,
                                     lerp_col(col, C_WHITE, 0.35),
                                     (px, y, pw, 3), border_radius=2)
        txt(self.screen, self.tournament.score_str,
            self.f_2xs, C_GREY, (cx, y + 4))

    # ═══════════════════════════════════════════════════════════════════════════
    #  SCOREBOARD
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_scoreboard(self):
        self.screen.blit(self._sb, (0, 0))

        p1_lbl = "YOU" if self.mode == "single" else "P1"
        p2_lbl = "AI"  if self.mode == "single" else "P2"
        p2_col = C_RED if self.mode == "single" else C_P2

        SB_Y = 2

        # ── Primary score tiles ────────────────────────────────────────────────
        primary = [
            ("p1",   p1_lbl,  str(self.score["p1"]),   C_P1,    0.140),
            ("draw", "DRAW",  str(self.score["draw"]), C_DRAW,  0.320),
            ("p2",   p2_lbl,  str(self.score["p2"]),   p2_col,  0.500),
        ]
        secondary = [
            (None, "ROUND",  str(self.score["rounds"]), C_GREY,   0.650),
            (None, "STREAK", str(self.streak),          C_YELLOW, 0.790),
            (None, "BEST",   str(self.best),            C_ACCENT, 0.920),
        ]

        for flash_key, lbl, val, col, xf in primary:
            tx = int(WIN_W * xf)

            if flash_key:
                fv = self._tile_fl[flash_key].v
                if fv > 0.01:
                    # Direct lerp-colour rects — no SRCALPHA surface allocation
                    fw_out = int(WIN_W * 0.16)
                    fw_in  = int(WIN_W * 0.09)
                    fh     = SB_H - 4
                    col_out = lerp_col(C_PANEL, col, fv * 0.15)
                    col_in  = lerp_col(C_PANEL, col, fv * 0.32)
                    pygame.draw.rect(self.screen, col_out,
                                     (tx - fw_out//2, SB_Y + 2, fw_out, fh),
                                     border_radius=12)
                    pygame.draw.rect(self.screen, col_in,
                                     (tx - fw_in//2, SB_Y + 2, fw_in, fh),
                                     border_radius=8)

            # Score number — with pop scale on flash
            fv2 = self._tile_fl[flash_key].v if flash_key else 0.0
            num_scale = 1.0 + fv2 * 0.18
            base_num  = self.f_num.render(val, True, col)
            if num_scale > 1.02:
                nw = max(1, int(base_num.get_width()  * num_scale))
                nh = max(1, int(base_num.get_height() * num_scale))
                scaled_num = pygame.transform.smoothscale(base_num, (nw, nh))
            else:
                scaled_num = base_num
            self.screen.blit(scaled_num,
                             (tx - scaled_num.get_width()//2,
                              SB_Y + 24 - scaled_num.get_height()//2))
            # Label
            txt(self.screen, lbl, self.f_xs, lerp_col(C_GREY, col, 0.40),
                (tx, SB_Y + 56))

        # ── Secondary stats ────────────────────────────────────────────────────
        num_sm = _font(30, mono=True)
        for _, lbl, val, col, xf in secondary:
            tx = int(WIN_W * xf)
            txt(self.screen, val, num_sm, lerp_col(C_DIM, col, 0.75),
                (tx, SB_Y + 26))
            txt(self.screen, lbl, self.f_2xs, C_DIM, (tx, SB_Y + 52))

        # ── Animated scan beam (uses pre-cached surface) ──────────────────────
        sx = int((self.t * 0.14 % 1.0) * WIN_W)
        self.screen.blit(self._sb_beam, (sx, SB_Y + 6))

        # ── AI difficulty + combat class badge ─────────────────────────────────
        if self.mode == "single":
            pct      = int((1 - self.ai.rand_prob) * 100)
            diff_col = lerp_col(C_GREEN, C_RED, pct / 100)
            badge_txt = f"AI  {pct}%"
            txt(self.screen, badge_txt, self.f_2xs,
                lerp_col(C_DIM, diff_col, 0.65), (WIN_W - 40, SB_Y + 34))

    # ═══════════════════════════════════════════════════════════════════════════
    #  COUNTDOWN
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_countdown(self):
        phase = max(0, self.cd_phase)
        el    = time.perf_counter() - self.cd_start
        frac  = min(1.0, (el % COUNTDOWN_DUR) / COUNTDOWN_DUR)

        label = str(phase) if phase > 0 else "GO!"
        col   = (C_YELLOW if phase == 1 else C_ACCENT) if phase > 0 else C_GREEN

        # Scale: big pop on each new number, settle fast
        scale = 1.0 + 0.55 * max(0.0, 1.0 - frac * 2.2)
        alpha = int(255 * min(1.0, 1.1 - frac * 0.95))

        base  = self._cd_surfs[label]
        bw0, bh0 = base.get_size()
        glyph = (pygame.transform.smoothscale(
                     base, (max(1, int(bw0*scale)), max(1, int(bh0*scale))))
                 if scale > 1.01 else base)
        gw, gh = glyph.get_size()

        # ── Background halo that pulses with number ────────────────────────────
        halo_a  = 0.10 * (1 - frac * 0.7) * scale
        halo_col = lerp_col(C_BG, col, halo_a)
        hr = int(90 * scale)
        pygame.draw.circle(self.screen, halo_col, (WIN_W//2, WIN_H//2), hr)

        # Secondary outer ring — contracting inward on each beat
        self._cd_ring.set(max(36, int(60 + 50 * (1 - frac))))
        rr = int(self._cd_ring.v)
        ring_a_f = 0.55 * (1 - frac ** 0.5)
        if ring_a_f > 0 and rr > 2:
            ring_col = lerp_col(C_BG, col, ring_a_f)
            pygame.draw.circle(self.screen, ring_col, (WIN_W//2, WIN_H//2), rr, 3)
            # Outer thin ring for depth
            pygame.draw.circle(self.screen,
                               lerp_col(C_BG, col, ring_a_f * 0.3),
                               (WIN_W//2, WIN_H//2), rr + 12, 1)

        # ── Number glyph ───────────────────────────────────────────────────────
        glyph_alpha = max(0, min(255, alpha))
        if glyph_alpha < 255:
            tmp = glyph.copy()
            tmp.set_alpha(glyph_alpha)
            self.screen.blit(tmp, (WIN_W//2 - gw//2, WIN_H//2 - gh//2))
        else:
            self.screen.blit(glyph, (WIN_W//2 - gw//2, WIN_H//2 - gh//2))

        # ── Tracking status pill ───────────────────────────────────────────────
        status = self._tracking_status(0)
        status_col = {
            "LOCK ACQUIRED": C_GREEN, "LOCKING TARGET": C_YELLOW,
            "HAND DETECTED": C_ACCENT, "SEARCHING": C_GREY,
        }.get(status, C_GREY)
        hint = status if phase > 0 else "LOCK ACQUIRED"
        hint_col = status_col if phase > 0 else C_GREEN
        txt(self.screen, hint, self.f_sm, hint_col, (WIN_W//2, WIN_H//2 + 102))

        # ── AI opponent panel (single mode only) ──────────────────────────────
        if self.mode == "single":
            AI_CX = WIN_W//2
            AI_CY = WIN_H//2 + 158
            AI_PW, AI_PH = 320, 82

            # Panel background
            panel_col2 = lerp_col(C_BG, C_PANEL_MD, 0.92)
            pygame.draw.rect(self.screen, panel_col2,
                             (AI_CX - AI_PW//2, AI_CY - AI_PH//2, AI_PW, AI_PH),
                             border_radius=14)
            pygame.draw.rect(self.screen, lerp_col(C_DIM, C_P2, 0.3),
                             (AI_CX - AI_PW//2, AI_CY - AI_PH//2, AI_PW, AI_PH),
                             width=1, border_radius=14)

            # ── AI state text  ─────────────────────────────────────────────────
            if phase > 0:
                ai_states = ["ANALYZING...", "SCANNING...", "PROCESSING..."]
                ai_state  = ai_states[int(self.t * 1.4) % len(ai_states)]
                pulse_c   = lerp_col(C_GREY, C_P2,
                                      0.5 + 0.5 * math.sin(self.t * 3.8))
                ai_s = self.f_xs.render(ai_state, True, pulse_c)
                ai_s.set_alpha(int(200 + 55 * math.sin(self.t * 5.0)))
                self.screen.blit(ai_s, (AI_CX - ai_s.get_width()//2,
                                        AI_CY - ai_s.get_height() - 4))

                # ── Flickering AI gesture avatar (cycling) ─────────────────────
                if self._ai_avatar_gest:
                    av_g   = self._ai_avatar_gest
                    av_col = GESTURE_COLOR.get(av_g, C_P2)
                    av_sz  = 46
                    # Flicker alpha — rapid on/off to simulate "thinking"
                    flicker = 0.5 + 0.5 * math.sin(self.t * 18.0 + random.uniform(-0.1,0.1))
                    av_alpha = int(100 * flicker + 60)
                    av_tmp   = pygame.Surface((av_sz, av_sz), pygame.SRCALPHA)
                    draw_gesture_icon(av_tmp, av_g, av_sz//2, av_sz//2, av_sz,
                                      lerp_col(C_DIM, av_col, 0.4))
                    av_tmp.set_alpha(av_alpha)
                    self.screen.blit(av_tmp,
                                     (AI_CX + AI_PW//2 - av_sz - 12,
                                      AI_CY - av_sz//2 + 10))

                # "AI OPPONENT" label
                ai_lbl = self.f_2xs.render("AI  OPPONENT", True,
                                            lerp_col(C_DIM, C_P2, 0.4))
                self.screen.blit(ai_lbl,
                                 (AI_CX - AI_PW//2 + 12,
                                  AI_CY + 6))
            else:
                # Phase 0 — GO! — AI has locked in
                go_col = lerp_col(C_P2, C_WHITE, 0.4)
                lock_s = self.f_xs.render("LOCKED IN!", True, go_col)
                self.screen.blit(lock_s,
                                 (AI_CX - lock_s.get_width()//2, AI_CY - 12))

    # ═══════════════════════════════════════════════════════════════════════════
    #  RESULT
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_result(self):
        el = time.perf_counter() - self.res_start

        # ── Veil (slightly darker for more cinematic feel) ─────────────────────
        va = min(190, int(el * 340))
        self._veil.fill((6, 8, 16, va))
        self.screen.blit(self._veil, (0, 0))

        # ── Determine winner for card highlighting ─────────────────────────────
        g1 = self.locked[0] if len(self.locked) > 0 else None
        g2 = self.locked[1] if len(self.locked) > 1 else None
        l1 = "YOU" if self.mode == "single" else "P1"
        l2 = "AI"  if self.mode == "single" else "P2"
        c1 = C_P1
        c2 = C_RED if self.mode == "single" else C_P2

        # Winner: 0=p1 wins, 1=p2 wins, -1=draw/no-hand
        if g1 and g2 and g1 != g2:
            winner_idx = 0 if BEATS.get(g1) == g2 else 1
        else:
            winner_idx = -1

        # ── Result headline: pop-in then gentle pulse ──────────────────────────
        if el < 0.18:
            pop = 1.0 + (0.18 - el) * 2.2
        elif el < 0.5:
            pop = 1.0
        else:
            pop = 1.0 + 0.018 * math.sin(el * 5.2)
        pop = max(0.85, min(pop, 1.55))

        res_font = _font(72)
        base_r   = res_font.render(self.res_txt, True, self.res_col)
        RES_CY   = 158

        # Glow plate — direct lerp colour draw, no SRCALPHA surface per frame
        gplate_a = max(0.0, min(1.0, el * 5))
        if gplate_a > 0:
            gp_col = lerp_col(C_BG, self.res_col, gplate_a * 0.15)
            rw_base = base_r.get_width()
            pygame.draw.rect(self.screen, gp_col,
                             (WIN_W//2 - (rw_base+80)//2, RES_CY - 36,
                              rw_base + 80, 72), border_radius=36)

        if abs(pop - 1.0) > 0.01:
            rw = max(1, int(base_r.get_width()  * pop))
            rh = max(1, int(base_r.get_height() * pop))
            res_surf = pygame.transform.smoothscale(base_r, (rw, rh))
        else:
            res_surf = base_r
        self.screen.blit(res_surf,
                         (WIN_W//2 - res_surf.get_width()//2,
                          RES_CY - res_surf.get_height()//2))

        if self.res_detail:
            detail_a = min(255, int((el - 0.15) * 600))
            if detail_a > 0:
                ds = self.f_sm.render(self.res_detail, True, C_GREY)
                ds.set_alpha(detail_a)
                self.screen.blit(ds, (WIN_W//2 - ds.get_width()//2, RES_CY + 46))

        # ── Large gesture cards ────────────────────────────────────────────────
        # Card 0 (YOU/P1) slides from LEFT, Card 1 (AI/P2) slides from RIGHT
        # AI card has a dramatic delay for reveal effect
        AI_DELAY = 0.22   # seconds

        CW, CH   = 310, 320   # tall cards — gesture is the hero
        CARD_TOP = 220
        HALF_GAP = 250
        ICON_SIZE = 130       # large vector icon
        NAME_FONT = _font(28)
        LBL_FONT  = _font(15)

        for i, (gest, lbl, col) in enumerate([(g1, l1, c1), (g2, l2, c2)]):
            card_delay = AI_DELAY if i == 1 else 0.0
            eff_el     = max(0.0, el - card_delay)
            slide      = ease_out(min(1.0, eff_el * 6.0))
            side_dir   = -1 if i == 0 else 1
            cx         = WIN_W//2 + (-HALF_GAP + i * HALF_GAP * 2) - CW//2
            cx_a       = int(cx + (1 - slide) * side_dir * 320)

            is_winner  = (winner_idx == i)
            is_loser   = (winner_idx != -1 and not is_winner)

            # Scale winning card up, loser shrinks
            if winner_idx == -1:
                scale = 1.0
            elif is_winner:
                scale = 1.0 + min(0.07, (eff_el - 0.3) * 0.18) if eff_el > 0.3 else 1.0
            else:
                scale = max(0.91, 1.0 - min(0.09, (eff_el - 0.3) * 0.14)) if eff_el > 0.3 else 1.0

            dCW = int(CW * scale)
            dCH = int(CH * scale)
            dcx = cx_a + (CW - dCW) // 2
            dcy = CARD_TOP + (CH - dCH) // 2

            # ── Winner energy aura ─────────────────────────────────────────────
            if is_winner and eff_el > 0.25:
                glow_strength = min(1.0, (eff_el - 0.25) * 2.5)
                pulse_t = (math.sin(el * 4.5 + i) + 1) / 2
                for extra, base_lerp in ((22, 0.05), (12, 0.09), (5, 0.13)):
                    glow_col = lerp_col(C_BG, col,
                                        base_lerp * glow_strength * (0.65 + 0.35 * pulse_t))
                    pygame.draw.rect(self.screen, glow_col,
                                     (dcx - extra, dcy - extra,
                                      dCW + extra*2, dCH + extra*2),
                                     border_radius=26 + extra//2)

            # Winner particle bursts (periodic)
            if is_winner and eff_el > 0.5 and int(el * 12) % 12 == 0:
                self._ring_burst(dcx + dCW//2, dcy + dCH//3, col, n=8, spd=2.5)

            # ── Loser dim ─────────────────────────────────────────────────────
            loser_dim = 0.0
            if is_loser and eff_el > 0.3:
                loser_dim = min(0.50, (eff_el - 0.3) * 1.3)

            # Drop shadow
            shad_off = 10 if is_winner else 4
            shad_col = lerp_col(C_BG, (0, 0, 0), 0.6 if is_winner else 0.32)
            pygame.draw.rect(self.screen, shad_col,
                             (dcx + shad_off, dcy + shad_off, dCW, dCH),
                             border_radius=24)

            # Card body
            if is_winner:
                card_fill = lerp_col(C_PANEL, C_PANEL_LT, 0.6)
                bw_width  = 3
            elif is_loser:
                card_fill = lerp_col(C_PANEL, C_BG, 0.45)
                bw_width  = 1
            else:
                card_fill = C_PANEL
                bw_width  = 2

            if is_winner and eff_el > 0.25:
                pulse_t  = (math.sin(el * 5.0) + 1) / 2
                bdr_col  = lerp_col(col, lerp_col(col, C_WHITE, 0.7), pulse_t * 0.65)
            elif is_loser:
                bdr_col  = lerp_col(col, C_DIM, 0.72)
            else:
                bdr_col  = col

            panel(self.screen, card_fill, (dcx, dcy, dCW, dCH), r=22,
                  bw=bw_width, bc=bdr_col)

            # Inner bevel
            pygame.draw.line(self.screen,
                             lerp_col(card_fill, C_WHITE, 0.24 if is_winner else 0.09),
                             (dcx + 22, dcy + 1), (dcx + dCW - 22, dcy + 1))

            # Top accent bar
            bar_col = col if not is_loser else lerp_col(col, C_DIM, 0.65)
            bar_h   = 5 if is_winner else 3
            pygame.draw.rect(self.screen, bar_col,
                             (dcx + 14, dcy, dCW - 28, bar_h), border_radius=3)
            if is_winner:
                pygame.draw.rect(self.screen,
                                 lerp_col(col, C_WHITE, 0.6),
                                 (dcx + 14, dcy, dCW - 28, 2), border_radius=2)

            # Player label
            lbl_col = col if not is_loser else lerp_col(col, C_DIM, 0.52)
            txt(self.screen, lbl, LBL_FONT, lbl_col,
                (dcx + dCW//2, dcy + 22))

            # ── GESTURE ICON — animated character reveal ───────────────────────
            ICON_CY = dcy + 90 + int(dCH * 0.10)

            reveal_styles = {"Rock": "slam", "Paper": "unfold", "Scissors": "slash"}
            rev_t   = self._reveal_t[i]
            rev_sty = reveal_styles.get(gest, "pop") if gest else "pop"

            icon_alpha = 255 if not is_loser else 130

            if gest:
                gc = GESTURE_COLOR.get(gest, col)
                icon_col = lerp_col(C_WHITE, gc, 0.15) if is_winner else (
                    lerp_col(C_GREY, C_DIM, 0.35) if is_loser else C_WHITE)

                # Winner glow burst behind icon
                if is_winner and eff_el > 0.28:
                    glow_el  = eff_el - 0.28
                    g_sz     = int(ICON_SIZE * scale * 1.6)
                    g_pulse  = 0.6 + 0.4 * math.sin(el * 5.2)
                    glow_a   = int(min(85, glow_el * 200) * g_pulse)
                    if glow_a > 4:
                        glow_tmp = pygame.Surface((g_sz, g_sz), pygame.SRCALPHA)
                        draw_gesture_icon(glow_tmp, gest, g_sz//2, g_sz//2, g_sz, gc)
                        glow_tmp.set_alpha(glow_a)
                        self.screen.blit(glow_tmp,
                                         (dcx + dCW//2 - g_sz//2, ICON_CY - g_sz//2))

                # ── Gesture-specific reveal + idle (character idle only after reveal)
                idle_intensity = min(1.0, max(0.0, (rev_t - 0.7) * 3.3))
                eff_icon_size  = max(40, int(ICON_SIZE * scale))
                draw_gesture_animated(
                    self.screen, gest,
                    dcx + dCW//2, ICON_CY,
                    eff_icon_size, icon_col,
                    self._idle_t[gest],
                    intensity=idle_intensity,
                    alpha=icon_alpha,
                    reveal_t=rev_t,
                    reveal_style=rev_sty)

                # ── Gesture-specific after-reveal effects ──────────────────────
                if rev_t > 0.5:
                    post_t = (rev_t - 0.5) / 0.5
                    if gest == "Rock":
                        # Impact shockwave ring
                        slam = max(0.0, math.sin(self._idle_t[gest] * 2.4) ** 9)
                        if slam > 0.4 or (rev_t < 0.85 and i == 1):
                            ring_t2 = max(slam, 1.0 - rev_t)
                            ring_r2 = int(20 + 90 * ring_t2)
                            ring_col = lerp_col(C_BG, gc, (1-ring_t2) * 0.5)
                            pygame.draw.circle(self.screen, ring_col,
                                               (dcx + dCW//2, ICON_CY), ring_r2, 3)
                    elif gest == "Paper":
                        # Ripple waves expand outward
                        for wn in range(2):
                            wr2 = int((post_t * 80 + wn * 35) % 110)
                            wa2 = int(max(0, 50 * (1 - wr2/110)))
                            if wa2 > 3:
                                wave_col = lerp_col(C_BG, gc, 0.18)
                                pygame.draw.circle(self.screen, wave_col,
                                                   (dcx + dCW//2, ICON_CY), wr2, 2)
                    elif gest == "Scissors":
                        # Slash trails
                        snap = math.sin(self._idle_t[gest] * 4.8) ** 3
                        if snap > 0.4:
                            sx_off2 = int(idle_intensity * 40)
                            sy_off2 = int(snap * 20)
                            self._slash_trail.add(
                                dcx + dCW//2 - sx_off2 - 30,
                                ICON_CY + sy_off2 - 15,
                                dcx + dCW//2 - sx_off2 + 30,
                                ICON_CY + sy_off2 + 15,
                                gc, width=3)

                # AI reveal energy burst ring (early reveal phase)
                if i == 1 and 0.05 < rev_t < 0.6:
                    ring_t3  = rev_t / 0.6
                    ring_r3  = int(15 + ring_t3 * 95)
                    ring_a3f = max(0, 1 - ring_t3)
                    ring_c3  = lerp_col(C_BG, col, ring_a3f * 0.75)
                    pygame.draw.circle(self.screen, ring_c3,
                                       (dcx + dCW//2, ICON_CY), ring_r3, 3)
                    # Outer thin ring
                    pygame.draw.circle(self.screen,
                                       lerp_col(C_BG, col, ring_a3f * 0.25),
                                       (dcx + dCW//2, ICON_CY), ring_r3 + 14, 1)
            else:
                # No gesture — placeholder
                draw_gesture_icon(self.screen, "Rock",
                                  dcx + dCW//2, ICON_CY, 60,
                                  C_DIM, alpha=60)

            # ── Gesture name ───────────────────────────────────────────────────
            gc_name = g_color(gest, C_WHITE) if gest else C_GREY
            if is_loser:
                gc_name = lerp_col(gc_name, C_DIM, 0.55)
            name_txt = gest or "???"
            txt(self.screen, name_txt, NAME_FONT, gc_name,
                (dcx + dCW//2, dcy + dCH - 68),
                glow=is_winner, glow_col=gc_name, glow_r=6)

            # ── "beats X" hint ─────────────────────────────────────────────────
            if gest:
                beats_col = lerp_col(C_DIM, col, 0.28) if is_winner else C_DIM
                txt(self.screen, f"beats  {BEATS.get(gest, '')}",
                    self.f_2xs, beats_col, (dcx + dCW//2, dcy + dCH - 44))

            # ── WINNER crown badge ─────────────────────────────────────────────
            if is_winner and eff_el > 0.4:
                crown_a = min(255, int((eff_el - 0.4) * 500))
                crown_s = self.f_xs.render("★  WINNER  ★", True, C_YELLOW)
                crown_s.set_alpha(crown_a)
                self.screen.blit(crown_s,
                                 (dcx + dCW//2 - crown_s.get_width()//2, dcy - 30))

        # Draw slash trails accumulated during card animations
        self._slash_trail.draw(self.screen)

        # ── VS badge ──────────────────────────────────────────────────────────
        vx = WIN_W // 2
        vy = CARD_TOP + CH // 2
        vs_r = 30
        # Outer glow ring
        pygame.draw.circle(self.screen, lerp_col(C_BG, C_ACCENT, 0.12), (vx, vy), vs_r + 6)
        # Main circle
        pygame.draw.circle(self.screen, C_PANEL_LT, (vx, vy), vs_r)
        pygame.draw.circle(self.screen, C_BORDER,   (vx, vy), vs_r, 2)
        # Inner accent ring
        pygame.draw.circle(self.screen,
                           lerp_col(C_DIM, C_ACCENT, 0.2), (vx, vy), vs_r - 4, 1)
        txt(self.screen, "VS", _font(16), lerp_col(C_GREY, C_WHITE, 0.4), (vx, vy))

        # ── Info row below cards ───────────────────────────────────────────────
        info_y = CARD_TOP + CH + 22
        if self.tournament:
            txt(self.screen, self.tournament.score_str,
                self.f_mid, C_YELLOW, (WIN_W//2, info_y))
            info_y += 32
            txt(self.screen,
                f"First to {self.tournament.needed}  ·  {self.tournament.played} rounds",
                self.f_xs, C_DIM, (WIN_W//2, info_y))
            info_y += 20
        if self.mode == "single":
            pct = int((1 - self.ai.rand_prob) * 100)
            txt(self.screen, f"AI adaptation  {pct}%",
                self.f_2xs, C_DIM, (WIN_W//2, info_y))
            info_y += 18

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_a = min(255, int((el - 0.4) * 400))
        if btn_a > 0:
            bw_b, bh_b = 290, 52
            by1        = info_y + 8
            self._btn("SPACE  ·  Next Round", "next",
                      WIN_W//2 - bw_b//2, by1, bw_b, bh_b, C_ACCENT)
            self._btn("R  ·  Main Menu", "to_menu",
                      WIN_W//2 - bw_b//2, by1 + bh_b + 8, bw_b, 42, C_GREY)

    # ═══════════════════════════════════════════════════════════════════════════
    #  CHAMPION
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_champion(self):
        if not self.tournament:
            self._to_menu()
            return

        champ = self.tournament.champion
        fmt   = Tournament.FORMATS[self.tourn_fmt][0]
        col   = C_GREEN if champ == "p1" else (C_RED if champ == "p2" else C_DRAW)

        if champ == "p1":
            lbl = "YOU WIN THE TOURNAMENT!" if self.mode == "single" else "P1 IS CHAMPION!"
        elif champ == "p2":
            lbl = "AI WINS THE TOURNAMENT!" if self.mode == "single" else "P2 IS CHAMPION!"
        else:
            lbl = "TOURNAMENT DRAW"

        el    = time.perf_counter() - self.res_start
        pulse = 1.0 + 0.030 * math.sin(el * 4.0)

        # Background glow — direct lerp rect, no full-screen SRCALPHA per frame
        glow_t = 0.04 + 0.02 * math.sin(el * 2.2)
        gp_col = lerp_col(C_BG, col, glow_t)
        pygame.draw.ellipse(self.screen, gp_col,
                            (WIN_W//2 - 400, WIN_H//2 - 180, 800, 360))

        # Format badge
        txt(self.screen, fmt, self.f_xs,
            lerp_col(C_GREY, col, 0.4), (WIN_W//2, WIN_H//2 - 140))
        self._hsep(WIN_H//2 - 114, hw=280, col=lerp_col(C_DIM, col, 0.3))

        # Champion headline
        champ_f = _font(60)
        base  = champ_f.render(lbl, True, col)
        if abs(pulse - 1.0) > 0.005:
            scaled = pygame.transform.smoothscale(
                base, (max(1, int(base.get_width()*pulse)),
                       max(1, int(base.get_height()*pulse))))
        else:
            scaled = base
        # Glow pass
        gbase = champ_f.render(lbl, True, lerp_col(col, C_WHITE, 0.4))
        gbase.set_alpha(35)
        for dx2, dy2 in ((4,0),(-4,0),(0,4),(0,-4)):
            self.screen.blit(gbase, (WIN_W//2 - scaled.get_width()//2 + dx2,
                                      WIN_H//2 - scaled.get_height()//2 - 60 + dy2))
        self.screen.blit(scaled,
                         (WIN_W//2 - scaled.get_width()//2,
                          WIN_H//2 - scaled.get_height()//2 - 60))

        # Score display
        score_f = _font(44, mono=True)
        txt(self.screen, self.tournament.score_str,
            score_f, C_WHITE, (WIN_W//2, WIN_H//2 + 22))
        txt(self.screen, f"{self.tournament.played} rounds played",
            self.f_xs, C_GREY, (WIN_W//2, WIN_H//2 + 70))

        self._hsep(WIN_H//2 + 104, hw=280, col=lerp_col(C_DIM, col, 0.25))

        # Celebration particles
        if random.random() < 0.10:
            self._burst(
                random.randint(WIN_W // 5, 4 * WIN_W // 5),
                random.randint(WIN_H // 6, WIN_H // 2),
                random.choice([col, C_YELLOW, C_ACCENT, C_P2]), 18)

        self._btn("SPACE  ·  Back to Menu", "to_menu",
                  WIN_W//2 - 190, WIN_H//2 + 122, 380, 56, C_ACCENT)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
