"""Tracks: a closed centerline, a road mask and a distance-to-wall field."""
import math
import random

import numpy as np

W, H = 1280, 720
STEP = 4.0  # pixels between two centerline points

# name: (road width, [(harmonic, amplitude, phase), ...], [(lap fraction, side offset, radius), ...])
SPECS = {
    "oval": (110, [], []),
    "bean": (96, [(2, 0.12, 0.5), (3, 0.08, 1.0)], []),
    "wiggle": (88, [(5, 0.09, 0.0), (3, 0.07, 2.0)], []),
    "clover": (82, [(4, 0.15, 0.8), (2, 0.05, 0.0)], []),
    "gauntlet": (76, [(3, 0.10, 0.4), (7, 0.065, 1.3), (11, 0.03, 2.2)], [(0.42, 12, 13)]),
}
TRAINING = ["oval", "bean", "wiggle", "clover"]
FINAL = "gauntlet"


def _disk(radius):
    r = int(math.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= radius * radius


def _stamp(mask, cx, cy, disk, value):
    r = disk.shape[0] // 2
    x, y = int(round(cx)), int(round(cy))
    mask[y - r:y + r + 1, x - r:x + r + 1][disk] = value


def _wall_distance(road):
    """Chebyshev distance (in pixels) from every road pixel to the nearest wall."""
    dist = np.zeros(road.shape, np.int16)
    cur = road.copy()
    while cur.any():
        dist += cur
        h = cur.copy()
        h[:, 1:] &= cur[:, :-1]
        h[:, :-1] &= cur[:, 1:]
        cur = h.copy()
        cur[1:, :] &= h[:-1, :]
        cur[:-1, :] &= h[1:, :]
    return dist


class Track:
    def __init__(self, name, width, harmonics, obstacles):
        self.name = name
        self.width = width

        # Closed loop in polar form, squashed into the window.
        t = np.linspace(0, 2 * math.pi, 4000, endpoint=False)
        r = np.ones_like(t)
        for k, amp, phase in harmonics:
            r += amp * np.sin(k * t + phase)
        r /= r.max()
        dense = np.stack([W / 2 + 0.41 * W * r * np.cos(t), H / 2 + 0.38 * H * r * np.sin(t)], 1)

        # Resample so the points are evenly spaced along the loop.
        seg = np.linalg.norm(np.roll(dense, -1, 0) - dense, axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        n = int(arc[-1] // STEP)
        closed = np.vstack([dense, dense[:1]])
        s = np.arange(n) * (arc[-1] / n)
        self.center = np.stack([np.interp(s, arc, closed[:, 0]), np.interp(s, arc, closed[:, 1])], 1)
        self.n = n
        self.length = float(arc[-1])

        tangent = np.roll(self.center, -1, 0) - np.roll(self.center, 1, 0)
        self.heading = np.arctan2(tangent[:, 1], tangent[:, 0])

        self.road = np.zeros((H, W), bool)
        disk = _disk(width / 2)
        for x, y in self.center:
            _stamp(self.road, x, y, disk, True)
        self.obstacles = []
        for frac, offset, radius in obstacles:
            i = int(frac * n) % n
            a = self.heading[i] + math.pi / 2
            ox = self.center[i, 0] + offset * math.cos(a)
            oy = self.center[i, 1] + offset * math.sin(a)
            _stamp(self.road, ox, oy, _disk(radius), False)
            self.obstacles.append((ox, oy, radius))

        self.dist = _wall_distance(self.road)
        self.start = (float(self.center[0, 0]), float(self.center[0, 1]), float(self.heading[0]))


def random_spec(seed):
    """A track nobody has trained on: same seed, same track."""
    rng = random.Random(seed)
    harmonics = [(k, rng.uniform(0.25, 0.5) / k, rng.uniform(0, 2 * math.pi))
                 for k in rng.sample(range(2, 10), 3)]
    obstacles = [(rng.uniform(0.2, 0.8), rng.choice([-12, 12]), 13)] if rng.random() < 0.5 else []
    return rng.choice([76, 82, 88]), harmonics, obstacles


_cache = {}


def load(name):
    """A named track from SPECS, or 'seed:123' for a random one."""
    if name not in _cache:
        if name.startswith("seed:"):
            spec = random_spec(int(name[5:]))
        elif name in SPECS:
            spec = SPECS[name]
        else:
            raise SystemExit(f"Unknown track '{name}'. Choose from {', '.join(SPECS)} or seed:<number>.")
        _cache[name] = Track(name, *spec)
    return _cache[name]
