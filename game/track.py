"""Tracks: a winding road from a start line to a finish line, with obstacles on it."""
import random

import numpy as np

W, H = 1280, 720
STEP = 4.0  # pixels between two centerline points
SALMON, RED = (237, 125, 118), (238, 84, 62)

ROWS3, ROWS4 = [130, 360, 590], [100, 270, 440, 610]

# The road snakes across the window row by row. For each row, wiggles gives
# (number of bends, bend height in px).
#
# Obstacles are placed by `at`, the fraction of the track where they sit (0 = start, 1 = finish):
#   ("circle", at, offset from the middle of the road, radius)
#   ("bar",    at, side (-1 or 1), length)             a block sticking out of one wall
#   ("pinch",  at, length of the section, factor)      the road narrows to factor x its width
#   ("split",  at, length of the section, widening, island radius)   the road forks around an island
SPECS = {
    "pillars": dict(color=SALMON, width=96, ys=ROWS3, wiggles=[(0, 0), (3, 22), (0, 0)], obstacles=[
        ("circle", 0.10, 20, 12), ("circle", 0.17, -20, 12), ("circle", 0.24, 20, 12),
        ("circle", 0.45, -22, 12), ("circle", 0.55, 22, 12),
        ("circle", 0.76, -20, 12), ("circle", 0.83, 20, 12), ("circle", 0.90, -20, 12)]),
    "chicane": dict(color=SALMON, width=86, ys=ROWS4, wiggles=[(0, 0), (0, 0), (5, 24), (0, 0)], obstacles=[
        ("bar", 0.08, 1, 40), ("bar", 0.12, -1, 40), ("bar", 0.16, 1, 40),
        ("bar", 0.33, -1, 42), ("bar", 0.38, 1, 42), ("bar", 0.43, -1, 42),
        ("bar", 0.82, 1, 44), ("bar", 0.86, -1, 44), ("bar", 0.90, 1, 44)]),
    "pinch": dict(color=SALMON, width=90, ys=ROWS4, wiggles=[(0, 0), (7, 28), (0, 0), (7, 28)], obstacles=[
        ("pinch", 0.10, 150, 0.45), ("pinch", 0.19, 110, 0.4),
        ("pinch", 0.57, 150, 0.4), ("pinch", 0.67, 110, 0.36)]),
    "split": dict(color=SALMON, width=84, ys=ROWS3, wiggles=[(0, 0), (5, 30), (0, 0)], obstacles=[
        ("split", 0.11, 300, 2.2, 46), ("split", 0.24, 220, 2.0, 38),
        ("split", 0.80, 340, 2.2, 48)]),
    "gauntlet": dict(color=RED, width=78, ys=ROWS4, wiggles=[(1, 18), (9, 30), (4, 26), (11, 30)], obstacles=[
        ("split", 0.13, 130, 1.9, 24), ("bar", 0.30, 1, 34), ("circle", 0.58, -16, 11),
        ("circle", 0.64, 16, 11), ("pinch", 0.70, 110, 0.5)]),
}
TRAINING = ["pillars", "chicane", "pinch", "split"]
FINAL = "gauntlet"


def _snake(ys, wiggles, x0=120, x1=1160):
    points = []
    for k, y in enumerate(ys):
        bends, height = wiggles[k]
        xs = np.linspace(x0, x1, bends + 2)
        row = [(x, y + (height if j % 2 else -height) * (0 < j <= bends)) for j, x in enumerate(xs)]
        points += row if k % 2 == 0 else row[::-1]
    return points


def _smooth(points, rounds=4):
    """Round the corners of a path (Chaikin), keeping both ends in place."""
    p = np.array(points, float)
    for _ in range(rounds):
        cut = np.empty((2 * (len(p) - 1), 2))
        cut[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        cut[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], cut, p[-1:]])
    return p


_disks = {}


def _stamp(mask, cx, cy, radius, value):
    r = int(np.ceil(radius))
    if (r, radius) not in _disks:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        _disks[r, radius] = xx * xx + yy * yy <= radius * radius
    x, y = int(round(cx)), int(round(cy))
    y0, x0 = max(y - r, 0), max(x - r, 0)
    part = _disks[r, radius][y0 - (y - r):, x0 - (x - r):][:mask.shape[0] - y0, :mask.shape[1] - x0]
    mask[y0:y + r + 1, x0:x + r + 1][part] = value


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
    def __init__(self, name, color, width, ys, wiggles, obstacles):
        self.name = name
        self.color = color
        self.width = width

        # Evenly spaced points along the middle of the road.
        dense = _smooth(_snake(ys, wiggles))
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        n = int(arc[-1] // STEP) + 1
        s = np.linspace(0, arc[-1], n)
        self.center = np.stack([np.interp(s, arc, dense[:, 0]), np.interp(s, arc, dense[:, 1])], 1)
        self.n = n
        self.length = float(arc[-1])
        tangent = np.gradient(self.center, axis=0)
        self.heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        normal = np.stack([-np.sin(self.heading), np.cos(self.heading)], 1)

        def section(at, length):
            """Indices of a stretch of road and a 0..1..0 bump over it."""
            c, h = int(at * n), max(int(length / STEP / 2), 1)
            i = np.arange(max(c - h, 0), min(c + h + 1, n))
            return i, (i - c) / h

        self.widths = np.full(n, float(width))
        for kind, at, *args in obstacles:
            if kind in ("pinch", "split"):
                i, t = section(at, args[0])
                self.widths[i] *= 1 + (args[1] - 1) * np.cos(t * np.pi / 2) ** 2

        self.road = np.zeros((H, W), bool)
        for (x, y), w in zip(self.center, self.widths):
            _stamp(self.road, x, y, round(w / 2), True)
        # Solid frame so nothing can ever leave the window.
        self.road[:2], self.road[-2:], self.road[:, :2], self.road[:, -2:] = False, False, False, False

        for kind, at, *args in obstacles:
            c = int(at * n)
            if kind == "circle":
                x, y = self.center[c] + normal[c] * args[0]
                _stamp(self.road, x, y, args[1], False)
            elif kind == "bar":
                side, length = args
                for d in np.arange(self.widths[c] / 2 + 8, self.widths[c] / 2 - length, -3):
                    x, y = self.center[c] + normal[c] * side * d
                    _stamp(self.road, x, y, 8, False)
            elif kind == "split":
                i, t = section(at, args[0])
                for j, tj in zip(i, t):
                    if abs(tj) < 0.6:
                        r = round(args[2] * np.sqrt(1 - (tj / 0.6) ** 2))
                        if r > 0:
                            _stamp(self.road, *self.center[j], r, False)

        self.dist = _wall_distance(self.road)
        self.start = (float(self.center[0, 0]), float(self.center[0, 1]), float(self.heading[0]))


def random_spec(seed):
    """A track nobody has trained on: same seed, same track."""
    rng = random.Random(seed)
    four = rng.random() < 0.5
    ys = ROWS4 if four else ROWS3
    spec = dict(color=RED, width=rng.choice([80, 86]), ys=ys, obstacles=[],
                wiggles=[(rng.choice([0, 3, 5, 7, 9]), rng.randint(16, 30 if four else 40)) for _ in ys])
    for k in range(len(ys)):
        at = (k + rng.uniform(0.25, 0.4)) / len(ys)
        for _ in range(2):
            kind = rng.choice(["circle", "bar", "pinch", "none"] + ([] if four else ["split"]))
            side = rng.choice([-1, 1])
            if kind == "circle":
                spec["obstacles"].append(("circle", at, side * 18, 11))
            elif kind == "bar":
                spec["obstacles"].append(("bar", at, side, 36))
            elif kind == "pinch":
                spec["obstacles"].append(("pinch", at, 120, 0.5))
            elif kind == "split":
                spec["obstacles"].append(("split", at, 200, 2.0, 36))
            at += rng.uniform(0.3, 0.4) / len(ys)
    return spec


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
        _cache[name] = Track(name, **spec)
    return _cache[name]
