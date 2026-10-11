"""Physics and sensors for many cars at once. Cars are ghosts: they never touch each other."""
import numpy as np

DT = 1 / 60
N_SENSORS = 7
SENSOR_RANGE = 250.0
SENSOR_ANGLES = np.radians(np.linspace(-90, 90, N_SENSORS))

MAX_SPEED = 420.0   # px/s
ACCEL = 320.0       # px/s² at full throttle
BRAKE = 560.0       # px/s² at full brake
DRAG = 60.0         # px/s² lost all the time
TURN = 3.2          # rad/s at full steering lock
CAR_L, CAR_W = 30.0, 15.0
STALL_TIME = 2.5    # seconds without progress before a car is removed

_CORNERS = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2],
                     [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2]])
_LOOKAHEAD = np.arange(13)


class Sim:
    def __init__(self, track, n):
        self.track = track
        self.n = n
        self.goal = track.n - 3   # centerline points to pass to reach the finish line
        x, y, a = track.start
        self.x = np.full(n, x)
        self.y = np.full(n, y)
        self.angle = np.full(n, a)
        self.speed = np.zeros(n)
        self.alive = np.ones(n, bool)       # False once crashed or stalled
        self.finished = np.zeros(n, bool)
        self.time = np.zeros(n)             # time driven so far
        self.prog = np.zeros(n, int)        # centerline points passed
        self._idx = np.zeros(n, int)
        self._last_gain = np.zeros(n)

    @property
    def live(self):
        return self.alive & ~self.finished

    @property
    def progress(self):
        """Fraction of the track done, 0 to 1."""
        return np.minimum(self.prog / self.goal, 1.0)

    def corners(self, i):
        c, s = np.cos(self.angle[i]), np.sin(self.angle[i])
        return [(self.x[i] + c * dx - s * dy, self.y[i] + s * dx + c * dy) for dx, dy in _CORNERS]

    def sense(self, who=None):
        """Wall distances, 0 (touching) to 1 (nothing within range). Shape (cars, 7), left to right."""
        who = np.flatnonzero(self.live) if who is None else np.asarray(who)
        out = np.ones((self.n, N_SENSORS))
        if not len(who):
            return out
        ang = self.angle[who, None] + SENSOR_ANGLES
        dx, dy = np.cos(ang), np.sin(ang)
        ox, oy = self.x[who, None], self.y[who, None]
        dist = self.track.dist
        t = np.zeros_like(ang)
        done = np.zeros(ang.shape, bool)
        for _ in range(int(SENSOR_RANGE)):
            d = dist[(oy + dy * t).astype(int), (ox + dx * t).astype(int)]
            done |= (d == 0) | (t >= SENSOR_RANGE)
            if done.all():
                break
            t = np.where(done, t, t + np.maximum(d - 1, 1))
        out[who] = np.minimum(t, SENSOR_RANGE) / SENSOR_RANGE
        return out

    def step(self, steer, throttle):
        """Advance one frame. steer and throttle are arrays in [-1, 1]."""
        m = self.live
        if not m.any():
            return
        steer, throttle = steer[m], throttle[m]
        acc = np.where(throttle >= 0, throttle * ACCEL, throttle * BRAKE) - DRAG
        speed = np.clip(self.speed[m] + acc * DT, 0, MAX_SPEED)
        angle = self.angle[m] + steer * TURN * np.minimum(speed / 100, 1) * DT
        c, s = np.cos(angle), np.sin(angle)
        x = self.x[m] + c * speed * DT
        y = self.y[m] + s * speed * DT
        time = self.time[m] + DT
        self.speed[m], self.angle[m], self.x[m], self.y[m], self.time[m] = speed, angle, x, y, time

        cx = x[:, None] + c[:, None] * _CORNERS[:, 0] - s[:, None] * _CORNERS[:, 1]
        cy = y[:, None] + s[:, None] * _CORNERS[:, 0] + c[:, None] * _CORNERS[:, 1]
        crashed = ~self.track.road[cy.astype(int), cx.astype(int)].all(axis=1)

        # Progress only counts forward: look for the closest centerline point just ahead.
        ahead = self.track.center[np.minimum(self._idx[m, None] + _LOOKAHEAD, self.track.n - 1)]
        gain = ((ahead[:, :, 0] - x[:, None]) ** 2 + (ahead[:, :, 1] - y[:, None]) ** 2).argmin(axis=1)
        self._idx[m] += gain
        self.prog[m] += gain
        self._last_gain[m] = np.where(gain > 0, time, self._last_gain[m])
        stalled = time - self._last_gain[m] > STALL_TIME

        self.finished[m] = self.prog[m] >= self.goal
        self.alive[m] = ~(crashed | stalled)
