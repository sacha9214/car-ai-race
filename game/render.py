"""Pygame drawing, shared by play.py, train.py --watch and race.py."""
import numpy as np
import pygame

from . import sim as S
from .track import W, H

ROAD = (8, 8, 22)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RAY, RAY_TIP = (255, 232, 60), (235, 50, 40)
_ANGLES = 72  # car sprites are pre-rotated in 5° steps


def car_sprite(color):
    """A little top-down car pointing right, drawn big then shrunk so it comes out smooth."""
    k = 6
    L, Wd = int(S.CAR_L) + 4, int(S.CAR_W) + 4
    s = pygame.Surface((L * k, Wd * k), pygame.SRCALPHA)
    dark = tuple(c * 2 // 5 for c in color)
    for x in (5, 23):       # wheels
        for y in (0, Wd - 4):
            pygame.draw.rect(s, (20, 20, 24), (x * k, y * k, 7 * k, 4 * k), border_radius=k)
    pygame.draw.rect(s, dark, (1 * k, 2 * k, (L - 2) * k, (Wd - 4) * k), border_radius=5 * k)
    pygame.draw.rect(s, color, (2 * k, 3 * k, (L - 4) * k, (Wd - 6) * k), border_radius=4 * k)
    pygame.draw.rect(s, (25, 28, 40), (17 * k, 5 * k, 6 * k, (Wd - 10) * k), border_radius=2 * k)   # windshield
    pygame.draw.rect(s, (25, 28, 40), (6 * k, 6 * k, 3 * k, (Wd - 12) * k), border_radius=k)        # rear window
    pygame.draw.rect(s, dark, (10 * k, 5 * k, 6 * k, (Wd - 10) * k), border_radius=2 * k)           # roof
    for y in (4, Wd - 7):   # headlights
        pygame.draw.rect(s, (255, 250, 200), ((L - 4) * k, y * k, 2 * k, 3 * k), border_radius=k)
    return pygame.transform.smoothscale(s, (L, Wd))


class View:
    def __init__(self, title):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        rounded = pygame.font.match_font("arialroundedmtbold")
        self.font = pygame.font.Font(rounded, 24) if rounded else pygame.font.SysFont(None, 30, bold=True)
        self.big = pygame.font.Font(rounded, 84) if rounded else pygame.font.SysFont(None, 100, bold=True)
        self._track = None
        self._sprites = {}

    def closed(self):
        """True when the window was closed or Escape was pressed."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return True
        return False

    def _checkers(self, track, i):
        """A checkered line across the road at centerline point i."""
        a, size = track.heading[i], 9
        along = np.array([np.cos(a), np.sin(a)]) * size
        across = np.array([-np.sin(a), np.cos(a)]) * size
        cols = int(track.widths[i] / size) + 1
        for r in range(2):
            for c in range(cols):
                p = track.center[i] + along * (r - 1) + across * (c - cols / 2)
                color = WHITE if (r + c) % 2 else BLACK
                pygame.draw.polygon(self._bg, color, [p, p + along, p + along + across, p + across])

    def track(self, track):
        if self._track is not track:
            # Average each pixel with its neighbours so the road edge is not jagged.
            pad = np.pad(track.road.astype(np.float32), 1)
            soft = sum(pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)) / 9
            wall, road = np.array(track.color, np.float32), np.array(ROAD, np.float32)
            rgb = (wall + (road - wall) * soft[..., None]).astype(np.uint8)
            self._bg = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self._checkers(track, 6)
            self._checkers(track, track.n - 3)
            self._track = track
        self.screen.blit(self._bg, (0, 0))

    def cars(self, sim, colors):
        # Wrecks first so the cars still racing stay on top.
        for i in np.argsort(sim.alive, kind="stable"):
            step = int(round(sim.angle[i] / (2 * np.pi) * _ANGLES)) % _ANGLES
            key = (colors[i], bool(sim.alive[i]), step)
            if key not in self._sprites:
                color = colors[i] if sim.alive[i] else tuple(c // 2 for c in colors[i])
                self._sprites[key] = pygame.transform.rotate(car_sprite(color), -step * 360 / _ANGLES)
            img = self._sprites[key]
            self.screen.blit(img, (sim.x[i] - img.get_width() / 2, sim.y[i] - img.get_height() / 2))

    def sensors(self, sim, who=None):
        """The sensor rays of the given cars (all the running ones by default)."""
        who = np.flatnonzero(sim.live) if who is None else np.asarray(who)
        hits = sim.sense(who)[who] * S.SENSOR_RANGE
        ang = sim.angle[who, None] + S.SENSOR_ANGLES
        ex = sim.x[who, None] + np.cos(ang) * hits
        ey = sim.y[who, None] + np.sin(ang) * hits
        for n, i in enumerate(who):
            for x, y in zip(ex[n], ey[n]):
                pygame.draw.line(self.screen, RAY, (sim.x[i], sim.y[i]), (x, y), 1)
                pygame.draw.circle(self.screen, RAY_TIP, (x, y), 2)

    def _outlined(self, font, s, color, x, y, t):
        shadow = font.render(s, True, BLACK)
        for dx, dy in ((-t, 0), (t, 0), (0, -t), (0, t), (-t, -t), (t, t), (-t, t), (t, -t)):
            self.screen.blit(shadow, (x + dx, y + dy))
        self.screen.blit(font.render(s, True, color), (x, y))

    def bar(self, items):
        """One line of text along the top. An item is a string or (string, color)."""
        x = 14
        for item in items:
            s, color = item if isinstance(item, tuple) else (item, WHITE)
            self._outlined(self.font, s, color, x, 8, 2)
            x += self.font.size(s)[0] + 26

    def banner(self, s, color=WHITE):
        w, h = self.big.size(s)
        self._outlined(self.big, s, color, (W - w) // 2, (H - h) // 2, 4)

    def flip(self, fps=0):
        pygame.display.flip()
        self.clock.tick(fps)
