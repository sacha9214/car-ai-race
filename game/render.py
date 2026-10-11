"""Pygame drawing, shared by play.py, train.py --watch and race.py."""
import numpy as np
import pygame

from . import sim as S
from .track import W, H

GRASS, ROAD, KERB = (24, 34, 30), (58, 60, 68), (225, 225, 230)
TEXT, DIM = (235, 235, 240), (140, 145, 150)


class View:
    def __init__(self, title):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.big = pygame.font.Font(None, 96)
        self._track = None

    def closed(self):
        """True when the window was closed or Escape was pressed."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return True
        return False

    def track(self, track):
        if self._track is not track:
            rgb = np.where(track.road[..., None], ROAD, GRASS).astype(np.uint8)
            rgb[(track.dist > 0) & (track.dist <= 3)] = KERB
            self._bg = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            x, y, a = track.start
            nx, ny = -np.sin(a) * track.width / 2, np.cos(a) * track.width / 2
            pygame.draw.line(self._bg, KERB, (x - nx, y - ny), (x + nx, y + ny), 4)
            self._track = track
        self.screen.blit(self._bg, (0, 0))

    def cars(self, sim, colors):
        # Wrecks first so the cars still racing stay on top.
        for i in np.argsort(sim.alive, kind="stable"):
            color = colors[i] if sim.alive[i] else tuple(c // 2 for c in colors[i])
            pygame.draw.polygon(self.screen, color, sim.corners(i))

    def sensors(self, sim, i):
        hits = sim.sense([i])[i] * S.SENSOR_RANGE
        for a, d in zip(sim.angle[i] + S.SENSOR_ANGLES, hits):
            end = (sim.x[i] + np.cos(a) * d, sim.y[i] + np.sin(a) * d)
            pygame.draw.line(self.screen, (250, 210, 80), (sim.x[i], sim.y[i]), end, 1)
            pygame.draw.circle(self.screen, (250, 210, 80), end, 3)

    def text(self, lines, x=W // 2, y=H // 2, color=TEXT):
        """Lines of text centred on (x, y). A line can be (string, color)."""
        y -= len(lines) * 13
        for line in lines:
            s, c = line if isinstance(line, tuple) else (line, color)
            img = self.font.render(s, True, c)
            self.screen.blit(img, (x - img.get_width() // 2, y))
            y += 26

    def banner(self, s, color=TEXT):
        img = self.big.render(s, True, color)
        box = img.get_rect(center=(W // 2, H // 2 - 110)).inflate(40, 20)
        pygame.draw.rect(self.screen, (10, 12, 14), box, border_radius=12)
        self.screen.blit(img, img.get_rect(center=box.center))

    def flip(self, fps=0):
        pygame.display.flip()
        self.clock.tick(fps)
