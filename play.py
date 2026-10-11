"""Drive a track yourself: python play.py [track]

Arrows or WASD to drive, R to restart, Escape to quit.
"""
import sys

import numpy as np
import pygame

from game import sim as S, track as T
from game.render import View, DIM

track = T.load(sys.argv[1] if len(sys.argv) > 1 else T.TRAINING[0])
view = View(f"Car AI Race: {track.name}")
sim, best = S.Sim(track, 1), None

while not view.closed():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        sim = S.Sim(track, 1)
    steer = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
    throttle = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
    was_live = sim.live[0]
    sim.step(np.array([float(steer)]), np.array([float(throttle)]))
    if sim.prog[0] == 0 and sim.speed[0] == 0:
        sim.time[0] = 0  # the clock starts when you do
    if was_live and sim.finished[0]:
        best = min(best or 999, sim.time[0])

    view.track(track)
    if sim.live[0]:
        view.sensors(sim, 0)
    view.cars(sim, [(230, 70, 60)])
    state = "FINISHED" if sim.finished[0] else "" if sim.alive[0] else "CRASHED"
    view.text([f"{sim.time[0]:.2f} s    {sim.speed[0]:.0f} px/s    {sim.progress[0]:.0%}",
               (f"best lap {best:.2f} s" if best else "arrows to drive, R to restart", DIM)])
    if state:
        view.banner(state)
    view.flip(60)
