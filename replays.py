"""Build the replay site: python replays.py [name ...] [--no-open]

Runs the champion of every trained AI (or only the ones you name) on all the tracks,
records the runs and writes them to site/, then opens site/index.html in your browser.
"""
import json
import os
import sys
import webbrowser

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # drawing happens off screen

import pygame

from game import runner as R, track as T
from game.render import View, car_sprite

SITE = os.path.join(R.ROOT, "site")
EVERY = 2   # keep one position out of two: 30 per second

args = [a for a in sys.argv[1:] if not a.startswith("--")]
names = args or sorted(f[:-3] for f in os.listdir(os.path.join(R.ROOT, "ais"))
                       if f.endswith(".py") and R.load_save(f[:-3]))
if not names:
    raise SystemExit("No trained AI yet. Run: python train.py <name>")
ais = [R.load_ai(name) for name in names]
teams = []
for ai in ais:
    save = R.load_save(ai.ID)
    if not save:
        raise SystemExit(f"{ai.ID} has no trained brain yet. Run: python train.py {ai.ID}")
    teams.append((ai, [save["best"]]))

os.makedirs(os.path.join(SITE, "img"), exist_ok=True)
view = View("replays")
for ai in ais:
    pygame.image.save(car_sprite(ai.COLOR), os.path.join(SITE, "img", f"car_{ai.ID}.png"))

tracks = []
for name in T.SPECS:
    track = T.load(name)
    view.track(track)
    pygame.image.save(view.screen, os.path.join(SITE, "img", f"{name}.png"))

    paths = [[] for _ in ais]

    def record(sim, frame):
        if frame % EVERY == 0:
            for i, path in enumerate(paths):
                if sim.live[i]:
                    path += [round(float(sim.x[i]), 1), round(float(sim.y[i]), 1), round(float(sim.angle[i]), 3)]

    sim, _ = R.run(track, teams, on_frame=record)
    for i, path in enumerate(paths):   # where each car ended up
        path += [round(float(sim.x[i]), 1), round(float(sim.y[i]), 1), round(float(sim.angle[i]), 3)]
    tracks.append({"name": name, "race": name == T.FINAL, "runs": [
        {"ai": ai.ID, "finished": bool(sim.finished[i]), "time": round(float(sim.time[i]), 2),
         "progress": round(float(sim.progress[i]), 3), "path": paths[i]}
        for i, ai in enumerate(ais)]})
    done = ", ".join(f"{ai.NAME} {sim.time[i]:.2f} s" if sim.finished[i] else f"{ai.NAME} {sim.progress[i]:.0%}"
                     for i, ai in enumerate(ais))
    print(f"{name:9s} {done}")

data = {"fps": round(1 / (EVERY * 1 / 60)), "width": T.W, "height": T.H, "tracks": tracks,
        "ais": [{"id": ai.ID, "name": ai.NAME, "color": list(ai.COLOR)} for ai in ais]}
with open(os.path.join(SITE, "runs.js"), "w") as f:
    f.write("const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")

page = os.path.join(SITE, "index.html")
print(f"Site ready: {page}")
if "--no-open" not in sys.argv:
    webbrowser.open("file://" + page)
