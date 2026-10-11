"""Race the trained champions: python race.py <name> <name> ... [--track gauntlet | --seed 42]

Each AI enters the best brain from its saves/<name>.pkl. First to the finish line wins;
if nobody finishes, the car that got furthest wins.
"""
import argparse

import numpy as np

from game import runner as R, track as T

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("names", nargs="+")
p.add_argument("--track", default=T.FINAL, help=f"one of {', '.join(T.SPECS)}")
p.add_argument("--seed", type=int, help="race on a random track instead (same seed, same track)")
p.add_argument("--headless", action="store_true", help="no window, just print the result")
args = p.parse_args()

teams = []
for name in args.names:
    ai, save = R.load_ai(name), R.load_save(name)
    if not save:
        raise SystemExit(f"{name} has no trained brain yet. Run: python train.py {name}")
    teams.append((ai, [save["best"]]))
ais = [ai for ai, _ in teams]
track = T.load(f"seed:{args.seed}" if args.seed is not None else args.track)


def standings(sim):
    """Car indices from first to last: finishers by time, then everyone else by distance."""
    return sorted(range(sim.n), key=lambda i: (not sim.finished[i], sim.time[i] if sim.finished[i] else -sim.prog[i]))


def status(sim, i):
    if sim.finished[i]:
        return f"{sim.time[i]:.2f} s"
    return f"{sim.progress[i]:.0%}" + ("" if sim.alive[i] else "  crashed")


view = None
if not args.headless:
    import pygame
    from game import sim as S
    from game.render import View
    view = View("THE RACE")
    colors = [ai.COLOR for ai in ais]

    def board(sim):
        view.track(track)
        view.cars(sim, colors)
        view.bar([(f"{pos}. {ais[i].NAME} {status(sim, i)}", ais[i].COLOR)
                  for pos, i in enumerate(standings(sim), 1)])

    def draw(sim, frame):
        if view.closed():
            return False
        board(sim)
        view.flip(60)

    grid = S.Sim(track, len(ais))
    for count in ("3", "2", "1", "GO!"):
        for _ in range(45):
            if view.closed():
                raise SystemExit
            board(grid)
            view.banner(count)
            view.flip(60)

sim, _ = R.run(track, teams, on_frame=draw if view else None)
order = standings(sim)
print(f"Track: {track.name}")
for pos, i in enumerate(order, 1):
    print(f"{pos}. {ais[i].NAME:16s} {status(sim, i)}")

if view:
    while not view.closed():
        board(sim)
        view.banner(f"{ais[order[0]].NAME} wins!", ais[order[0]].COLOR)
        view.flip(30)
