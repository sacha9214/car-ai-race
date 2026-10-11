"""Train an AI: python train.py <name> [--gens 100] [--pop 250] [--watch] [--fresh]

Runs ais/<name>.py generation after generation on the training tracks and keeps the
best brain ever seen in saves/<name>.pkl. Stop any time with Ctrl+C: training resumes
from the last finished generation the next time you run it.
"""
import argparse
import time

import numpy as np

from game import runner as R, track as T

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("name")
p.add_argument("--gens", type=int, default=100, help="generations to run now")
p.add_argument("--pop", type=int, default=250, help="cars per generation (new training only)")
p.add_argument("--watch", action="store_true", help="show the cars while they train (slower)")
p.add_argument("--fresh", action="store_true", help="forget the saved training and start over")
args = p.parse_args()

ai = R.load_ai(args.name)
save = None if args.fresh else R.load_save(args.name)
if save is None:
    brains = list(ai.create_population(args.pop))
    save = {"gen": 0, "brains": brains, "best": None, "best_score": float("-inf")}
tracks = [T.load(name) for name in T.TRAINING]
view = None
if args.watch:
    from game.render import View
    view = View(f"Training {ai.NAME}")

try:
    for _ in range(args.gens):
        brains, t0 = save["brains"], time.time()
        colors = [ai.COLOR] * len(brains)
        total, laps = np.zeros(len(brains)), np.zeros(len(brains), int)
        for track in tracks:
            def draw(sim, frame):
                if frame % R.DECIDE_EVERY:
                    return
                if view.closed():
                    raise KeyboardInterrupt
                view.track(track)
                view.cars(sim, colors)
                view.text([f"{ai.NAME}   generation {save['gen'] + 1}   {track.name}",
                           f"{int(sim.live.sum())} of {sim.n} cars still running"])
                view.flip()
            sim, scores = R.run(track, [(ai, brains)], on_frame=draw if view else None)
            total += scores
            laps += sim.finished
        top = int(total.argmax())
        if total[top] > save["best_score"]:
            save["best"], save["best_score"] = brains[top], float(total[top])
        new = list(ai.evolve(brains, total.tolist()))
        if not new:
            raise SystemExit("evolve() returned no brains.")
        save["gen"] += 1
        save["brains"] = new
        R.write_save(args.name, save)
        print(f"gen {save['gen']:4d}   best {total[top]:6.0f}   mean {total.mean():6.0f}   "
              f"best car finished {laps[top]}/{len(tracks)} tracks   "
              f"record {save['best_score']:6.0f}   {time.time() - t0:.1f}s", flush=True)
except KeyboardInterrupt:
    print(f"\nStopped. Saved up to generation {save['gen']}.")
