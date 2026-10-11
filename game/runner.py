"""Loads the AIs in ais/ and runs them on a track."""
import importlib.util
import math
import os
import pickle

import numpy as np

from . import sim as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECIDE_EVERY = 2    # an AI picks its controls every 2 frames (30 times a second)
TIME_LIMIT = 60.0   # seconds allowed per track


def load_ai(name):
    path = os.path.join(ROOT, "ais", name + ".py")
    if not os.path.exists(path):
        raise SystemExit(f"No AI named '{name}': {path} does not exist.")
    spec = importlib.util.spec_from_file_location("ais." + name, path)
    ai = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai)
    for fn in ("create_population", "drive", "evolve"):
        if not callable(getattr(ai, fn, None)):
            raise SystemExit(f"ais/{name}.py must define {fn}().")
    ai.ID = name
    ai.NAME = getattr(ai, "NAME", name)
    ai.COLOR = tuple(getattr(ai, "COLOR", (230, 70, 60)))
    return ai


def save_path(name):
    return os.path.join(ROOT, "saves", name + ".pkl")


def load_save(name):
    try:
        with open(save_path(name), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def write_save(name, data):
    os.makedirs(os.path.dirname(save_path(name)), exist_ok=True)
    tmp = save_path(name) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, save_path(name))


def _control(v):
    v = float(v)
    return 0.0 if math.isnan(v) else max(-1.0, min(1.0, v))


def run(track, teams, on_frame=None):
    """Drive every brain of every team on the track at the same time.

    teams is a list of (ai module, brains). on_frame(sim, frame) is called after each
    frame and can return False to stop early. Returns the Sim and one score per car.
    """
    n = sum(len(brains) for _, brains in teams)
    sim = S.Sim(track, n)
    steer, throttle = np.zeros(n), np.zeros(n)
    for frame in range(int(TIME_LIMIT / S.DT)):
        if frame % DECIDE_EVERY == 0:
            sensors = sim.sense().tolist()
            speeds = (sim.speed / S.MAX_SPEED).tolist()
            live = sim.live
            i = 0
            for ai, brains in teams:
                for brain in brains:
                    if live[i]:
                        s, t = ai.drive(brain, sensors[i], speeds[i])
                        steer[i], throttle[i] = _control(s), _control(t)
                    i += 1
        sim.step(steer, throttle)
        if on_frame and on_frame(sim, frame) is False:
            break
        if not sim.live.any():
            break
    scores = 1000 * sim.progress + np.where(sim.finished, 1000 * (1 - sim.time / TIME_LIMIT), 0)
    return sim, scores.tolist()
