"""Copy this file to ais/<yourname>.py and write your AI in it.

This template has no intelligence: every car floors it in a straight line and nothing
is ever learned. The three functions below are the whole contract with the game.

A "brain" is whatever you decide it is (a list of numbers, a dict, an object...), as
long as pickle can save it. The game never looks inside; it only hands your brains
back to your own functions.
"""

NAME = "Template"
COLOR = (230, 70, 60)


def create_population(size):
    """Return a list of `size` brand-new brains. Called once, for generation 1."""
    return [None] * size


def drive(brain, sensors, speed):
    """Called 30 times a second for each car that is still running.

    sensors: 7 wall distances, from the car's left (-90°) to its right (+90°) in 30°
             steps, each from 0.0 (touching the wall) to 1.0 (no wall within 250 px).
    speed:   0.0 (stopped) to 1.0 (top speed).

    Return (steer, throttle), both from -1 to 1:
    steer    -1 = full left, +1 = full right
    throttle +1 = full gas, -1 = full brake (there is no reverse)
    """
    return 0.0, 1.0


def evolve(brains, scores):
    """Called at the end of every generation. Return the brains of the next one.

    scores[i] is how well brains[i] did, summed over the training tracks.
    Per track: 1000 x the fraction of the track completed, plus up to 1000 more
    for reaching the finish line (the faster, the bigger the bonus).
    """
    return brains
