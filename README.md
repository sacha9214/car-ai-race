# Car AI Race

A top-down racing game with no AI in it. Each player writes their own self-driving AI,
trains it, and the trained cars race each other on a track they have never seen.

Inspired by commonLuke's video "Can I Make a Better AI Than AI?".

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Try the game yourself first (arrows or WASD, R to restart):

```bash
.venv/bin/python play.py
```

## Make your AI

Copy `ais/template.py` to `ais/<yourname>.py` and fill in its three functions:

| Function | Job |
|---|---|
| `create_population(size)` | Return the first generation of brains |
| `drive(brain, sensors, speed)` | Return `(steer, throttle)` for one car, 30 times a second |
| `evolve(brains, scores)` | Return the next generation from the scores of this one |

A brain is whatever you want it to be. The car sees 7 wall distances and its own speed,
nothing else. The template explains every value.

## Train

```bash
.venv/bin/python train.py yourname --gens 200
```

Add `--watch` to see the cars learn, `--fresh` to start over. Training runs on four
tracks (`oval`, `bean`, `wiggle`, `clover`), saves after every generation and resumes
where it stopped. The best brain ever seen is kept as your champion in `saves/yourname.pkl`.

## Race

```bash
.venv/bin/python race.py yourname friendname
```

The race runs on `gauntlet`, a harder track with an obstacle that is not part of training.
For neutral ground, pick a number together on race day and use a random track:

```bash
.venv/bin/python race.py yourname friendname --seed 42 --laps 3
```

First across the line wins. If nobody finishes, the car that got furthest wins.
The simulation is deterministic and cars never touch each other, so the result
depends only on the two AIs.

To race a friend, they send you `ais/<theirname>.py` and `saves/<theirname>.pkl`.
Both files run code on your machine, so only accept them from someone you trust.

## Suggested rules

- Same number of generations and same population size for everyone.
- Nobody trains on `gauntlet` or on the race-day seed.
- Agree beforehand on whether machine-learning libraries and AI assistants are allowed.

## Car physics

Top speed 420 px/s. Steering turns the car at up to 3.2 rad/s whatever the speed, so
the faster you go the wider you turn: tight corners need braking. Touching a wall, or
making no progress for 2.5 seconds, takes the car out. Each lap has a 30 second limit.
