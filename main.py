"""FlyDrive command line.

    python main.py train      --generations 30
    python main.py evaluate
    python main.py report
    python main.py demo
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from flydrive.connectome import load as load_connectome
from flydrive.controllers import FlyController, PIDController, rollout
from flydrive.evaluate import evaluate, table
from flydrive.snn import DEFAULT
from flydrive.train import load_params, train


def cmd_train(a):
    train(generations=a.generations, population=a.population,
          seeds_per_eval=a.seeds, duration=a.duration, speed=a.speed,
          curvature=a.curvature, obstacles=a.obstacles, workers=a.workers,
          out=a.out)


def cmd_evaluate(a):
    p = load_params(os.path.join(a.out, "trained_params.json"))
    res = evaluate(p, seeds=range(5000, 5000 + a.seeds), duration=a.duration,
                   speed_kmh=a.speed, curvature=a.curvature,
                   n_obstacles=a.obstacles, out=a.out)
    print("\nControllers\n" + table(res["controllers"]))
    print("\nAblations\n" + table(res["ablations"]))


def cmd_report(a):
    from flydrive import viz
    p = load_params(os.path.join(a.out, "trained_params.json"))
    figs = []

    hist_path = os.path.join(a.out, "trained_params.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            figs.append(viz.learning_curve(json.load(fh)["history"]))

    logs = {}
    m_fly, trace = rollout(FlyController(params=p, seed=7), seed=7,
                           duration=a.duration, record=True)
    logs["fly (trained)"] = m_fly["log"]
    m_un, _ = rollout(FlyController(params=DEFAULT, seed=7), seed=7, duration=a.duration)
    logs["fly (untrained)"] = m_un["log"]
    m_pid, _ = rollout(PIDController(), seed=7, duration=a.duration)
    logs["PID baseline"] = m_pid["log"]

    figs += [viz.deviation(logs), viz.trajectory(m_fly["log"]), viz.raster(trace)]

    ev_path = os.path.join(a.out, "evaluation.json")
    if os.path.exists(ev_path):
        with open(ev_path) as fh:
            figs.append(viz.ablation_bars(json.load(fh)["ablations"]))
    print("figures written:")
    for f in figs:
        print("  " + f)


def cmd_demo(a):
    p = load_params(os.path.join(a.out, "trained_params.json"))
    for name, ctrl in [("fly", FlyController(params=p, seed=1)),
                       ("pid", PIDController())]:
        m, _ = rollout(ctrl, seed=1, duration=a.duration)
        m.pop("log")
        print(f"{name:>4}: " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float)
                                         else f"{k}={v}" for k, v in m.items()))


def cmd_connectome(a):
    print(load_connectome().summary())


def main():
    ap = argparse.ArgumentParser(description="FlyDrive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, duration=30.0):
        p.add_argument("--duration", type=float, default=duration)
        p.add_argument("--speed", type=float, default=18.0)
        p.add_argument("--curvature", type=float, default=1.0)
        p.add_argument("--obstacles", type=int, default=2)
        p.add_argument("--out", default="artifacts")

    t = sub.add_parser("train"); common(t, 25.0)
    t.add_argument("--generations", type=int, default=25)
    t.add_argument("--population", type=int, default=32)
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--workers", type=int, default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate"); common(e)
    e.add_argument("--seeds", type=int, default=8)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report"); common(r)
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("demo"); common(d)
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("connectome"); c.set_defaults(func=cmd_connectome)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
