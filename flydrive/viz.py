"""Figures for the report. Everything writes to artifacts/figures/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK = {"figure.facecolor": "#0e1622", "axes.facecolor": "#0e1622",
        "savefig.facecolor": "#0e1622", "text.color": "#e6eef8",
        "axes.labelcolor": "#e6eef8", "xtick.color": "#8fa3ba",
        "ytick.color": "#8fa3ba", "axes.edgecolor": "#1e2c3f",
        "grid.color": "#1e2c3f", "font.size": 10}
GREEN, BLUE, RED, AMBER, VIOLET = "#4ade80", "#5aa9ff", "#ff5f57", "#ffab4d", "#b98cff"


def _fig(path, fig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def learning_curve(history, out="artifacts/figures/learning_curve.png"):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(8, 4))
        g = [h["generation"] for h in history]
        ax.plot(g, [h["best"] for h in history], color=GREEN, label="best of generation")
        ax.plot(g, [h["elite_mean"] for h in history], color=BLUE, label="elite mean")
        ax.plot(g, [h["mean"] for h in history], color="#5d6f88", label="population mean")
        ax.set_xlabel("generation"); ax.set_ylabel("fitness")
        ax.set_title("Learning the lane")
        ax.grid(alpha=.3); ax.legend(facecolor="#121d2c", edgecolor="#1e2c3f")
        return _fig(out, fig)


def deviation(logs: dict, lane_half=3.5, out="artifacts/figures/lane_deviation.png"):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for (label, log), c in zip(logs.items(), [GREEN, AMBER, BLUE, VIOLET, RED]):
            ax.plot(log["t"], log["dev"], color=c, lw=1.4, label=label)
        ax.axhspan(-0.5, 0.5, color=GREEN, alpha=.08)
        for y in (-lane_half, lane_half):
            ax.axhline(y, color=RED, ls="--", lw=1, alpha=.6)
        ax.set_xlabel("time (s)"); ax.set_ylabel("deviation (m)")
        ax.set_title("Distance from lane centre")
        ax.grid(alpha=.3); ax.legend(facecolor="#121d2c", edgecolor="#1e2c3f")
        return _fig(out, fig)


def trajectory(log, out="artifacts/figures/trajectory.png"):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(log["s"], log["center"], color="#3d5273", ls="--", label="road centreline")
        ax.plot(log["s"], log["lat"], color=GREEN, label="driven path")
        ax.set_xlabel("distance along road (m)"); ax.set_ylabel("lateral (m)")
        ax.set_title("Trajectory"); ax.grid(alpha=.3)
        ax.legend(facecolor="#121d2c", edgecolor="#1e2c3f")
        return _fig(out, fig)


def raster(trace, out="artifacts/figures/raster.png"):
    rows = ["h2_l", "h2_r", "dn_l", "dn_r"]
    colors = [BLUE, VIOLET, RED, AMBER]
    with plt.rc_context(DARK):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                       gridspec_kw={"height_ratios": [1.3, 1]})
        for r, (key, c) in enumerate(zip(rows, colors)):
            for t, sp in zip(trace["t"], trace["spikes"]):
                for i in sp[key]:
                    ax1.vlines(t, r + i / 24, r + i / 24 + 0.03, color=c, lw=.8)
        ax1.set_yticks(np.arange(len(rows)) + .25)
        ax1.set_yticklabels(["H2 L", "H2 R", "DNa02 L", "DNa02 R"])
        ax1.set_title("Spiking activity"); ax1.grid(alpha=.2, axis="x")
        for key, c in zip(rows, colors):
            ax2.plot(trace["t"], trace[key], color=c, lw=1.3, label=key)
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("rate (Hz)")
        ax2.grid(alpha=.3); ax2.legend(ncol=4, facecolor="#121d2c", edgecolor="#1e2c3f")
        return _fig(out, fig)


def ablation_bars(rows, out="artifacts/figures/ablation.png"):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [r["condition"] for r in rows]
        vals = [r["mean_abs_dev"] for r in rows]
        colors = [GREEN if l == "intact" else AMBER for l in labels]
        ax.barh(labels, vals, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("mean |deviation| (m)")
        ax.set_title("Ablating the pathway")
        ax.grid(alpha=.3, axis="x")
        return _fig(out, fig)
