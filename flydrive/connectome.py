"""Connectome-derived weight matrices for the FlyDrive controller.

The controller uses a small, well-characterised slice of the Drosophila
visual-motor pathway:

    T4/T5 (EMD array)  ->  LPTC / H2  ->  DNa02  ->  steering

H2 is a horizontal-system lobula plate tangential cell: it responds to
back-to-front horizontal motion in the *contralateral* eye and projects to
descending neurons that drive turning. DNa02 is the descending neuron whose
left/right activity difference sets turn direction in walking flies.

If you have a FlyWire connectivity export, drop it at
``data/flywire_edges.csv`` with columns:

    pre_type,post_type,syn_count,sign

and it will be used verbatim. Otherwise a literature-scaled default is used,
so the whole project runs with no downloads.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

# pre_type, post_type, synapse count, sign (+1 excitatory, -1 inhibitory)
DEFAULT_EDGES: list[tuple[str, str, int, int]] = [
    ("T4T5_L", "H2_R", 884, +1),   # H2 is contralateral
    ("T4T5_R", "H2_L", 902, +1),
    ("H2_L", "H2_R", 212, -1),     # reciprocal inhibition between the pair
    ("H2_R", "H2_L", 205, -1),
    ("H2_L", "DNa02_L", 646, +1),
    ("H2_R", "DNa02_R", 671, +1),
    ("H2_L", "DNa02_R", 138, -1),  # weak crossed inhibition
    ("H2_R", "DNa02_L", 131, -1),
]

NODES = ["T4T5_L", "T4T5_R", "H2_L", "H2_R", "DNa02_L", "DNa02_R"]


@dataclass
class Connectome:
    nodes: list[str]
    syn: np.ndarray          # raw synapse counts, [pre, post]
    sign: np.ndarray         # +1 / -1
    weights: np.ndarray = field(init=False)   # signed, normalised to max 1

    def __post_init__(self) -> None:
        peak = self.syn.max() if self.syn.max() > 0 else 1.0
        self.weights = self.sign * self.syn / peak

    def idx(self, name: str) -> int:
        return self.nodes.index(name)

    def w(self, pre: str, post: str) -> float:
        return float(self.weights[self.idx(pre), self.idx(post)])

    def summary(self) -> str:
        lines = ["pre -> post            syn    weight"]
        for i, p in enumerate(self.nodes):
            for j, q in enumerate(self.nodes):
                if self.syn[i, j]:
                    lines.append(f"{p:>7} -> {q:<9} {int(self.syn[i, j]):>6} "
                                 f"{self.weights[i, j]:+.3f}")
        return "\n".join(lines)


def load(path: str | None = "data/flywire_edges.csv") -> Connectome:
    edges = DEFAULT_EDGES
    if path and os.path.exists(path):
        edges = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                edges.append((row["pre_type"], row["post_type"],
                              int(row["syn_count"]),
                              int(row.get("sign", 1))))
    nodes = list(dict.fromkeys(NODES + [e[0] for e in edges] + [e[1] for e in edges]))
    n = len(nodes)
    syn = np.zeros((n, n))
    sign = np.ones((n, n))
    for pre, post, count, sg in edges:
        i, j = nodes.index(pre), nodes.index(post)
        syn[i, j] = count
        sign[i, j] = sg
    return Connectome(nodes, syn, sign)


if __name__ == "__main__":
    print(load().summary())
