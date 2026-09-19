# FlyDrive — teaching a fruit fly brain to drive

A simulation study: take the wiring of a real *Drosophila melanogaster* visual-motor
pathway from the FlyWire connectome, build a spiking network that obeys that wiring,
and see whether it can keep a car in its lane.

The connectome fixes **what connects to what and with which sign**. It does not tell
you membrane time constants, thresholds, resting drive, or how a descending neuron's
firing rate maps onto a steering wheel. Those are the free parameters, and that is what
gets trained here. The brain is never rewired — only tuned.

```
retina (32 ommatidia)
   -> T4/T5 elementary motion detectors      vision.py
   -> H2 (lobula plate tangential cell, contralateral, mutually inhibitory)
   -> DNa02 (descending neuron, left/right pair)     snn.py
   -> steering command                              env.py
```

## Quick start

```bash
pip install -r requirements.txt

python main.py connectome          # print the wiring actually used
python main.py demo                # one run, fly vs PID
python main.py train --generations 24
python main.py evaluate            # controllers + full ablation study
python main.py report              # figures into artifacts/figures/
```

No downloads needed. A literature-scaled default connectome ships in
`flydrive/connectome.py`.

## Using your own FlyWire export

Drop a CSV at `data/flywire_edges.csv`:

```csv
pre_type,post_type,syn_count,sign
T4T5_L,H2_R,884,1
H2_L,H2_R,212,-1
H2_L,DNa02_L,646,1
```

It is loaded automatically and overrides the defaults. Synapse counts are normalised
to the largest edge, so absolute counts do not matter — ratios do.

## How the vision works

Real flies do not do frame differencing. They use correlation-type motion detectors:
a delayed signal from one ommatidium multiplied by the undelayed signal from its
neighbour, minus the mirror-image product. `vision.EMDArray` implements exactly that
with a first-order delay line.

One detail matters a lot. The right hemifield is mirrored before pooling, so that both
pools are expressed as back-to-front motion *for their own eye* — H2's preferred
direction. A yaw rotation then drives the two pools in opposite directions, and forward
translation drives them together. That asymmetry is the course-stabilisation signal,
and without the mirror the rotation cue cancels out and the controller oscillates.

## How training works

Hard spike thresholds give no usable gradient, so `train.py` uses the cross-entropy
method: sample parameter vectors from a Gaussian, keep the top 25%, refit the Gaussian.
Ten free parameters, listed in `snn.PARAM_NAMES`.

Each candidate is scored on several **fresh road seeds per generation**, so it cannot
memorise one track, and the winner is chosen on a held-out set of roads it never saw.
Fitness is RMS lane deviation, penalised for lane exits, crashes and jerky steering,
and rewarded for staying alive.

## What gets measured

`python main.py evaluate` reports, averaged over seeds:

| metric | meaning |
|---|---|
| mean/max abs deviation | distance from lane centre, metres |
| lane exits | crossings of the lane edge |
| crash rate | fraction of runs ending in a collision |
| steer effort | total steering change, a smoothness proxy |

Three controllers are compared: the trained fly, the fly with untrained default
parameters, and a PID baseline that cheats by reading the true vehicle state.

The ablation study silences parts of the pathway one at a time — H2 left, H2 right,
DNa02 right, the motion channel, the positional channel. This is the part that shows
the connectome is doing real work rather than decorating a hand-written controller: if
silencing H2 changed nothing, the fly would not be driving.

## Honest limits

- The eye is 1-D and low resolution. There is no elevation, no texture, no colour.
- Throttle and braking are conventional, not neural. Only steering comes from the fly.
- H2 in the real animal is one cell per side; here each population is a small group of
  LIF units so the firing rate is not quantised to a few spikes.
- The PID baseline gets ground-truth state, so it should win. The interesting result is
  how close a connectome-constrained spiking network gets using vision alone.

## Layout

```
flydrive/
  connectome.py   wiring, CSV loader, normalised weights
  vision.py       retina + Reichardt motion detectors
  snn.py          LIF populations, free parameters, ablations
  env.py          road, vehicle dynamics, obstacles, metrics
  controllers.py  fly and PID controllers, rollout, fitness
  train.py        cross-entropy method training loop
  evaluate.py     controller comparison and ablation study
  viz.py          figures
main.py           command line
artifacts/        trained_params.json, evaluation.json, figures/
```

## Dashboard

`flydrive_dashboard.html` is a standalone, self-contained live view of the same model
running in the browser — road, spike raster, connectome graph, motor gauges, ablation
switches. Open it directly, no server needed. To drive it from this Python model
instead of its built-in one, stream state over a websocket and replace its `step()`.
