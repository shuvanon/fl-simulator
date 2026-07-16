# Mini Federated Learning Simulator

A tiny, dependency-free, **visual demo of Federated Learning (FL)** that runs entirely in the
browser. Several "hospitals"/clients each train their own small neural network on their *own*
local data; a central server averages those models into a single global model — without any raw
data ever leaving a client. You tune the scenario and watch how per-client and global accuracy
respond.

> The training is **real**: a small MLP trained with hand-written gradient descent in plain
> JavaScript. No ML libraries, no network calls, no faked numbers.

📘 **New to federated learning, or want the full technical/teaching write-up?** See
**[`REPORT.md`](REPORT.md)** — how it works, the data and math, a beginner step-by-step example,
and an honest comparison with real FL systems.

## Run it

Just open **`index.html`** in any modern browser (double-click, or drag it into a tab). It runs
fully offline — there is nothing to install or build.

## What you can control

| Control | Effect |
| --- | --- |
| **Hospitals / clients** (2–10) | How many participants train separately each round. |
| **Data imbalance (non-IID)** (0–100) | 0 = every client sees all classes evenly; 100 = each client is dominated by a single class. Implemented as a Dirichlet label-skew split. |
| **Aggregation method** | `FedAvg` (size-weighted average), `Simple mean` (unweighted), or `Coordinate median` (robust to outlier clients). |
| **Local epochs** (1–20) | Training passes each client runs before syncing. High epochs + high imbalance ⇒ *client drift*. |
| **Communication rounds** (1–40) | How many times we train → average → repeat. |

**Reshuffle** draws a fresh random dataset + initialization. Everything is seeded, so the same
settings reproduce the same run.

## What you see

- **Results readout** — `Client k accuracy` for every client + the `Global model accuracy`, plus a
  plain-language **Observation** explaining *why* the run turned out the way it did.
- **Accuracy over rounds** — the global model's accuracy per round (bold) with faint per-client
  lines, so you can watch convergence (or divergence).
- **Data distribution per client** — stacked bars showing each client's class mix and size; the
  clearest picture of how imbalance is spread.
- **Decision boundaries** — one small map per client vs. the aggregated global model. When tiles
  disagree, you are literally looking at client drift.

## How it works (under the hood)

- **Data:** synthetic 2-D, 3-class Gaussian blobs (~600 points). A shared 30% held-out **test
  set** (used for every accuracy number) is split off first; the rest is partitioned across
  clients.
- **Preprocessing (federated):** standardization works the way a real FL system's would — each
  client shares only aggregate moments (count, sum, sum of squares) of its raw training points,
  and the server combines them into a global mean/std that is broadcast back. No raw coordinates
  leave a client, and the test set never influences the stats. (Production systems would protect
  even these aggregates with secure aggregation / differential privacy.)
- **Model:** MLP `2 → 10 (tanh) → 3 (softmax)`, cross-entropy loss, full-batch gradient descent.
- **FL loop:** each round every client copies the global model, trains locally for the chosen
  number of epochs, and the server aggregates the client weights back into the global model.

```
each client → server: (count, Σx, Σx²) of its raw points   # aggregates, never raw data
server → everyone: global mean/std, applied locally
init global model
for each round:
  for each client:
     local = copy(global)
     train local for E epochs on the client's data
  global = aggregate(all local models)   # FedAvg / mean / median
```

## Project layout

Everything lives in a single `index.html`. The `<script>` is split into clearly labelled
sections — `rng · data · model · fl · insights · ui · main` — and all of the simulation logic is
kept **pure and DOM-free**, so only the `ui` section needs rewriting to later embed this as a
component in a portfolio website.

## Roadmap / ideas

- Embed as an interactive component on the portfolio site.
- Real datasets (e.g. downsampled MNIST) via TensorFlow.js.
- More FL methods: FedProx, secure aggregation, differential privacy.
- Quantity skew and client-participation (stragglers / partial participation).
