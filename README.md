# OrganoidTwin

A small demo exploring one possible approach to real-time monitoring and
adaptive control of lab-grown organoids in a multi-well culture plate — built
entirely on synthetic data, with no external APIs, LLMs, or real hardware
anywhere in the system. This is a first, rough sketch of a project idea, not
a finished or validated system, and none of it has been checked against real
biology.

The general direction: fuse multiple simulated biosensor streams into an
assessment of organoid state and microenvironment condition, use that to
flag trouble early, and see whether acting on it (adjusting media) actually
helps — measured with a controlled experiment rather than assumed.

## How this works

We simulate organoid biology: a metabolic model tracks glucose, oxygen, and
lactate in each well as the organoid grows and feeds over a week-long
culture, occasionally hit by a random contamination or temperature event. We
simulate what noisy, laggy real sensors would report from that biology — pH,
dissolved oxygen, a glucose/lactate proxy, impedance — never handing the
model ground truth directly. A graph neural network fuses those four streams
across all 24 wells (wells that share a plate influence each other) to infer
a continuous health score with an uncertainty estimate, with a hard
constraint baked in that ties its oxygen and lactate predictions together
the way real metabolism does. A Jacobian-sensitivity detector watches for the
moment a well's health prediction becomes unusually sensitive to its recent
readings — meant as an early signature of a regime shift "setting in" — and
fires a warning. A rule-based engine turns that into a suggested action, and
the system can then act on it: a closed-loop control arena runs a controlled
experiment to check whether doing so actually changes the outcome.

## Four additional pieces built on top

1. **Closed-loop control arena** (`backend/control/`, "Control Arena" tab) —
   three arms run under **identical** random seeds (same adverse events, same
   sensor noise) so only the intervention differs: **no control** (passive),
   **AI-driven** (the trained GNN + recommendation engine, sensor data only),
   and an **oracle** upper bound (acts on ground truth). Kaplan-Meier
   survival curves + a log-rank test check whether control makes a measurable
   difference.
2. **Interactive control panel** ("Control Panel" tab) — a live GATv2
   attention graph (which wells currently influence which) and a "what-if"
   slider: preview a hypothetical media adjustment's predicted effect on the
   live twin 12h ahead, before deciding whether to apply it.
3. **Drug-screening dose-response mode** (`backend/analysis/dose_response.py`,
   "Drug Screening" tab) — simulate a toxin dose gradient across plate
   columns and fit an EC50/Hill curve from the model's own inferred readout,
   the kind of curve a pharmacology assay would report.
4. **Real-data ingestion + uncertainty checks** — a CSV upload tab
   (`backend/analysis/csv_ingest.py`) runs real or externally-generated
   sensor readings through the same trained pipeline, degrading gracefully
   for non-standard well layouts. The GNN's MC-dropout uncertainty
   (`backend/gnn/uncertainty.py`) is checked against actual error rather than
   just reported (see below). Every control-arena experiment also persists to
   SQLite (`backend/models/db.py`) so results are queryable history, not a
   single in-memory run.

## What was measured

Everything below is a synthetic-data measurement — the ground truth comes
from the same simulation being evaluated against, so treat these as a check
that the pieces do roughly what they're meant to, not as evidence this would
work on a real culture. Numbers vary somewhat run to run with retraining;
these are one representative pass, reproducible via `eval/run_benchmark.py`
and `eval/uncertainty_calibration.py`.

### Detection (8 held-out plates, 192 wells never seen in training)

| Metric | Result |
|---|---|
| Bifurcation detection recall | 64% of decline events detected |
| Detection lead time | mean 14.3h, median 4.8h before the labeled onset |
| False positive rate | 0.34 firings per healthy well per simulated week |

### Recommendation accuracy

The original hand-tuned sensor-delta heuristic couldn't track how the
dominant limiting factor shifts after decline onset. Replacing it with a
classifier head trained directly on the same ground-truth cause labels did
better, though it's still far from solved:

| | Accuracy |
|---|---|
| Hand-tuned heuristic (original approach) | 29% |
| Learned cause classifier (current) | 48% |
| — oxygen-limited cases specifically | 18% → 82% |

Adverse-event accuracy traded off somewhat in the process (a "grace period"
label-timing fix was tried, found to trade better adverse-event accuracy for
worse glucose accuracy with no net gain, and reverted) — reported in
`eval/results.json`'s confusion breakdown, not smoothed over.

### Uncertainty calibration

Checked, not assumed: does the model's MC-dropout uncertainty actually track
its error?

| | Result |
|---|---|
| Spearman(predicted uncertainty, actual error) | 0.64 (p < 0.0001) |
| Coverage within ±1σ | 58% (target 68%) |
| Coverage within ±2σ | 85% (target 95%) |

The uncertainty is a genuinely useful *ranking* signal (higher-uncertainty
predictions are more often wrong), but under-confident in absolute
magnitude — stated plainly rather than rounded up. See
`eval/uncertainty_calibration.py`.

### Ablations

**Hard consistency constraint**: the constrained model's flux outputs
deviate from the fitted stoichiometric relationship about 3x less than the
unconstrained model's (0.014 vs. 0.041 residual), at a modest cost to raw
accuracy (0.0087 vs. 0.0068 health MSE) — a real trade-off, not a free lunch.

**EWC continual adaptation**: after online fine-tuning on a drifted batch
stream, the model *without* EWC measurably forgot its original task
(+0.000118 reference loss) while the model *with* the EWC penalty didn't
move at all (−0.000018) — see `eval/run_benchmark.py`'s EWC ablation.

### Closed-loop control (8 plates, 192 wells — regenerate via the Control Arena tab)

| Arm | Mean health |
|---|---|
| No control | 0.789 |
| AI-driven control | 0.848 (p = 0.027 vs. no control, log-rank) |
| Oracle (upper bound) | 0.883 |

By root cause, AI-driven control lifts oxygen-limited wells from 0.52 → 0.74
mean health (oracle: 0.88), while the oracle leaves adverse-event
(mitochondrial damage) wells essentially untouched (0.785 → 0.785) — the
built-in negative control: the oracle policy deliberately withholds
intervention from damage no amount of feed/O2 can fix, and the measurement
backs that up.

### Drug-screening dose-response (regenerate via the Drug Screening tab)

Simulated a 6-point dose gradient (0–50, true EC50 = 10) across plate
columns, 3 replicate plates:

| | EC50 | Hill slope | R² |
|---|---|---|---|
| Ground truth | 8.2 | 2.09 | 0.998 |
| Recovered from noisy sensors + GNN alone | 3.9 | 5.16 | 0.937 |

Curve-fit quality survives the pipeline reasonably well; the EC50 point
estimate is biased low, which is honestly attributable to the model never
having seen a drug-perturbation scenario during training rather than
anything more subtle.

## Honest limitations

- Recommendation accuracy (48%) is a real improvement over the original
  heuristic but far from solved — the three decline causes overlap in
  sensor-space more than a single fix resolved.
- Uncertainty is well-ranked but under-confident in absolute magnitude.
- The model is trained entirely on synthetic data; predictions on real data
  uploaded via the Data Upload tab are only as meaningful as how close that
  data's distribution is to what the model was trained on.
- Small control-arena runs (few plates) can show a statistically
  non-significant result by chance — a reflection of limited statistical
  power at low sample size, not a failure of the method.

## Repo layout

```
backend/
  main.py                FastAPI app: live routes + the research-extension endpoints
  config.py               plate size, sensor noise params, paths
  models/                  Pydantic schemas + SQLAlchemy DB (Plate/Well/Reading/Event,
                          Experiment/ArmOutcome for cohort analytics)
  biology/
    metabolic_sim.py       aerobic/Warburg flux model (extends an earlier side project's heuristic)
    organoid_trajectory.py WellSimulator: steppable per-well sim, interventions, drug dose
    decline_dynamics.py    latent stress process + adverse-event injection
  sensors/
    noise_profiles.py      per-sensor noise + response-lag calibration
    sensor_model.py         StreamingSensorState (causal) + batch wrapper
  gnn/
    plate_graph.py          well-to-well plate adjacency graph
    architecture.py          temporal GRU + GATv2 fusion model, dropout, cause classifier, attention extraction
    constraints.py            hard stoichiometric consistency constraint
    train.py                   training + checkpointing + bifurcation threshold calibration
    coevolution.py               EWC continual adaptation
    bifurcation.py                 Jacobian-norm regime-shift detector
    uncertainty.py                 MC-dropout uncertainty
  control/
    policies.py             no-control / oracle / model-driven intervention policies
    closed_loop.py           three-arm plate stepper
    whatif.py                 clone-and-preview hypothetical intervention
  analysis/
    dose_response.py        Hill-curve fitting + dose-response plate runner
    survival.py               Kaplan-Meier + log-rank test
    cohort.py                  experiment persistence + aggregation
    csv_ingest.py               real-data CSV upload + inference
  recommend/engine.py       rule-based recommendations, driven by the learned cause classifier
  explain/narrator.py       plain-language narration (no LLM, template-based)
  ws/plate_stream.py        wires every layer above into one live WebSocket stream
frontend/
  src/App.jsx               tab navigation + landing page toggle
  src/pages/LandingPage.jsx overview page shown before entering the dashboard
  src/tabs/                 Control Arena / Control Panel / Drug Screening / Data Upload
  src/components/charts/    LineChart (+ KM step mode), DoseResponseChart, AttentionGraph
  src/components/           PlateView, WellDetail, AlertFeed, RecommendationPanel,
                            ExplainerPanel, CalibrationPanel, PageIntro, PanelHeader, Tooltip
eval/
  run_benchmark.py          lead-time / false-positive / recommendation-accuracy / ablations
  uncertainty_calibration.py  checks whether predicted uncertainty tracks actual error
  plots.py                    renders eval/plots/*.png from eval/results.json
```

## Running it locally

**Backend** (from repo root — first run trains the model and calibrates the
bifurcation threshold, which takes a few minutes on CPU):

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu -r backend/requirements.txt
.venv/bin/python -m backend.gnn.train                 # trains model_constrained.pt + model_unconstrained.pt
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health` and `curl http://localhost:8000/api/plate`.

**Frontend** (from `frontend/`):

```bash
npm install
npm run dev -- --port 5180
```

Open the printed local URL — it opens on an overview page first, with a
button into the live dashboard. **Live Monitor** replays a simulated 7-day
culture; click a well for its raw streams + narration. **Control Arena**
runs the three-arm experiment on demand (a few seconds for 5 plates).
**Control Panel** lets you click a well and preview a hypothetical
intervention. **Drug Screening** runs a dose-response plate and fits the
curve live. **Data Upload** runs a CSV of sensor readings through the same
pipeline.

If port 8000 or 5173/5180 is already in use, pass a different `--port` to
either command; point the frontend at a non-default backend with:

```bash
VITE_WS_URL=ws://localhost:8010/ws/plate VITE_API_URL=http://localhost:8010 npm run dev -- --port 5180
```

**Evaluation** (from repo root, after training):

```bash
.venv/bin/python -m eval.run_benchmark             # writes eval/results.json (~5 min, CPU)
.venv/bin/python -m eval.uncertainty_calibration   # writes eval/uncertainty_results.json (~30s)
.venv/bin/python -m eval.plots                     # writes eval/plots/*.png
```

## Deploying it

Backend on Render, frontend on Netlify — both free tiers. `render.yaml` and
`netlify.toml` in the repo root record the exact settings used.

**Backend (Render, web service, Python runtime):**
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- The trained checkpoints (`backend/gnn/checkpoints/*.pt`) are committed to
  the repo rather than trained at build time — there's no GPU or spare build
  minutes on a free plan for a several-minute CPU training run on every
  deploy, so the backend just loads the checkpoint that's already there.
- `CORS_EXTRA_ORIGINS` (comma-separated) can add specific allowed origins;
  any `*.netlify.app` origin is already allowed via a regex, so a first
  deploy works without setting this.

**Frontend (Netlify, static site, build from `frontend/`):**
- Build command: `npm run build`, publish directory: `dist` (both picked up
  automatically from `netlify.toml`).
- Set `VITE_API_URL` and `VITE_WS_URL` in the site's environment variables to
  the deployed backend's address, e.g. `https://<your-backend>.onrender.com`
  and `wss://<your-backend>.onrender.com/ws/plate`.

**Caveats, since this is a free-tier deploy of a demo, not production
infrastructure:**
- Render's free plan spins the backend down after inactivity — the first
  request after a while can take up to ~a minute while it wakes back up.
- SQLite (`backend/data/organoid_twin.db`, used by the cohort-analytics
  database) lives on local disk and is **not persisted** across deploys or
  restarts on a free plan — every deploy starts from an empty database. Fine
  for a demo; would need a persistent disk or a managed Postgres otherwise.

## Design notes

**Relationship to an earlier side project.** The metabolic simulation
extends an aerobic/anaerobic flux heuristic prototyped in a separate,
smaller side project (`cell-digital-twin`) — extended here with
temperature/enzyme-activity dependence and a decoupling of oxygen
availability from mitochondrial capacity, so decline can shift metabolism
toward fermentation even when oxygen is available (the Warburg-like
mechanism this project is trying to model). The EWC and Jacobian
bifurcation-detection patterns follow the same general idea but are
implemented fresh here.

**No sparse-graph library.** GATv2 attention is implemented densely over a
24-node adjacency matrix rather than via PyTorch Geometric — the plate is
small enough that this needs no extra native-wheel dependencies.

**Two distinct decline mechanisms, deliberately.** Substrate-limited decline
(glucose or oxygen running low — enzyme activity stays intact, in principle
correctable by feeding more) and adverse-event decline (contamination /
temperature shock — permanently impairs mitochondrial function, not fixed by
feeding or oxygenating more) are modeled as mechanistically separate, which
is what gives the recommendation engine and the closed-loop arena's oracle
policy a correctable-vs-not distinction to work with, and the control-arena
experiment a built-in negative control.

**Bifurcation threshold is calibrated, not adaptive.** A live per-well EWMA
baseline was tried first and found unstable (a well's own short history is
too noisy to characterize "normal"). What worked better is a fixed threshold
calibrated once from Jacobian-norm values observed during known-healthy
stretches across several simulated plates, then firing on consecutive
above-threshold ticks with a cooldown.

**The four extra pieces reuse the same steppable primitives**
(`WellSimulator.step()`, `StreamingSensorState.step()`) as the live
dashboard — the closed-loop arena, what-if preview, and dose-response runner
are all built by driving those same two objects differently, not separate
simulation code paths.
