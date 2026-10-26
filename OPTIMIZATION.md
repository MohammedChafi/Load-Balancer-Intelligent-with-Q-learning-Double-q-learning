# Parameter optimization section

Section 4 is appended to `static/index.html`. The original sections 0–3 and
their inline JavaScript remain unchanged. Styles are scoped to `#optimization`;
the frontend is in `static/optimization.js`.

The only change to the original backend is importing and calling
`install_optimization(app, ROOT)` in `app/api.py`. The search service lives in
`app/optimization.py`; its routes live in `app/optimization_api.py`. No extra
package installation is required.

## Search methods

- Grid: Cartesian product of logarithmically spaced eta and linearly spaced gamma.
- Random: independent uniform draws in the transformed two-dimensional space.
- Bayesian: Gaussian-process surrogate with an RBF kernel (length .25),
  standardized scores, repeat-based observation variance, and expected
  improvement. Five initial random trials; subsequent suggestions maximize EI
  over 384 random acquisition candidates. At least six trials are required.
- Latin hypercube: one randomized sample per stratum along each transformed axis.
- Hyperband: at most three successive-halving brackets, reduction factor three.
  Canonical bracket population weights 3^s/(s+1) are scaled to the selected total
  candidate cap. Initial budgets are about the full budget, full/3 and full/9,
  with a minimum of about 50 episodes. At least 150 full-budget episodes are
  required. Survivors resume their existing tables and RNG streams. This is a
  finite-candidate-budget Hyperband variant, not an unconstrained paper benchmark.

## Comparison protocol

Every candidate uses the original QLearning or DoubleQLearning agent and the
original 200-request episode simulator. Learning updates are not duplicated.
Each algorithm is optimized separately. Candidate repeats have independent
training streams; all parameter pairs share those streams for fair comparisons.
Search, training, validation and final-test streams have disjoint namespaces.

The full training budget determines the epsilon schedule, including for every
Hyperband checkpoint. Validation uses clones, so neither training tables nor
training RNG state are modified by intermediate evaluations.

The maximized objective is mean undiscounted reward per request, with equal
weight for nominal traffic settings .4, .7 and .9. The default eta=.1/gamma=.9
is freshly trained at the same budget and participates in selection. Only
full-budget results can be selected. The final test uses fresh traffic seeds,
epsilon zero, and no learning; final-test performance does not reselect a pair.

Global variability is the sample standard deviation across independently trained
replicates, each evaluated on common scenarios. Scenario summaries in JSON also
include variability across individual scenario/replicate observations. One
replicate cannot establish training variability. Latencies describe predicted
latency among admitted requests; rejection rates include immediately rejected
requests. No claim of universal optimality is made.

## Persistence and operation

The original `models/qtables.json` is never written by this section. Each search
stores its report and all selected model replicates under `models/optimization/`.
`last.json` restores the latest results after a restart. Export includes the
configuration, evaluated trials, checkpoints, aggregate learning curves,
validation/test metrics, protocol and selected tables. Search state is saved at
completion or cancellation; an interrupted process does not resume training.

Routes: `POST /api/optimization/start`, `POST /api/optimization/stop`,
`GET /api/optimization/status`, `GET /api/optimization/report`.

Run the existing project as usual. Run the full checks with
`python -m unittest discover -s tests -v`.
