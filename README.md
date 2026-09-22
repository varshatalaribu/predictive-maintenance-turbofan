# Predictive Maintenance: Turbofan RUL Model

Predicting Remaining Useful Life (RUL) for aircraft turbofan engines using
NASA's C-MAPSS (FD001) dataset, and translating those predictions into a
maintenance-scheduling decision with a cost-tradeoff simulation.

## The business problem

Three maintenance strategies, compared:

| Strategy | How it works | Downside |
|---|---|---|
| Reactive | Fix it after it breaks | Unplanned downtime, most expensive failure mode |
| Preventive | Service on a fixed schedule | Wastes good remaining life; still misses early failures |
| Predictive | Service based on a model's predicted remaining life | Needs a trustworthy model and a decision rule |

This project builds the predictive option: a regression model that estimates
how many operating cycles an engine has left, evaluated on NASA's held-out
test set and then wired into a simple cost simulation to decide *when* to act
on a prediction.

## Data and approach

**Dataset**: NASA C-MAPSS FD001 — 100 training engines and 100 test engines,
each simulated to failure (train) or truncated partway through life (test),
with 21 sensor readings and 3 operational settings per cycle. Source:
`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.

**Label**: Remaining Useful Life = `max_cycle_for_this_engine - current_cycle`,
computed per engine. The label is capped ("clipped") at 125 cycles — early in
an engine's life RUL is not meaningfully predictable from sensor data alone,
so capping keeps the model from wasting capacity trying to distinguish
"250 cycles left" from "300 cycles left" instead of focusing on the window
that matters operationally, near failure.

**Features**: raw sensor readings, plus per-engine rolling statistics
(5-cycle and 20-cycle rolling mean, rolling std, rolling min/max), a
5-cycle rate-of-change, and each sensor's deviation from that engine's own
first-5-cycle baseline. 7 constant/non-informative columns were dropped.
None of the engineered features use information from an engine's own final
cycle or max cycle — that would leak the label.

**Model**: XGBoost gradient-boosted trees, tuned via `RandomizedSearchCV`
with `GroupKFold` cross-validation (engines never split across folds, so no
engine's cycles appear in both train and validation).

**Evaluation methodology**: two numbers are tracked for every model —
MAE across all rows, and MAE restricted to the "decision zone" (true
RUL ≤ 30 cycles), since that is the window where a real maintenance
decision actually gets made. The two do not always move together, and the
decision-zone number is the one that matters.

## Model performance

| Model | All-engine MAE | Decision-zone MAE | Notes |
|---|---|---|---|
| Linear regression | 13.37 | — | Baseline |
| XGBoost (default/hand-picked params) | 10.49 | 4.89 | On NASA's held-out test set |
| XGBoost (tuned) | — | 4.18 | ~15% improvement in the decision zone |

Hyperparameters were tuned against a custom decision-zone scorer, not plain
MAE, so the search rewards accuracy where it is operationally useful rather
than accuracy on engines with hundreds of cycles left. Best hyperparameters
are saved in `data/best_hyperparams.json`.

## Cost-tradeoff analysis

A MAE number doesn't say when to schedule maintenance. So predictions were
run through a simulation: sweep a trigger threshold (predicted RUL) from 1
to 120 cycles, and for each of the 20 validation engines, price the outcome
of scheduling maintenance the first time predicted RUL drops to or below
that threshold.

Assumed costs: $2,000 for a planned visit, plus $35 for every cycle of
remaining life still on the clock when the visit happens (waste), or
$10,000 if the engine is never caught before failure. A fully reactive
strategy (no model) costs $10,000 × 20 = $200,000.

The resulting cost-vs-threshold curve is U-shaped: trigger too late (low
threshold, waiting for near-certainty) and engines get missed, paying the
failure cost; trigger too early (high threshold, acting on the first hint
of wear) and good engine life gets wasted. The minimum lands at a threshold
of **6 cycles**: all 20 engines caught, 5.5 cycles average lead time before
failure, total cost **$43,850** — a **78% reduction** from the $200,000
reactive baseline. See `plots/cost_tradeoff.png` and
`data/cost_tradeoff_results.csv` for the full curve.

**Caveat**: the dollar figures above are assumptions used to explore the
shape of the tradeoff, not measured costs from a real maintenance budget.
The existence of an interior optimum is a robust finding; the exact
threshold (6 cycles) depends on those inputs.

## Key findings and lessons learned

- **The RUL cap value barely matters, as long as one exists.** Caps of 100,
  110, 125, 140, and 160 all scored similarly on decision-zone MAE; removing
  the cap entirely measurably hurt accuracy near failure.
- **All-rows MAE and decision-zone MAE don't always move together** — this
  showed up both in the RUL-cap comparison and in hyperparameter tuning.
  Optimizing and evaluating against the metric that matches the business
  question is more important than optimizing the metric that's easiest to
  compute.
- **Engine-grouped splitting is non-negotiable.** Any row-based split leaks
  near-identical adjacent cycles of the same engine across train/validation,
  producing validation scores that look great and mean nothing.
- **Hyperparameter tuning delivered a real, if modest, gain**: ~15% lower
  decision-zone MAE on the held-out NASA test set.

## Limitations and next steps

- Cost figures are assumptions, not measured facts — re-run
  `scripts/11_cost_tradeoff.py` with real numbers before using this to
  inform an actual maintenance schedule.
- Trained and evaluated only on FD001 (one operating condition, one fault
  mode). FD002–FD004 are harder variants and a natural next step.
- LightGBM/CatBoost as alternative models, and an LSTM/sequence model as a
  more ambitious follow-up, were discussed but not implemented.
- No anomaly-detection layer — this model assumes normal degradation and
  would not flag an unusual failure mode outside its training distribution.
- Validation is a single 20-engine holdout; averaging across several random
  splits (or extending `GroupKFold` to every evaluation) would tighten the
  estimates.

## Project structure

```
PredictiveMaintenance/
├── FD001/                  # single operating condition, single fault mode
│   ├── data/                # raw + processed CSVs, tuned hyperparameters, cost results
│   ├── plots/                # sensor trends, RUL curve, cost-tradeoff plot
│   └── scripts/               # 01_load_explore.py ... 11_cost_tradeoff.py, run top to bottom
├── FD002/                  # 6 operating conditions, single fault mode
│   ├── data/
│   ├── plots/
│   └── scripts/
├── venv/                   # shared Python environment for all datasets
├── requirements.txt
└── README.md
```

Each dataset's scripts are self-contained and run from inside their own
`scripts/` folder (they reference `../data/` and `../plots/` relatively),
sharing the one project-level `venv/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, for a given dataset, run the scripts inside its own `scripts/`
folder in numeric order (each one reads the previous script's output from
that dataset's `data/`), e.g.:

```bash
cd FD001/scripts   # or FD002/scripts
python3 01_load_explore.py
```

