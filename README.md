# Predictive Maintenance: Turbofan RUL Model

Predicting Remaining Useful Life (RUL) for aircraft turbofan engines using
NASA's C-MAPSS dataset, and translating those predictions into a
maintenance-scheduling decision with a cost-tradeoff simulation. Built
dataset by dataset, starting with the simplest case (FD001) and working up
to harder variants (FD002, and eventually FD003/FD004).

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

**Evaluation methodology (both datasets)**: two MAE numbers are tracked for
every model — across all rows, and restricted to the "decision zone" (true
RUL ≤ 30 cycles), the window where a real maintenance decision actually gets
made. The two don't always move together, and the decision-zone number is
the one that matters. Engines are always split by unit, never by row, so no
engine's cycles appear in both train and validation/test.

---

## FD001: single operating condition, single fault mode

**Dataset**: 100 training engines, 100 test engines, each simulated to
failure (train) or truncated partway through life (test), 21 sensors + 3
operational settings per cycle, one flight regime for the whole dataset.
Source: `FD001/data/train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.

**Label**: RUL = `max_cycle_for_this_engine - current_cycle`, capped at 125
cycles — early in an engine's life RUL isn't meaningfully predictable from
sensor data alone, so capping stops the model from wasting capacity on
"250 cycles left" vs. "300 cycles left" instead of the window that matters.

**Features**: raw sensor readings plus per-engine rolling statistics
(5/20-cycle rolling mean, rolling std, rolling min/max), 5-cycle
rate-of-change, and deviation from each engine's own first-5-cycle
baseline. 7 constant/non-informative columns dropped.

**Model**: XGBoost, tuned via `RandomizedSearchCV` + `GroupKFold`
cross-validation, scored on a custom decision-zone metric rather than plain
MAE.

**Model performance** (NASA's held-out test set):

| Model | All-engine MAE | Decision-zone MAE | Notes |
|---|---|---|---|
| Linear regression | 13.37 | — | Baseline |
| XGBoost (hand-picked params) | 10.49 | 4.89 | |
| XGBoost (tuned) | — | 4.18 | ~15% improvement in the decision zone |

**Cost-tradeoff analysis**: sweeping a maintenance-trigger threshold
(predicted RUL) from 1–120 cycles against assumed costs ($2,000/planned
visit, $10,000/failure, $35 per cycle of wasted remaining life) produces a
U-shaped cost curve. Minimum at threshold = **6 cycles**: all 20 validation
engines caught, 5.5 cycles average lead time, total cost **$43,850** — a
**78% reduction** from the $200,000 fully-reactive baseline. See
`FD001/plots/cost_tradeoff.png` and `FD001/data/cost_tradeoff_results.csv`.

---

## FD002: 6 operating conditions, single fault mode

**What's different from FD001**: engines cycle through 6 distinct
combinations of altitude/Mach number/throttle (`setting_1/2/3`) instead of
one fixed regime — the same raw sensor reading can be normal under one
condition and abnormal under another, so a value can't be interpreted
without knowing which regime produced it (think: engine temperature reading
different but equally healthy values at cruise vs. climbing vs. idle).

**Dataset**: 260 training engines, 259 test engines. Source:
`FD002/data/train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`.

**Operating-condition normalization**: rows are clustered into 6 operating
conditions via `KMeans` on `setting_1/2/3`, then every sensor is converted
to "how many standard deviations from normal-for-this-condition" using
per-condition mean/std computed from training data only. The rolling-window
feature recipe from FD001 is then applied on top of these normalized
values, and operating condition is one-hot encoded as 6 additional
features. 4 sensors that showed zero variance within any single condition
(`sensor_1, sensor_5, sensor_18, sensor_19`) were dropped — a different,
narrower list than FD001's, since `setting_3` and two sensors that were
constant in FD001 (`sensor_10, sensor_16`) turned out to carry real signal
here.

**Model**: XGBoost. Hyperparameter tuning was attempted (`RandomizedSearchCV`
+ `GroupKFold`, same as FD001) but **the tuned model was rejected** — it
scored better under cross-validation (decision-zone MAE 4.84 → 4.68) but
*worse* on the real held-out test set (3.70 → 4.32), and its training-set
MAE dropped sharply (9.31 → 5.52) while test performance didn't improve, a
clear overfitting signature: the tuned parameters (deeper trees, more
trees, no feature subsampling) gave the model enough capacity to fit
training-engine-specific quirks rather than general wear patterns. The
final model uses the original hand-picked hyperparameters instead — a real
held-out test set overrules a CV score when they disagree.

**Model performance** (NASA's held-out test set):

| Model | All-engine MAE | Decision-zone MAE | Notes |
|---|---|---|---|
| Linear regression | 12.70 | — | Baseline, already better than FD001's linear baseline |
| XGBoost (hand-picked params) | 17.68 | 3.70 | **Final model** |
| XGBoost (CV-tuned) | 17.69 | 4.32 | Rejected — overfit, see above |

Worth noting: all-engine MAE is *worse* than FD001's (17.68 vs. 10.49) but
decision-zone MAE is *better* (3.70 vs. 4.18/4.89) — FD002 engines simply
run far longer overall (max observed RUL 377 vs. FD001's shorter runs), so
there's more "doesn't matter operationally" room for the model to be
imprecise in, inflating all-engine MAE without affecting the number that
actually matters.

**RUL cap**: re-tested for FD002 the same way as FD001 (candidate caps
30–200). On raw decision-zone MAE, lower caps looked dramatically better
(cap=30 gave 3.38 vs. cap=125's 5.02) — but this was largely an artifact of
training the model to ignore everything outside the window it's scored on
(cap=30 equals the decision-zone boundary itself). Re-running the actual
cost-tradeoff simulation at several candidate caps showed the dollar
outcome barely moves (77.2%–78.0% savings across caps 30/40/50/125,
$87,800–$91,300 total cost) — confirming the earlier lesson that the RUL
cap doesn't meaningfully affect real operational outcomes, and MAE alone
can be a misleading way to choose it. Cap = 125 was kept for consistency
with FD001.

**Cost-tradeoff analysis**: same simulation as FD001, same assumed costs.
Minimum at threshold = **8–9 cycles** (varies slightly run to run):
all 40 validation engines caught, ~7 cycles average lead time, total cost
**~$90,000** — a **~77% reduction** from the $400,000 fully-reactive
baseline (40 validation engines × $10,000). See `FD002/plots/cost_tradeoff.png`
and `FD002/data/cost_tradeoff_results.csv`.

---

## Key findings and lessons learned

- **All-rows MAE and decision-zone MAE don't always move together** — the
  central lesson of this project, confirmed independently in FD001 (RUL cap
  comparison, hyperparameter tuning) and FD002 (RUL cap comparison, and the
  general gap between the two metrics being much larger than in FD001).
  Optimize and evaluate against the metric that matches the actual business
  question, not whichever is easiest to compute.
- **A better cross-validation score doesn't guarantee a better real-world
  model.** FD002's hyperparameter tuning is a concrete example: CV improved,
  the real test set got worse, and the training-fit-vs-test-performance gap
  made the overfitting visible and explainable, not just a mystery drop.
- **The RUL cap value barely matters once you measure the right thing.**
  True in both datasets, for different reasons: FD001 showed any reasonable
  cap works about the same; FD002 showed that even a cap that looks much
  better on MAE can turn out to barely matter — or actively mislead — once
  translated into actual dollar cost.
- **Engine-grouped splitting is non-negotiable.** Any row-based split leaks
  near-identical adjacent cycles of the same engine across train/validation,
  producing validation scores that look great and mean nothing.
- **Feature engineering has to be re-earned per dataset, not copy-pasted.**
  FD001's "constant column" drop list didn't transfer directly to FD002 —
  two sensors that were dead in FD001 turned out to be informative in FD002
  once operating-condition noise was accounted for, and vice versa for
  `setting_3`.

## Limitations and next steps

- Cost figures (both datasets) are assumptions, not measured facts — re-run
  the `cost_tradeoff.py` script for the relevant dataset with real numbers
  before using this to inform an actual maintenance schedule.
- FD003 (2 fault modes, 1 operating condition) and FD004 (2 fault modes, 6
  operating conditions) are the remaining C-MAPSS subsets — natural next
  steps, especially FD004 as the hardest variant combining both challenges.
- LightGBM/CatBoost as alternative models, and an LSTM/sequence model as a
  more ambitious follow-up, were discussed but not implemented.
- No anomaly-detection layer — both models assume normal degradation and
  wouldn't flag a failure mode outside their training distribution.
- Validation is a single random holdout per dataset; averaging across
  several random splits (or extending `GroupKFold` to every evaluation)
  would tighten the estimates.

## Project structure

```
PredictiveMaintenance/
├── FD001/                  # single operating condition, single fault mode
│   ├── data/                # raw + processed CSVs, tuned hyperparameters, cost results
│   ├── plots/                # sensor trends, RUL curve, cost-tradeoff plot
│   └── scripts/               # 01_load_explore.py ... 11_cost_tradeoff.py, run top to bottom
├── FD002/                  # 6 operating conditions, single fault mode
│   ├── data/                 # raw + processed CSVs (generated feature CSVs are gitignored, see below)
│   ├── plots/
│   └── scripts/               # 01_load_explore.py ... 10_cost_tradeoff.py, run top to bottom
├── venv/                   # shared Python environment for all datasets
├── requirements.txt
└── README.md
```

Each dataset's scripts are self-contained and run from inside their own
`scripts/` folder (they reference `../data/` and `../plots/` relatively),
sharing the one project-level `venv/`. Large generated feature CSVs
(`*_features.csv`, `*_with_RUL.csv`, `*_with_clusters.csv`) are excluded
from git — they're fully reproducible by running the scripts in order, and
some exceed GitHub's file-size limits. Small result artifacts (tuned
hyperparameters, cost-tradeoff results, normalization stats) are kept since
they document a specific run's output.

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
