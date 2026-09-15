# Assignment 1 — lexicase vs tournament parent selection

**Research question.** When one robot body must match five target bodies of different sizes,
does selecting parents one target at a time (lexicase) instead of on the combined
mean + std score (tournament) change what the EA finds — population diversity and bodies
specialised to single targets — and does that help or hurt the combined fitness?

Only parent selection differs between conditions. Everything else is fixed: tree
encoding, 20-module budget, population 100, 100 generations (10,100 evaluations),
subtree crossover (p = 0.7), one random ariel tree mutation per child, generational
replacement with one elite.

All commands below are run from the **repository root**.

## Files

| File | Purpose |
|---|---|
| `tree_edit_distance.py` | Fitness metric (course-provided, unchanged) |
| `common.py` | Targets, fitness evaluation, seeding, random bodies, diversity, CSV log format |
| `selection.py` | `tournament` and `lexicase` parent selection — the aspect we study |
| `ea_tree.py` | The EA, built on `ariel.ec.EA` |
| `random_search.py` | Random-search baseline with the same evaluation budget |
| `calibrate_k.py` | Picks the tournament size whose selection strength matches lexicase |
| `run_all.sh` | Final experiments: 10 seeds × 4 conditions |
| `analysis.py` | Figures and statistics for the report |
| `sanity_checks.py` | Automated checks of the code; run after every change |
| `A1_template_2026.py` | Course template (reference only) |

## Workflow

```bash
# 0. Check the code (< 1 minute)
uv run python assignments/assignment_1/sanity_checks.py

# 1. Choose k for the matched-strength tournament control (~1 minute)
uv run python assignments/assignment_1/calibrate_k.py

# 2. Final experiments (~10 minutes); use the k printed by step 1
K_CONTROL=30 bash assignments/assignment_1/run_all.sh

# 3. Figures and statistics
uv run python assignments/assignment_1/analysis.py
```

A single run, e.g. for debugging:

```bash
uv run python assignments/assignment_1/ea_tree.py --selection lexicase --seed 0
uv run python assignments/assignment_1/ea_tree.py --selection tournament --k 2 --seed 0 --pop-size 20 --generations 10
```

## Outputs

```
results/
  tournament_k2/seed_00/   config.json  history.csv  best_body.json  specialists.json  database.db
  lexicase/seed_00/        …
  tournament_k30/seed_00/  …                (k from calibrate_k.py)
  random/seed_00/          config.json  history.csv  best_body.json
  _calibration/            calibration.csv  calibration.json
figures/
  fitness_curve.pdf  diversity_curve.pdf  selection_strength.pdf
  closest_per_target.pdf  final_fitness.pdf  summary.csv  stats.csv
```

`database.db` (ariel's full record of every individual, ~12 MB per run) is not committed;
`run_all.sh` regenerates it.

### `history.csv` columns (one row per generation)

| Column | Meaning |
|---|---|
| `generation`, `evaluations` | Generation 0 is the initial population; evaluations = (generation + 1) × population size |
| `best_fitness`, `mean_fitness`, `std_fitness` | Fitness = mean + std of the tree edit distance to the 5 targets (lower is better). For random search, `best_fitness` is the best found so far |
| `best_size`, `mean_size` | Nodes per body, including the core |
| `diversity` | Mean pairwise tree edit distance between 20 randomly sampled bodies |
| `distinct_parents` | Number of different bodies picked as parents this generation (out of 200 picks) |
| `closest_7` … `closest_25` | Lowest distance of any body in the population to the 7- … 25-node target |
| `best_dist_7` … `best_dist_25` | The best body's distance to each target |

## Reproducibility

- Every run seeds `random` and `numpy` from `--seed`; the same seed gives an identical `history.csv` (checked by `sanity_checks.py`).
- Final experiments use seeds 0–9; calibration uses seeds 100–102.
- Diversity is measured with a separate random generator, so logging never changes a run.
- Starting bodies (and random-search bodies) have a module count drawn uniformly from 1–20 (`--init-size uniform`, the default).
- The ariel framework (`src/ariel`) is not modified.
