# GA4XGB

Reference implementation for the paper:

> **GA4XGB: An Effective Student Performance Prediction System Through Optimizing
> XGBoost Hyperparameters Using Genetic Algorithm With High-School Transcript Data**
> Nguyen-Ba-Tien, Ngo-Thi-Thu-Trang, Ha-Nam Nguyen.

GA4XGB is a genetic algorithm that tunes XGBoost hyperparameters under a
**tail-weighted fitness function**, so that the search is not dominated by the dense
centre of a concentrated target distribution.

## The problem

When a regression target is concentrated, an optimizer driven by an average error
metric spends its budget where error is cheapest to reduce. On our data, random search
under a fixed budget improves the dense middle of the score distribution by 4.4% while
leaving the low-score region **0.7% worse than an untuned model** — and still reports a
3.2% aggregate improvement. The optimizer is faithfully serving the objective it was
given; the objective is the problem.

## The fitness function

The score range is partitioned into bins, and per-bin mean squared errors are combined
under weights that flatten with a single coefficient α:

```
F_tail = sqrt( Σ_b w_b · MSE_b ),    w_b = n_b^(1-α) / Σ_j n_j^(1-α)
```

- `α = 0` → weights are proportional to bin population; `F_tail` **is** ordinary RMSE
  (the *micro* average).
- `α = 1` → every bin weighs `1/B` regardless of population (the *macro* average).
- Intermediate values interpolate continuously.

An optional eighth gene `β` lets the GA also search density-inverse **loss** weighting
`w_i ∝ n_b(i)^(-β)` inside XGBoost's training objective. The paper's central empirical
finding is that the two levels are coupled: scored by plain RMSE the search drives
β → 0.03 (i.e. rejects loss weighting), while under the tail-weighted objective it
drives β → 1.0.

`α` and `β` are a documented trade-off, not a free win. Turning them up improves the
tails and degrades the aggregate; see the frontier in Fig. 6(c) of the paper.

## Layout

| Path | What it is |
|---|---|
| `src/ga_xgb.py` | The GA: real-coded chromosome, BLX-α crossover, Gaussian mutation, tail-weighted fitness, optional β gene |
| `src/bins.py` | Bin edges and the low/middle/high tail regions — **the single source of truth** for every reported region |
| `src/preprocess.py` | Transcript → 186 numeric features |
| `src/search_baselines.py` | Grid and random search under a matched evaluation budget |
| `src/baseline.py` | Default-configuration XGBoost reference |
| `src/bootstrap_test.py` | Paired bootstrap over test students (10,000 resamples) |
| `src/paper_numbers.py` | Emits every number quoted in the paper into `results/paper_numbers.json` |
| `src/figures.py` | Figures 3–7 |
| `results/` | All tuned configurations and aggregate metrics reported in the paper |
| `run_experiments.sh` | Runs the whole pipeline in order |

`src/build_docx.py`, `src/docx_ieee.py`, `src/mathrun.py` and `src/build_refs.py` build
the IEEE Access manuscript from Markdown; they are unrelated to the method and are kept
only so the results in the paper are traceable to the code that produced them.

## Data availability

**The dataset is not in this repository and cannot be distributed by us.**

The experiments use administrative examination and transcript records held by the
Institute of Digital Education and Testing, Vietnam National University, Hanoi (VNU).
The records were released to the research team in de-identified form under a written
data-use authorization whose terms prohibit transfer to any third party without prior
written approval from the Director of the Institute.

Requests for access should be directed to the Institute of Digital Education and Testing,
VNU (idt@vnu.edu.vn). Aggregated summary statistics sufficient to reproduce the reported
tables and figures are in `results/`, and are available from the corresponding author on
reasonable request.

To run the code on your own data, provide a CSV at `data/data_final.csv` with the column
structure documented in `src/preprocess.py` and the target score in the final column.

## Reproducing

```bash
pip install -r requirements.txt
./run_experiments.sh
```

Versions used in the paper: scikit-learn 1.8, XGBoost 3.2, Python 3.12. A full run takes
roughly 15 hours on a 10-core CPU; individual searches take 1.7–3.3 hours each.

**Every search reported in the paper is single-seed.** The paired bootstrap quantifies
uncertainty over test-set composition, not run-to-run variability of the stochastic
searches. This is stated in the paper (Sections IV-H and V) and repeated here so nobody
reads the numbers as more stable than they are.

## Citation

Bibliographic details will be added once the paper is published.

## License

MIT — see `LICENSE`.
