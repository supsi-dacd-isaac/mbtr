# MBTR benchmarks

The benchmark suite uses [ASV](https://asv.readthedocs.io/) to keep
commit-to-commit timing and memory results.  Benchmarks change one input axis
at a time around the representative baseline defined in `common.py`; this
avoids an impractically large Cartesian product while still exposing scaling
regressions.

The default `quick` profile keeps local discovery and smoke runs inexpensive:

```bash
python -m pip install -e ".[benchmark,fast]"
asv run --quick
```

The scheduled performance job should select the full matrix:

```bash
MBTR_BENCHMARK_PROFILE=full asv run
```

Run benchmarks on an otherwise idle, fixed-frequency runner. Record CPU,
memory, operating system, Python, NumPy, SciPy, accelerator version, thread
count, and commit in published results. Set BLAS and accelerator thread counts
explicitly when comparing commits.

The weekly GitHub-hosted workflow retains JSON samples and an HTML report as
trend evidence. Hosted-runner timings are not a hard gate: reproduce suspected
regressions on the same controlled machine before making backend decisions.

The suite reports cold import, first and warm fits, ordinary and linear-leaf
fits, batch prediction, leaf application, peak memory, and serialized model
size. The NumPy engine is the correctness/performance reference. Accelerated
engines should be compared using the same data, bins, dtype, and thread count.
Inputs are currently normalized to float64 by the estimator, so dtype cases
measure conversion overhead rather than native float32 kernel execution.

`bench_real_data_gate.py` is the sequential acceptance gate used for tree
builder optimizations. It uses prepared RF1 and SCM20D fold 0, exactly 100
trees by default, fixed leakage-safe response columns, and reports fit time,
RMSE, engine, tree count, and prediction hashes as each stage finishes. It can
save baseline predictions and compare later implementations numerically:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m benchmarks.bench_real_data_gate \
  --n-estimators 100 --repeat 3 \
  --save-predictions /tmp/mbtr-gate-predictions.npz
```

Set `--variant standard` or `--variant linear` for a focused gate. Optional
`--output-sketch-size` benchmarks approximate top-gradient-output split search;
it must not be selected using test metrics.

`bench_linear_response_scaling.py` is the focused guard for the feature-selected
study. It fixes every tree at depth 3 and varies rows, split columns, outputs,
bins, response-design columns, and tree count one axis at a time. Both NumPy
and warm Numba fits are timed for regression and upper-bound linear-logistic
leaves, and the response-width axis also records peak memory. Small classifier
cases compare the shared ``1/4`` curvature bound with exact per-label
Hessian-weighted Grams; the exact matrix is intentionally bounded because its
cost scales with label count and squared design width. Keeping the topology
fixed prevents a timing change from being confused with a different number of
fitted nodes.
