# Getting started

Install from the repository with Python 3.10+:

```bash
python -m pip install -e '.[backends,plot]'
```

For academic notebooks use the pinned environment described in the repository
README. No PyPI publication of this project is assumed.
SCUL and ISC use the original R implementations; follow the
[reference-backend installation guide](reference_backends.md) before fitting them.

```python
from pysynth import SyntheticControl, SyntheticDID, compare
from pysynth.datasets import DATASETS, load_dataset

frame = load_dataset("california")
panel = DATASETS["california"].panel(frame)
sdid = SyntheticDID().fit(panel)
scm = SyntheticControl().fit(panel)
print(compare({"SDID": sdid, "SCM": scm}))
sdid.plot(effects=True)
```

`fit` returns an `Estimate`; native estimators do not require a separate
prediction call. `observed`, `counterfactual`, `effects`, `weights`, and
`time_weights` are labelled pandas objects. `att` and `pre_rmse` are floats.
`metadata` contains tuning and convergence diagnostics. `to_frame()` exports
the full paths, and `.to_csv(...)` can save that DataFrame.

Plotting lazily imports matplotlib. Numerical estimation works without the
plotting extra. Upstream dependencies are also imported only when used.

A negative ATT means the observed outcome is lower than the estimated
untreated outcome. For log outcomes, apply `100 * numpy.expm1(gap)` for the
corresponding percentage gap; this is not the same as averaging individual
percentage effects. `compare` reports estimates and fit diagnostics and does
not select a causally preferred estimator.
