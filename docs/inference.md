# Placebo diagnostics and uncertainty

`Estimate.att` is a point estimate. No standard error is derived by treating
serially correlated time observations as independent.

```python
from pysynth import SyntheticControl, placebo_test

diagnostic = placebo_test(SyntheticControl(), panel, max_pre_rmse_ratio=5)
print(diagnostic.p_value)
print(diagnostic.table)
```

Each original donor becomes pseudo-treated in turn. The actually treated
unit is excluded from every placebo donor pool, as is the pseudo-treated
unit itself. The estimator is refitted and any failure is raised. The same
treatment time is used. Models with a fixed custom Dataprep are not suitable
for this generic helper; use a method that can fit each new Panel.

The statistic is post-RMSPE divided by pre-RMSPE. The optional cutoff excludes
placebos whose pre-RMSE exceeds the given multiple of the treated pre-RMSE.
The treated fit is always retained. With $B$ eligible placebo statistics,
the reported rank is

$$p=\frac{1+\#\{\text{placebo statistic}\geq\text{treated statistic}\}}{1+B}.$$

Ties count in the tail. A zero pre-gap with a nonzero post-gap gives an
infinite statistic; both gaps zero gives zero, using a numerical threshold
of $10^{-12}$ in outcome units. Inspect the table when fit is effectively
perfect or when no placebo passes the cutoff (then the rank is one).

An assignment-based interpretation needs exchangeability and a credible
donor pool. The rank is not automatically a causal p-value in an observational
study. The cutoff changes the reference set and should be chosen transparently.

This generic placebo diagnostic is **not** R synthdid's placebo variance
estimator. PySynth 0.1 does not implement synthdid's three variance procedures,
conformal inference as a common API, time-block resampling, or survey-design
variance. Selected backend methods expose additional procedures on their
native fitted object; consult their assumptions and documentation.

`IndividualSC(n_bootstrap=..., random_state=...)` and
`TwoStageIndividualSC(n_neighbors=..., n_bootstrap=..., random_state=...)`
return donor-resampled ATT draws in metadata and average the fitted paths.
ISC resamples its full eligible pool; 2SISC resamples the selected nearest
donors without repeating screening. Those draws measure conditional donor
sensitivity and omit selection uncertainty. They are not confidence
intervals, and shared donors can induce dependence between individual fits.
