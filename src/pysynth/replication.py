"""Explicit historical specifications for the empirical replication notebooks.

These helpers encode the predictor choices in the cited upstream examples.
They are kept separate from the common outcome-only comparisons.
"""

from .backends import PySyncon, _require
from .datasets import DATASETS
from .sdid import SyntheticDID


def replicate(name, frame):
    """Fit a documented academic example and return its common Estimate.

    California uses the SDID paper's no-covariate specification. Basque and
    Germany use the pysyncon replication specifications; Kansas uses the R
    augsynth vignette's no-covariate specification. See docs/replication.md
    for exact targets, windows, tolerances and limits on replication claims.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown replication: {name}")
    panel = DATASETS[name].panel(frame)
    if name == "california":
        return SyntheticDID().fit(panel)
    if name == "kansas":
        return PySyncon("augsynth").fit(panel)
    upstream = _require("pysyncon")
    spec = DATASETS[name]
    common = dict(
        foo=frame,
        dependent=spec.outcome,
        unit_variable=spec.unit,
        time_variable=spec.time,
        treatment_identifier=spec.treated_unit,
        controls_identifier=list(panel.units[:-1]),
        predictors_op="mean",
    )
    if name == "basque":
        dp = upstream.Dataprep(
            **common,
            predictors=[
                "school.illit",
                "school.prim",
                "school.med",
                "school.high",
                "school.post.high",
                "invest",
            ],
            time_predictors_prior=range(1964, 1970),
            time_optimize_ssr=range(1960, 1970),
            special_predictors=[
                ("gdpcap", range(1960, 1970), "mean"),
                *[
                    (c, range(1961, 1970, 2), "mean")
                    for c in [
                        "sec.agriculture",
                        "sec.energy",
                        "sec.industry",
                        "sec.construction",
                        "sec.services.venta",
                        "sec.services.nonventa",
                    ]
                ],
                ("popdens", [1969], "mean"),
            ],
        )
        return PySyncon(dataprep=dp).fit(panel)
    # Historical 2015 replication convention includes 1990 in predictor means
    # and the V-training validation window. This is explicit, not a default
    # for new studies; the common comparison always fits strictly before 1990.
    train = upstream.Dataprep(
        **common,
        predictors=["gdp", "trade", "infrate"],
        time_predictors_prior=range(1971, 1981),
        time_optimize_ssr=range(1981, 1991),
        special_predictors=[
            ("industry", range(1971, 1981), "mean"),
            ("schooling", [1970, 1975], "mean"),
            ("invest70", [1980], "mean"),
        ],
    )
    training_fit = upstream.Synth()
    training_fit.fit(dataprep=train)
    dp = upstream.Dataprep(
        **common,
        predictors=["gdp", "trade", "infrate"],
        time_predictors_prior=range(1981, 1991),
        time_optimize_ssr=range(1960, 1990),
        special_predictors=[
            ("industry", range(1981, 1991), "mean"),
            ("schooling", [1980, 1985], "mean"),
            ("invest80", [1980], "mean"),
        ],
    )
    return PySyncon(dataprep=dp, options={"custom_V": training_fit.V}).fit(panel)


def check_replication(name, result):
    """Assert predeclared reference targets; return a report of tested values.

    A tolerance is fixed by the source's precision or published upstream
    parity tolerance, not tuned after observing this implementation's result.
    A failure raises AssertionError and must be investigated.
    """
    if name == "california":
        value, target, tolerance = result.att, -15.603827872733849, 1e-6
        kind = "R synthdid reference ATT (paper reports -15.6)"
    elif name == "basque":
        value, target, tolerance = (
            result.effects.loc[1975:1997].mean(),
            -0.6995647842110987,
            0.025 * 0.6995647842110987,
        )
        kind = "pysyncon reference ATT, 1975–1997 (not the entire post-period)"
        assert abs(result.weights["Cataluna"] - 0.850816306) <= 0.025
        assert abs(result.weights["Madrid (Comunidad De)"] - 0.149183694) <= 0.025
    elif name == "germany":
        value, target, tolerance = result.att, -1555.1346777620479, 0.025 * 1555.1346777620479
        kind = "pysyncon reference ATT, 1990–2003"
    elif name == "kansas":
        value, target, tolerance = result.att, -0.0401, 0.001
        kind = "R augsynth vignette ATT, log GDP per capita"
    else:
        raise ValueError(f"Unknown replication: {name}")
    assert abs(value - target) <= tolerance, (
        f"{name} replication failed: {value} vs {target}, tolerance {tolerance}"
    )
    return {
        "source": kind,
        "estimate": float(value),
        "reference": target,
        "absolute_tolerance": tolerance,
    }
