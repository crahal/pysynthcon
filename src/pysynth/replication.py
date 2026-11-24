"""Explicit historical specifications for the empirical replication notebooks.

These helpers encode the predictor choices in the cited upstream examples.
They are kept separate from the common outcome-only comparisons.
"""

import subprocess
from io import StringIO

import numpy as np

from ._r import reference_directory, require_rscript
from .backends import PySyncon, _require
from .datasets import DATASETS
from .plotting import LiteratureBenchmark, ReferenceComparison
from .sdid import SyntheticDID

LITERATURE_BENCHMARKS = {
    "california": LiteratureBenchmark(
        headline="SDID mean effect: −15.6 packs per capita",
        citation="Arkhangelsky et al. (2021), Table 1 · 1989–2000",
        source_url="https://arxiv.org/pdf/1812.09970v4#page=8",
        context="No-covariate SDID; other estimators need not match this value.",
        value=-15.6,
        window=(1989, 2000),
    ),
    "basque": LiteratureBenchmark(
        headline="SCM: approximately 10% lower GDP per capita",
        citation="Abadie & Gardeazabal (2003), p. 117 · 1980s–1990s",
        source_url="https://economics.mit.edu/sites/default/files/publications/The%20Economic%20Costs%20of%20Conflict.pdf#page=5",
        context="Proportional headline; not a dollar ATT. No numeric guide is drawn on these level-scale axes.",
    ),
    "germany": LiteratureBenchmark(
        headline="SCM mean effect: approximately −1,600 US$ per capita",
        citation="Abadie, Diamond & Hainmueller (2015), p. 504 · 1990–2003",
        source_url="https://economics.mit.edu/sites/default/files/publications/Comparative%20Politics%20and%20the%20Synthetic%20Control.pdf#page=10",
        context="PPP current US$ (2025 erratum). Historical covariate specification differs from comparison fits.",
        value=-1600,
        window=(1990, 2003),
    ),
    "kansas": LiteratureBenchmark(
        headline="Ridge ASCM mean effect: −0.0401 log points",
        citation="Ben-Michael, Feller & Rothstein · augsynth vignette · 2012 Q2–2016 Q1",
        source_url="https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/vignettes/singlesynth-vignette.md#L217",
        context="Authors’ no-covariate worked example accompanying the 2021 paper; not its printed headline.",
        value=-0.0401,
        window=(2012.25, 2016.0),
        label="Authors’ reference",
    ),
}


_CALIFORNIA_DONORS = tuple(
    "Alabama|Arkansas|Colorado|Connecticut|Delaware|Georgia|Idaho|Illinois|Indiana|Iowa|"
    "Kansas|Kentucky|Louisiana|Maine|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    "New Hampshire|New Mexico|North Carolina|North Dakota|Ohio|Oklahoma|Pennsylvania|"
    "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|"
    "West Virginia|Wisconsin|Wyoming".split("|")
)


def california_paper_inference(panel, estimate):
    """Reproduce California Table 1's SDID placebo standard error.

    The authors' paper-results vignette uses 200 draws with R seed 12345.
    An isolated base-R process generates their Mersenne-Twister sampling plan;
    Python then performs all original SDID placebo refits. This requires Rscript
    and the separately provisioned local ``california_rng.R`` resource, but no
    R packages. Donors must have the original alphabetical ordering,
    California must be last, and annual observations must span 1970–2000 with
    treatment beginning in 1989. ``estimate`` must be an SDID fit of this panel.

    The numerical R reference standard error is 8.36799308138054 packs per
    capita, which rounds to the published 8.4. A NumPy seed with the same
    integer does not define the same resampling plan.
    """
    if (
        panel.n_control != 38
        or panel.n_treated != 1
        or panel.n_pre != 19
        or not np.array_equal(panel.times, np.arange(1970, 2001))
        or panel.units != (*_CALIFORNIA_DONORS, "California")
    ):
        raise ValueError(
            "California paper inference requires the original 38 alphabetically ordered "
            "donors followed by California, annual 1970–2000 data, and 1989 onset"
        )
    settings = SyntheticDID()._settings(panel)
    metadata = getattr(estimate, "metadata", {})
    if (
        any(metadata.get(key) != value for key, value in settings.items())
        or metadata.get("sparsify") is not True
        or metadata.get("omega_intercept") is not True
    ):
        raise ValueError("estimate must use the original default SDID fitting settings")
    script = reference_directory("california_rng.R") / "california_rng.R"
    executable = require_rscript()
    completed = subprocess.run(
        [executable, "--vanilla", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            "R failed to generate the California paper's resampling plan:\n"
            + completed.stderr.strip()
        )
    indices = np.loadtxt(StringIO(completed.stdout), delimiter=",", dtype=int)
    result = SyntheticDID().inference(
        panel, method="placebo", replications=200, estimate=estimate, resamples=indices
    )
    result.metadata["paper_replication"] = {
        "citation": "Arkhangelsky et al. (2021), Table 1, California SDID standard error",
        "source_url": "https://github.com/synth-inference/synthdid/blob/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/vignettes/paper-results.Rmd#L113-L123",
        "seed": 12345,
        "rng_kind": ["Mersenne-Twister", "Inversion", "Rejection"],
        "replications": 200,
        "published_standard_error": 8.4,
        "reference_standard_error": 8.36799308138054,
    }
    return result


def reference_comparisons(name, reference, *, additional=None):
    """Bind sourced results to their reproduced historical specifications.

    Values are recomputed on each source window. Approximate paper headlines
    and precise upstream software targets are identified separately. California
    accepts optional ``additional={"SC": sc_estimate, "DID": did_estimate}``;
    SC must be the synthdid paper's SC specification, not an arbitrary SCM fit.
    The caller is responsible for fitting the historical specification, as in
    :func:`replicate`; this helper does not certify a method from its label.
    """
    if name not in LITERATURE_BENCHMARKS:
        raise ValueError(f"Unknown replication: {name}")
    if additional and (name != "california" or set(additional) - {"SC", "DID"}):
        raise ValueError("additional references support California SC and DID only")
    source = LITERATURE_BENCHMARKS[name]
    common = dict(estimate=reference, citation=source.citation, source_url=source.source_url)
    if name == "california":
        comparisons = [
            ReferenceComparison(
                label="SDID",
                reference=-15.6,
                window=(1989, 2000),
                units="packs per capita",
                context="No covariates. Source rounded to one decimal; Δ includes rounding.",
                **common,
            )
        ]
        for method, value in {"SC": -19.6, "DID": -27.3}.items():
            if method in (additional or {}):
                comparisons.append(
                    ReferenceComparison(
                        label=method,
                        estimate=additional[method],
                        reference=value,
                        window=(1989, 2000),
                        units="packs per capita",
                        citation=source.citation,
                        source_url=source.source_url,
                        context="Table 1 specification; source rounded to one decimal.",
                    )
                )
        return tuple(comparisons)
    if name == "kansas":
        return (
            ReferenceComparison(
                label="Ridge ASCM",
                reference=-0.0401,
                window=(2012.25, 2016.0),
                window_label="2012 Q2–2016 Q1",
                units="log GDP per capita",
                source_label="Authors’ vignette",
                reference_precision=4,
                precision=5,
                context="No covariates; 1-SE ridge selection. Source rounded to four decimals.",
                **common,
            ),
        )
    commit = "b77dc20e29b012ab03ca1d4caeef7b41861e86f5"
    upstream = dict(
        estimate=reference,
        citation=f"pysyncon parity fixture · {commit[:7]}",
        source_url=f"https://github.com/sdfordham/pysyncon/blob/{commit}/parity_tests/test_synth_{name}.py",
        source_label="Upstream software",
        context="Historical predictor specification. This precise target is not a paper-printed estimate.",
    )
    if name == "basque":
        return (
            ReferenceComparison(
                label="Historical SCM",
                reference=-10,
                window=(1980, 1997),
                units="% of observed GDP",
                statistic="mean_percent_effect",
                approximate=True,
                reference_precision=0,
                precision=2,
                context="Approximate 1980s–1990s headline; available years 1980–97. Mean of 100 × gap / observed GDP (Fig. 2 denominator).",
                **common,
            ),
            ReferenceComparison(
                label="Historical SCM",
                reference=-0.6995647842110987,
                window=(1975, 1997),
                units="thousand 1986 US$ per capita",
                precision=4,
                reference_precision=4,
                **upstream,
            ),
        )
    return (
        ReferenceComparison(
            label="Historical SCM",
            reference=-1600,
            window=(1990, 2003),
            units="PPP current US$ per capita",
            approximate=True,
            reference_precision=0,
            precision=1,
            context="Approximate paper headline. Units follow the authors’ 2025 erratum.",
            **common,
        ),
        ReferenceComparison(
            label="Historical SCM",
            reference=-1555.1346777620479,
            window=(1990, 2003),
            units="PPP current US$ per capita",
            precision=2,
            reference_precision=2,
            **upstream,
        ),
    )


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
