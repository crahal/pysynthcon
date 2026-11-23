"""Regenerate the four reviewable notebook sources (execution is separate)."""

from pathlib import Path

import nbformat as nbf

EXAMPLES = {
    "california": (
        "California Proposition 99",
        "Cigarette consumption, 1970–2000; first affected year 1989. Outcome: cigarette packs per capita.",
        "The SDID reference is −15.6038278727 packs per person. This is the no-covariate example in Arkhangelsky et al. (2021), [AER](https://doi.org/10.1257/aer.20190159), using the tobacco-control setting of Abadie, Diamond and Hainmueller (2010), [JASA](https://doi.org/10.1198/jasa.2009.ap08746). The latter's covariate-based SCM specification is **not** reproduced by the outcome-only SCM row below.",
    ),
    "basque": (
        "The economic costs of conflict in the Basque Country",
        "GDP per capita, 1960–1997; comparison post-period begins 1970. Outcome: thousands of 1986 US dollars.",
        "Abadie and Gardeazabal (2003), [AER](https://doi.org/10.1257/000282803321455188). The historical predictor specification is reproduced separately. Check Catalonia/Madrid weights against 0.851/0.149 and the upstream reference ATT of −0.699565 **over 1975–1997**. This window differs from the table's 1970–1997 ATT. The upstream Synth example includes the Spain aggregate in the donor pool; its fitted weight is zero. The paper's approximate 10% headline is not treated as an exact numeric target.",
    ),
    "germany": (
        "German reunification",
        "GDP per capita, 1960–2003; first comparison post-period 1990. Outcome: GDP per capita in the source's US-dollar units.",
        "Abadie, Diamond and Hainmueller (2015), [AJPS](https://doi.org/10.1111/ajps.12116). The historical two-stage predictor-weight specification is checked against the pysyncon reference ATT −1555.13468, with its predeclared 2.5% numerical tolerance. That reference includes 1990 in predictor means and the training validation window. We retain this historical convention explicitly; all common-comparison models below fit only years **strictly before 1990**.",
    ),
    "kansas": (
        "The Kansas income-tax cuts",
        "Quarterly log GDP per capita, 1990 Q1–2016 Q1; treatment starts 2012.25 as in the authors' vignette. Outcome: natural log GDP per capita.",
        "Ben-Michael, Feller and Rothstein (2021), [JASA](https://doi.org/10.1080/01621459.2021.1929245). Reproduce the no-covariate Ridge ASCM example from the authors' [augsynth vignette](https://github.com/ebenmichael/augsynth/blob/master/vignettes/singlesynth-vignette.md): ATT −0.0401, tolerance 0.001. The exact percent change corresponding to a log gap is `100 * expm1(gap)`. The 2012.25 onset follows the reference data and should not be interpreted as a fresh legislative-timing claim.",
    ),
}

ROOT = Path(__file__).resolve().parents[1]


def build(name, title, description, reference):
    cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{description}\n\n{reference}\n\nRun from a checkout installed using `requirements-replication.txt` and `pip install -e .`. This notebook downloads immutable public data using **requests**; an internet connection is required. See [the replication guide](../docs/replication.md) for provenance and scope."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\nimport hashlib\nfrom io import BytesIO\nfrom importlib.metadata import version\n\nimport numpy as np\nimport pandas as pd\nimport requests\nimport matplotlib.pyplot as plt\n\nfrom pysynth import (SyntheticControl, SCUL, IndividualSC, TwoStageIndividualSC, SyntheticDID,\n                     DifferenceInDifferences, PySyncon, SCM, compare)\nfrom pysynth.datasets import DATASETS\nfrom pysynth.replication import replicate, check_replication\n\npd.Series({p: version(p) for p in ['pysynth-control', 'numpy', 'scipy', 'pandas', 'pysyncon', 'SyntheticControlMethods']}, name='version')"
        ),
        nbf.v4.new_markdown_cell(
            "## Download and verify the academic data\n\nThe URL names an immutable upstream commit. SHA256 verification fails visibly if the bytes differ. No manual download or bundled dataset is required."
        ),
        nbf.v4.new_code_cell(
            f"name = {name!r}\nspec = DATASETS[name]\nresponse = requests.get(spec.url, timeout=30)\nresponse.raise_for_status()\nassert hashlib.sha256(response.content).hexdigest() == spec.sha256\ndata = pd.read_csv(BytesIO(response.content), sep=spec.separator)\npanel = spec.panel(data)\nprint(spec.citation)\nprint(spec.url)\nprint(f'{{panel.n_control}} donors; {{panel.n_pre}} pre-periods; {{panel.n_post}} post-periods')\ndata.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## Reference specification and numerical check\n\nThis cell uses the historical specification described above. Its assertion is separate from the new-method comparison. See `pysynth.replication.replicate` for the full predictor windows and optimization settings."
        ),
        nbf.v4.new_code_cell(
            "reference = replicate(name, data)\nreport = check_replication(name, reference)\ndisplay(pd.Series(report, name='Replication check'))\nreference.plot()\nplt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Compare all applicable methods\n\nAll rows use the same outcome panel, donor eligibility and intervention date. Native SCM assigns equal importance to outcome periods; the pysyncon SCM adapter standardizes predictor rows. Penalized and robust SCM use explicit illustrative tuning values below. The SyntheticControlMethods rows use at most ten evenly spaced pre-outcome lag predictors plus its outcome mean; this keeps the nested optimization practical for quarterly panels. SCUL selects its penalty by rolling validation entirely within the pre-period, using a shared pretreatment penalty grid.\n\nVanilla Individual Synthetic Control (ISC) fits over the full eligible donor pool. Two-Stage Individual Synthetic Control (2SISC) first selects up to ten nearest donors by pretreatment Euclidean distance, then fits convex weights within that group. Both are shown separately. With one aggregate treated case, default outcome features and no resampling, vanilla ISC coincides with Outcome SCM; that agreement is expected. These applications do not reproduce the Vagni–Breen survey study. SDID uses donor post-period outcomes to learn time weights, and robust SCM denoises donors over the full panel; neither uses treated post outcomes to estimate weights. The newer methods' rows are **new applications**, not estimates published in their development papers.\n\nUpstream convergence warnings remain visible. No method is silently dropped if fitting fails."
        ),
        nbf.v4.new_code_cell(
            "models = {\n    'Outcome SCM': SyntheticControl(),\n    'SCUL': SCUL(initial_window=max(3, panel.n_pre // 2), horizon=1),\n    'ISC (full donor pool)': IndividualSC(),\n    '2SISC (nearest 10)': TwoStageIndividualSC(n_neighbors=10),\n    'Synthetic DID': SyntheticDID(),\n    'DID': DifferenceInDifferences(),\n    'pysyncon SCM': PySyncon('synth'),\n    'Augmented SCM': PySyncon('augsynth'),\n    'Penalized SCM (0.01)': PySyncon('pensynth', options={'lambda_': 0.01}),\n    'Robust SCM (rank 3, ridge 0.1)': PySyncon('robust', options={'lambda_': 0.1, 'sv_count': 3}),\n    'SyntheticControlMethods SCM': SCM('synth', n_lags=10),\n    'Differenced SCM': SCM('diffsynth', n_lags=10),\n}\nresults = {label: model.fit(panel) for label, model in models.items()}\ncomparison = compare(results)\ncomparison.round(4)"
        ),
        nbf.v4.new_markdown_cell(
            "## Estimated effect paths\n\nDifferences across estimates reflect different restrictions, extrapolation, regularization and level adjustments. Pre-period RMSE measures fit; it does not validate the identifying assumptions or rank causal credibility."
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)\nfor ax, (label, result) in zip(axes.flat, results.items()):\n    result.plot(ax=ax, effects=True)\n    ax.set_title(label, fontsize=10)\n    ax.get_legend().remove()\nfor ax in list(axes.flat)[len(results):]:\n    ax.set_visible(False)\nfig.tight_layout()\nplt.show()"
        ),
        nbf.v4.new_code_cell(
            "weights = pd.concat({label: fit.weights for label, fit in results.items()}, axis=1)\nweights.loc[weights.abs().max(axis=1).sort_values(ascending=False).index].head(15).round(4)"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation and reproducibility\n\nThe ATT is the average observed-minus-counterfactual gap in the post-period, in the stated outcome units. Negative effects mean an outcome below its estimated untreated path. No automatic confidence interval is reported: serial dependence, donor eligibility, spillovers and assignment assumptions need a defensible inferential design.\n\n`pysynth.placebo_test` offers in-space RMSPE-ratio diagnostics and a rank p-value under exchangeability assumptions. It is not the synthdid package's placebo variance estimator. SCUL's original scanner-data empirical study and the costofcare/ISC survey study are not reproduced here: their original data and full research designs differ from these public canonical panels.\n\nThe downloaded bytes and software versions are recorded above. Re-run the notebook from a fresh kernel to reproduce both the reference check and the comparison."
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    destination = ROOT / "empirical_examples" / f"{name}.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(nb, destination)


if __name__ == "__main__":
    for name, (title, description, reference) in EXAMPLES.items():
        build(name, title, description, reference)
