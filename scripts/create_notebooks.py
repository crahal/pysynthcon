"""Regenerate the four reviewable notebook sources (execution is separate)."""

from pathlib import Path
from textwrap import dedent

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
        "GDP per capita, 1960–2003; first comparison post-period 1990. Outcome: GDP per capita in PPP-adjusted current US dollars, following the authors' 2025 erratum.",
        "Abadie, Diamond and Hainmueller (2015), [AJPS](https://doi.org/10.1111/ajps.12116). The historical two-stage predictor-weight specification is checked against the pysyncon reference ATT −1555.13468, with its predeclared 2.5% numerical tolerance. That reference includes 1990 in predictor means and the training validation window. We retain this historical convention explicitly; all common-comparison models below fit only years **strictly before 1990**.",
    ),
    "kansas": (
        "The Kansas income-tax cuts",
        "Quarterly log GDP per capita, 1990 Q1–2016 Q1; treatment starts 2012.25 as in the authors' vignette. Outcome: natural log GDP per capita.",
        "Ben-Michael, Feller and Rothstein (2021), [JASA](https://doi.org/10.1080/01621459.2021.1929245). Reproduce the no-covariate Ridge ASCM example from the authors' [augsynth vignette](https://github.com/ebenmichael/augsynth/blob/master/vignettes/singlesynth-vignette.md): ATT −0.0401, tolerance 0.001. The exact percent change corresponding to a log gap is `100 * expm1(gap)`. The 2012.25 onset follows the reference data and should not be interpreted as a fresh legislative-timing claim.",
    ),
}

FIGURE_DETAILS = {
    "california": {
        "case": "California Proposition 99",
        "outcome": "Cigarette sales (packs per capita)",
        "effect": "Effect on cigarette sales (packs per capita)",
        "units": "packs per capita",
        "periods": "1970–1988 pretreatment · 1989–2000 post-treatment · 38 donors",
        "onset": "1989",
        "reference": "SDID counterfactual",
        "reference_note": "Reference: Arkhangelsky et al. (2021), no-covariate SDID. Dashed vertical line: first affected year, 1989. Shading marks the post-treatment period.",
    },
    "basque": {
        "case": "The Basque Country",
        "outcome": "GDP per capita (thousands of 1986 US dollars)",
        "effect": "Effect on GDP per capita (thousands of 1986 US dollars)",
        "units": "thousand 1986 US$",
        "periods": "1960–1969 pretreatment · 1970–1997 post-treatment · 17 donors",
        "onset": "1970",
        "reference": "Historical SCM counterfactual",
        "reference_note": "Historical predictor specification. The numerical validation uses the 1975–1997 ATT; comparison figures use 1970–1997. The paper's percentage headline is shown above. Shading starts in 1970.",
    },
    "germany": {
        "case": "German reunification",
        "outcome": "GDP per capita (PPP current US$)",
        "effect": "Effect on GDP per capita (PPP current US$)",
        "units": "PPP current US$ per capita",
        "periods": "1960–1989 pretreatment · 1990–2003 post-treatment · 16 donors",
        "onset": "1990",
        "reference": "Historical SCM counterfactual",
        "reference_note": "Reference: Abadie, Diamond & Hainmueller (2015). The historical specification includes 1990 in predictor training. Comparison estimators use only periods before 1990.",
    },
    "kansas": {
        "case": "The Kansas income-tax cuts",
        "outcome": "Log GDP per capita",
        "effect": "Effect on log GDP per capita (log points)",
        "units": "log points",
        "periods": "1990 Q1–2012 Q1 pretreatment · 2012 Q2–2016 Q1 post-treatment · 49 donors",
        "onset": "2012 Q2",
        "reference": "Ridge ASCM counterfactual",
        "reference_note": "Reference: Ben-Michael, Feller & Rothstein (2021), no-covariate Ridge ASCM vignette. Shading starts at 2012.25, as in the source. Effects are log differences, not percentage changes.",
    },
}

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build(name, title, description, reference):
    cells = [
        markdown(
            f"# {title}\n\n{description}\n\n{reference}\n\n"
            "Run from a checkout installed using `requirements-replication.txt` and "
            "`pip install -e .`. Full execution also needs the separately supplied "
            "local reference bundle, which is excluded from this repository. "
            "Once it is provisioned, install the original R dependencies with "
            "`python .local/reference/scripts/install_r_references.py --library .r-library` after "
            "following [the R setup guide](../docs/reference_backends.md). "
            "The notebook downloads immutable public data with "
            "**requests**, checks the reference specification, and exports three "
            "publication figures locally as PDF, SVG and 300 dpi PNG. "
            "The figures are embedded below; generated exports stay outside version control. "
            "See [the replication guide](../docs/replication.md) and "
            "[the figure guide](../docs/plotting.md)."
        ),
        code("""
            %matplotlib inline
            import hashlib
            import os
            import json
            import time
            from dataclasses import replace
            from io import BytesIO
            from pathlib import Path
            from importlib.metadata import version

            import numpy as np
            import pandas as pd
            import requests
            import matplotlib.pyplot as plt
            from IPython.display import Markdown, display

            from pysynth import (
                SyntheticControl, SCUL, IndividualSC, TwoStageIndividualSC,
                SyntheticDID, DifferenceInDifferences, PySyncon, SCM, compare,
                plot_paths, plot_effects, plot_comparison, save_figure,
            )
            from pysynth.datasets import DATASETS
            from pysynth.replication import (
                LITERATURE_BENCHMARKS, replicate, check_replication, reference_comparisons,
                california_paper_inference,
            )
            from pysynth.plotting import reference_comparison_table

            pd.Series({p: version(p) for p in [
                'pysynth-control', 'numpy', 'scipy', 'pandas', 'matplotlib',
                'pysyncon', 'SyntheticControlMethods',
            ]}, name='version')
        """),
        markdown("""
            ## Download and verify the academic data

            The URL names an immutable upstream commit. SHA256 verification fails
            visibly if the bytes differ. No manual download or bundled data is required.
        """),
        code(f"""
            name = {name!r}
            spec = DATASETS[name]
            response = requests.get(spec.url, timeout=30)
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == spec.sha256
            data = pd.read_csv(BytesIO(response.content), sep=spec.separator)
            panel = spec.panel(data)
            print(spec.citation)
            print(spec.url)
            print(f'{{panel.n_control}} donors; {{panel.n_pre}} pre-periods; {{panel.n_post}} post-periods')
            data.head()
        """),
        markdown("""
            ## Reference specification and numerical check

            This uses the historical specification above. Its numerical assertion is
            separate from the new-method comparison. The path figure identifies the
            reference estimator explicitly. See `pysynth.replication.replicate` for
            predictor windows and optimization settings.

            The warm-toned annotation gives the source-reported result at its stated
            precision, with a source link and specification context. It is separate
            from the numerical replication check. See the
            [benchmark provenance](../docs/replication.md#literature-annotations)
            for the paper/author-example distinction, compatible units, and the
            German paper's 2025 erratum.
        """),
        code(f"""
            figure_details = {FIGURE_DETAILS[name]!r}
            benchmark = LITERATURE_BENCHMARKS[name]
            # Works when the kernel starts in either the repository or notebook folder.
            example_dir = Path.cwd() if Path.cwd().name == 'empirical_examples' else Path.cwd() / 'empirical_examples'
            figure_dir = Path(os.environ.get('PYSYNTH_FIGURE_DIR', example_dir / 'figures')) / name
            figure_dir.mkdir(parents=True, exist_ok=True)

            def export_figure(fig, kind):
                paths = save_figure(fig, figure_dir / f'{{name}}-{{kind}}')
                paths_text = ' · '.join(
                    f'`{{os.path.relpath(p, example_dir)}}`' for p in paths
                )
                display(Markdown('**Local figure exports** (relative to the notebook directory): ' + paths_text))
                display(Markdown(f'**Literature source:** [{{benchmark.citation}}]({{benchmark.source_url}})'))
                return paths

            reference = replicate(name, data)
            report = check_replication(name, reference)
            additional = None
            paper_uncertainty = None
            if name == 'california':
                additional = {{'SC': SyntheticDID().fit_sc(panel), 'DID': DifferenceInDifferences().fit(panel)}}
                paper_uncertainty = california_paper_inference(panel, reference)
                se_reference = paper_uncertainty.metadata['paper_replication']['reference_standard_error']
                np.testing.assert_allclose(paper_uncertainty.standard_error, se_reference, atol=1e-9, rtol=0)
                assert abs(additional['SC'].att - (-19.6)) <= 0.05
                assert abs(additional['DID'].att - (-27.3)) <= 0.05
                report.update({{'SE estimate': paper_uncertainty.standard_error,
                               'SE reference': se_reference, 'SE absolute tolerance': 1e-9}})

            def matched_references(fit, additional=None):
                refs = reference_comparisons(name, fit, additional=additional)
                if paper_uncertainty is not None:
                    refs = [replace(r, context=r.context + (
                        f' SE: paper 8.4; reproduced {{paper_uncertainty.standard_error:.3f}} '
                        '(authors’ seed 12345).'
                    )) if r.label == 'SDID' else r for r in refs]
                return refs

            references = matched_references(reference, additional=additional)
            reference_table = reference_comparison_table(references)
            display(reference_table)
            reference_table.to_csv(figure_dir / f'{{name}}-references.csv', index=False)
            display(pd.Series(report, name='Replication check'))
            ax = plot_paths(
                reference,
                title=figure_details['case'],
                subtitle='Observed outcome and reference counterfactual',
                ylabel=figure_details['outcome'], xlabel='Year',
                observed_label='Observed',
                counterfactual_label=figure_details['reference'],
                intervention_label=figure_details['onset'],
                references=references,
                note=figure_details['reference_note'],
            )
            reference_files = export_figure(ax.figure, 'reference')
            plt.show()
        """),
        markdown("""
            ## Fit all applicable methods on a common sample

            Every comparison uses the same outcome panel, eligible donors and treatment
            date. Classical pysyncon SCM estimates predictor importance by nested
            optimization. SyntheticControlMethods uses **all pretreatment outcome
            lags and all ten optimization starts**, with checked numerical recovery
            when a gradient line search fails. Neither adapter truncates the fit
            to accelerate this example. Native outcome SCM is retained as the
            explicitly named equal-period baseline.

            SCUL calls the pinned original R functions and glmnet, selecting
            `lambda.median` from fold-specific paths. Its rolling validation windows
            are stated in code. Basque has only ten pretreatment observations, so
            its validation horizon must be shorter than its 28-period post window.
            Penalized and robust SCM retain the explicit illustrative tuning values
            shown below; these are new applications, not paper-specific calibrations.

            ISC calls **MSCMT**, using the costofcare bootstrap workflow with **50
            donor resamples**. 2SISC first selects up to ten nearest donors, then
            runs the same ISC procedure within that group. These aggregate panels
            do not provide the survey-specific demographic predictors, eligibility
            rules, or individual sampling design of the original ISC studies.

            SDID uses donor post-period outcomes to estimate time weights; robust
            SCM uses them for donor denoising. Neither fits to treated post-outcomes.
            Each model runs in full; failures stop execution. Elapsed fitting times
            and optimizer diagnostics are exported alongside the estimates.
        """),
        code("""
            scul_windows = {'california': (5, 12), 'basque': (3, 5),
                            'germany': (14, 14), 'kansas': (16, 16)}
            initial, horizon = scul_windows[name]
            models = {
                'Outcome SCM': SyntheticControl(),
                'SCUL': SCUL(initial_window=initial, horizon=horizon),
                'ISC (full donor pool)': IndividualSC(),
                '2SISC (nearest 10)': TwoStageIndividualSC(n_neighbors=10),
                'Synthetic DID': SyntheticDID(),
                'DID': DifferenceInDifferences(),
                'pysyncon SCM': PySyncon('synth'),
                'Augmented SCM': PySyncon('augsynth'),
                'Penalized SCM (0.01)': PySyncon('pensynth', options={'lambda_': 0.01}),
                'Robust SCM (rank 3, ridge 0.1)': PySyncon('robust', options={'lambda_': 0.1, 'sv_count': 3}),
                'SyntheticControlMethods SCM': SCM('synth'),
                'Differenced SCM': SCM('diffsynth'),
            }
            results, elapsed = {}, {}
            for label, model in models.items():
                started = time.perf_counter()
                results[label] = model.fit(panel)
                elapsed[label] = time.perf_counter() - started
                print(f'{label}: {elapsed[label]:.2f} seconds')
            comparison = compare(results)
            comparison['elapsed_seconds'] = pd.Series(elapsed)
            if name == 'california':
                references = matched_references(results['Synthetic DID'], additional={
                    'SC': additional['SC'], 'DID': results['DID'],
                })
            elif name == 'kansas':
                references = matched_references(results['Augmented SCM'])
            diagnostics = {label: fit.metadata for label, fit in results.items()}
            (figure_dir / f'{name}-diagnostics.json').write_text(
                json.dumps(diagnostics, indent=2, default=str) + '\\n'
            )
            display_labels = {
                'ISC (full donor pool)': 'ISC · full donor pool',
                '2SISC (nearest 10)': '2SISC · nearest 10',
                'pysyncon SCM': 'SCM · pysyncon',
                'Penalized SCM (0.01)': 'Penalized SCM · λ = 0.01',
                'Robust SCM (rank 3, ridge 0.1)': 'Robust SCM · rank 3',
                'SyntheticControlMethods SCM': 'SCM · SCMethods',
                'Differenced SCM': 'Differenced SCM',
            }
            comparison.round(4)
        """),
        markdown("""
            ## Average effects and pretreatment fit

            The left panel reports the average post-treatment gap (ATT); the right
            reports pretreatment root mean squared error in the same outcome units.
            Methods retain the same order in every figure. They are not ranked by
            estimated effect or fit. Each dot is a point estimate: **there are no
            estimated confidence intervals in these figures**. Better pretreatment
            fit alone does not establish causal credibility.

            The reference strip pairs each source result with its matched reproduction
            and signed difference, using the same statistic and evaluation window.
            Approximate paper headlines and numerical upstream examples are labelled
            separately. They are not targets for every estimator.
        """),
        code("""
            fig, axes = plot_comparison(
                results, labels=display_labels,
                title=figure_details['case'],
                subtitle=figure_details['periods'],
                units=figure_details['units'],
                references=references,
            )
            comparison_files = export_figure(fig, 'comparison')
            comparison.to_csv(figure_dir / f'{name}-estimates.csv', float_format='%.12g')
            plt.show()
        """),
        markdown("""
            ## Effect paths on a common scale

            Every panel shows **observed minus counterfactual**. All twelve panels
            share the same horizontal and vertical scales; no extreme estimate is
            clipped. The zero line denotes no estimated effect. The dashed vertical
            line and shaded region identify the common treatment onset and post-period.
            Shading does not represent uncertainty. Calendar spacing is preserved,
            including quarterly observations. Panel labels replace repeated legends.
            The source/reproduction strip uses matched specifications and windows.
            Any numeric reference guide applies only to its matched estimator.
        """),
        code("""
            fig, axes = plot_effects(
                results, labels=display_labels,
                title=figure_details['case'],
                subtitle='Estimated effect paths · shared scales across all twelve methods',
                ylabel=figure_details['effect'], xlabel='Year',
                references=references,
                note=figure_details['periods'] + '. Shading marks the post-period; no uncertainty intervals are shown.',
            )
            effects_files = export_figure(fig, 'effects')
            plt.show()
        """),
        markdown("""
            ## Original SDID placebo inference

            The original synthdid placebo variance estimator refits donor and time
            weights in each of 200 resamples. The seed is fixed. These standard errors
            apply to SDID alone; they are not intervals for the other methods.
            The bootstrap and jackknife are undefined with only one treated unit.
            California uses the authors' exact R sampling plan (seed 12345),
            reproducing the paper's standard error 8.4 at its printed precision.
            The other examples use NumPy seed 0 for reproducible new applications.

        """),
        code("""
            if paper_uncertainty is not None:
                np.testing.assert_allclose(results['Synthetic DID'].att, reference.att, atol=1e-12)
                uncertainty = paper_uncertainty
            else:
                uncertainty = SyntheticDID().inference(
                    panel, method='placebo', replications=200, random_state=0,
                    estimate=results['Synthetic DID'],
                )
            inference = pd.Series({
                'SDID ATT': results['Synthetic DID'].att,
                'Placebo standard error': uncertainty.standard_error,
                '95% normal interval': uncertainty.confidence_interval(),
                'Replications': 200,
            }, name='SDID inference')
            display(inference)
            inference.to_csv(figure_dir / f'{name}-sdid-inference.csv')
            np.savetxt(figure_dir / f'{name}-sdid-resamples.csv', uncertainty.resamples, delimiter=',', fmt='%d')
            np.savetxt(figure_dir / f'{name}-sdid-draws.csv', uncertainty.draws, delimiter=',')
        """),
        markdown("""
            ## Inspect the donor coefficients

            These are the fifteen donors with the largest absolute coefficient in
            any method, not an exhaustive donor list. The coefficients are not
            renormalized. SCUL and some augmented/robust estimates can have negative
            weights or weight sums different from one. Intercepts and SDID time
            weights are separate components; this table does not represent them.
        """),
        code("""
            weights = pd.concat({label: fit.weights for label, fit in results.items()}, axis=1)
            weights.loc[weights.abs().max(axis=1).sort_values(ascending=False).index].head(15).round(4)
        """),
        markdown("""
            ## Interpretation, exports and reproducibility

            Negative effects mean the observed outcome is below its estimated
            untreated path. These are point estimates under different restrictions,
            tuning choices and identifying assumptions. They are not an uncertainty
            distribution. The historical reference check is distinct from the new
            comparison specifications and their stated tuning choices.

            Each figure is exported at **7.2 inches wide** in three formats: PDF with
            embedded fonts, SVG with editable text, and 300 dpi PNG. The local paths
            are printed beside each embedded figure. Generated figures, CSV summaries,
            and diagnostic files stay outside version control; rerunning the notebook
            recreates them. Use vector formats for manuscripts and adjust figure size, labels
            and captions through the plotting API before exporting to a journal's
            requirements. Nothing depends on system fonts or an external TeX install.

            `pysynth.placebo_test` provides in-space RMSPE-ratio diagnostics and a rank
            p-value under exchangeability assumptions; it is not synthdid's placebo
            variance estimator. The original SCUL scanner-data and ISC/costofcare
            survey studies are not reproduced here. Re-run from a fresh kernel to
            reproduce the source checks, estimates and figures together.
        """),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    # Stable identifiers keep figure updates reviewable in version control.
    for position, cell in enumerate(nb.cells):
        cell.id = f"{name}-{position:02d}"
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    destination = ROOT / "empirical_examples" / f"{name}.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(nb, destination)


if __name__ == "__main__":
    for name, (title, description, reference) in EXAMPLES.items():
        build(name, title, description, reference)
