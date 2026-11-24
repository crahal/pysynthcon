"""Regenerate the four reviewable notebook sources (execution is separate)."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

EXAMPLES = {
    "california": (
        "California Proposition 99",
        "Cigarette consumption, 1970–2000; first affected year 1989. Outcome: cigarette packs per capita.",
        "The SDID reference is −15.6038278727 packs per person. This is the no-covariate example in Arkhangelsky et al. (2021), [AER](https://doi.org/10.1257/aer.20190159), using the tobacco-control setting of Abadie, Diamond and Hainmueller (2010), [JASA](https://doi.org/10.1198/jasa.2009.ap08746). The final section reconstructs the 2010 study’s covariate-based SCM and all 39 state reassignments.",
    ),
    "basque": (
        "The economic costs of conflict in the Basque Country",
        "GDP per capita, 1960–1997; comparison post-period begins 1970. Outcome: thousands of 1986 US dollars.",
        "Abadie and Gardeazabal (2003), [AER](https://doi.org/10.1257/000282803321455188). The authors’ historical predictor specification is reproduced separately: 16 regional donors, excluding Spain’s national aggregate, and four annual education shares averaged over 1964–1969, matching paper Table 3. Check Catalonia/Madrid weights against the published 0.8508/0.1492 and an independent original-R Synth ATT of −0.6995627 **over 1975–1997**. This window differs from the table's 1970–1997 ATT. The paper's approximate 10% headline is not treated as an exact numeric target.",
    ),
    "germany": (
        "German reunification",
        "GDP per capita, 1960–2003; first comparison post-period 1990. Outcome: GDP per capita in PPP-adjusted current US dollars, following the authors' 2025 erratum.",
        "Abadie, Diamond and Hainmueller (2015), [AJPS](https://doi.org/10.1111/ajps.12116). The paper reports that reunification reduced West Germany’s annual GDP per capita by approximately US$1,600 on average over 1990–2003. The historical two-stage predictor-weight specification includes 1990 in predictor means and in the training validation window. We retain this historical convention explicitly; all common-comparison models below fit only years **strictly before 1990**.",
    ),
    "kansas": (
        "The Kansas income-tax cuts",
        "Quarterly log GDP per capita, 1990 Q1–2016 Q1; treatment starts 2012.25 as in the authors' vignette. Outcome: natural log GDP per capita.",
        "Ben-Michael, Feller and Rothstein (2021), [JASA](https://doi.org/10.1080/01621459.2021.1929245). Reproduce the no-covariate Ridge ASCM example from the authors' [augsynth vignette](https://github.com/ebenmichael/augsynth/blob/7e70072232fe75057aa8b5d7af08c7e1aef53ae8/vignettes/singlesynth-vignette.md): ATT −0.0401, printed to four decimal places. The exact percent change corresponding to a log gap is `100 * expm1(gap)`. The 2012.25 onset follows the reference data and should not be interpreted as a fresh legislative-timing claim.",
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
        "reference_note": "No-covariate SDID. Shading marks the post-treatment period, not uncertainty.",
        "comparison_note": "The paper’s SC comparison uses its own weight specification. The outcome and predictor SCM rows show the specifications chosen here.",
    },
    "basque": {
        "case": "The Basque Country",
        "outcome": "GDP per capita (thousands of 1986 US dollars)",
        "effect": "Effect on GDP per capita (thousands of 1986 US dollars)",
        "units": "thousand 1986 US$",
        "periods": "1960–1969 pretreatment · 1970–1997 post-treatment · 16 donors",
        "onset": "1970",
        "reference": "Historical SCM counterfactual",
        "reference_note": "Historical predictor SCM. Numerical check: 1975–1997; common comparison: 1970–1997. Shading starts in 1970.",
        "comparison_note": "The full post-treatment mean covers 1970–1997.",
    },
    "germany": {
        "case": "German reunification",
        "outcome": "GDP per capita (PPP current US$)",
        "effect": "Effect on GDP per capita (PPP current US$)",
        "units": "PPP current US$ per capita",
        "periods": "1960–1989 pretreatment · 1990–2003 post-treatment · 16 donors",
        "onset": "1990",
        "reference": "Historical SCM counterfactual",
        "reference_note": "Historical predictor SCM includes 1990 in training; common-comparison fits use only years before 1990.",
        "comparison_note": "Historical SCM includes 1990 in predictor training; common-comparison fits use only years before 1990.",
    },
    "kansas": {
        "case": "The Kansas income-tax cuts",
        "outcome": "Log GDP per capita",
        "effect": "Effect on log GDP per capita (log points)",
        "units": "log points",
        "periods": "1990 Q1–2012 Q1 pretreatment · 2012 Q2–2016 Q1 post-treatment · 49 donors",
        "onset": "2012 Q2",
        "reference": "Ridge ASCM counterfactual",
        "reference_note": "No-covariate Ridge ASCM. Effects are log differences. Shading starts at the authors' 2012.25 onset.",
        "comparison_note": "Reference: the authors' Ridge ASCM vignette. Effects are log differences, not percentage changes.",
    },
}

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def details(summary, source):
    return f"<details>\n<summary>{summary}</summary>\n\n{dedent(source).strip()}\n\n</details>"


def build(name, title, description, reference):
    donor_display_names = (
        "\n                unit_labels=data.groupby('fips')['state'].first().to_dict(),"
        if name == "kansas"
        else ""
    )
    cells = [
        markdown(
            f"# {title}\n\n{description}\n\n"
            "Historical results and matched reproductions are annotated directly on the figures. "
            "Expand the notes and audit tables for full specifications and numerical checks.\n\n"
            + details("Historical specification and source", reference)
            + "\n\n"
            + details(
                "Setup, data and figure exports",
                "Run from a checkout installed using `requirements-replication.txt` and "
                "`pip install -e .`. Full execution also needs the separately supplied "
                "local reference bundle, which is excluded from this repository. "
                "Once it is provisioned, install the original R dependencies with "
                "`python .local/reference/scripts/install_r_references.py --library .r-library` after "
                "following [the R setup guide](../docs/reference_backends.md). "
                "The notebook downloads immutable public data with "
                f"**requests**, checks the reference specification, and exports {'six' if name == 'california' else 'five'} "
                "publication figures locally as PDF, SVG and 300 dpi PNG. "
                f"Figures, result tables and diagnostics are written under `outputs/empirical_examples/{name}/` "
                "relative to the checkout root. "
                "See [the replication guide](../docs/replication.md) and "
                "[the figure guide](../docs/plotting.md).",
            )
        ),
        code("""
            %matplotlib inline
            import hashlib
            import json
            import time
            import warnings
            from collections import Counter
            from dataclasses import replace
            from functools import partial
            from html import escape
            from io import BytesIO
            from pathlib import Path
            from importlib.metadata import version

            import numpy as np
            import pandas as pd
            import requests
            import matplotlib.pyplot as plt
            from IPython.display import HTML, Markdown, display

            from pysynth import (
                SyntheticControl, SCUL, IndividualSC, TwoStageIndividualSC,
                SyntheticDID, DifferenceInDifferences, PredictorSyntheticControl,
                AugmentedSyntheticControl, PenalizedSyntheticControl, RobustSyntheticControl,
                DifferencedSyntheticControl, compare,
                plot_paths, plot_effects, plot_comparison, save_figure,
                select_estimator, donor_sensitivity, plot_backtest, plot_donor_sensitivity,
            )
            from pysynth.datasets import DATASETS
            from pysynth.replication import (
                LITERATURE_BENCHMARKS, replicate, check_replication, reference_comparisons,
                california_paper_inference,
            )
            from pysynth.plotting import reference_comparison_table

            versions = pd.Series({p: version(p) for p in [
                'pysynth-control', 'numpy', 'scipy', 'pandas', 'matplotlib',
                'pysyncon', 'SyntheticControlMethods',
            ]}, name='version').rename_axis('package')
            versions
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
        """),
        markdown("""
            ## Reference specification and numerical check

            The figure pairs each source result with its matched reproduction.
            Historical specifications and their numerical checks are separate from
            the common-sample applications below.
        """),
        code(f"""
            figure_details = {FIGURE_DETAILS[name]!r}
            benchmark = LITERATURE_BENCHMARKS[name]
            cwd = Path.cwd()
            root = cwd.parent if cwd.name == 'empirical_examples' else cwd
            output_dir = root / 'outputs' / 'empirical_examples' / name
            output_dir.mkdir(parents=True, exist_ok=True)
            versions.to_csv(output_dir / f'{{name}}-versions.csv')
            warning_rows = []

            def export_figure(fig, kind):
                paths = save_figure(fig, output_dir / f'{{name}}-{{kind}}')
                paths_text = ' · '.join(f'`{{p.relative_to(root).as_posix()}}`' for p in paths)
                display(Markdown('Project exports · ' + paths_text + f' · [Literature source]({{benchmark.source_url}})'))
                return paths

            def display_audit(title, *tables):
                tables_html = ''.join(
                    (table.to_frame() if isinstance(table, pd.Series) else table).to_html(escape=True)
                    for table in tables
                )
                display(HTML(f'<details><summary>{{escape(title)}}</summary>{{tables_html}}</details>'))

            def display_warnings(section, records):
                if records:
                    counts = Counter((w.category.__name__, str(w.message)) for w in records)
                    rows = [(kind, message, count) for (kind, message), count in counts.items()]
                    warning_rows.extend((section, *row) for row in rows)
                    display_audit(f'{{section}} warnings ({{len(records)}} emitted)',
                                  pd.DataFrame(rows, columns=['Category', 'Message', 'Count']))

            with warnings.catch_warnings(record=True) as reference_warnings:
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
            reference_table.to_csv(output_dir / f'{{name}}-references.csv', index=False)
            replication_checks = pd.Series(report, name='Replication check')
            replication_checks.to_csv(output_dir / f'{{name}}-replication-checks.csv', index_label='statistic')
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
            plt.show()
            reference_files = export_figure(ax.figure, 'reference')
            display_warnings('Historical fit and paper inference', reference_warnings)
            display_audit('Reference values and numerical replication checks',
                          reference_table, replication_checks)
        """),
        markdown(
            "## Fit all applicable methods on a common sample\n\n"
            "All methods share the outcome panel, eligible donors and treatment date. "
            "These are new applications, distinct from the historical replications.\n\n"
            + details(
                "Estimator specifications and full computation settings",
                """
            **Outcome SCM** minimizes equally weighted squared gaps across all
            pretreatment periods. **Predictor SCM** matches three predictors: the
            mean outcome in each of three consecutive, nearly equal-length blocks
            of pretreatment dates. Its nested fit chooses predictor importance using
            error across the full pretreatment path. The three-block design is a
            prespecified application of classical SCM in these comparisons.
            [Abadie, Diamond and Hainmueller (2010)](https://doi.org/10.1198/jasa.2009.ap08746)

            **Ridge-augmented SCM** uses a ridge outcome model to adjust for
            remaining pretreatment imbalance. The penalty is chosen by
            pretreatment cross-validation.
            [Ben-Michael, Feller and Rothstein (2021)](https://doi.org/10.1080/01621459.2021.1929245)

            **Penalized SCM** matches pretreatment outcomes with a penalty on
            the distance from the treated unit to each contributing donor.
            The comparison fixes the pairwise penalty at 0.01.
            [Abadie and L’Hour (2021)](https://doi.org/10.1080/01621459.2021.1971535)

            **Robust SCM** denoises the full donor outcome matrix, fits weights
            by ridge regression against pretreatment treated outcomes, and predicts
            with the denoised donors. Here the rank is three and the ridge penalty
            is 0.1. Both penalized and robust SCM use study-design tuning choices
            for these applications.
            [Amjad, Shah and Shen (2018)](https://jmlr.org/papers/v19/17-777.html)

            **Differenced SCM** targets pretreatment changes and restores the
            mean pretreatment level gap when predicting the untreated path.
            This application uses all pretreatment outcome lags as level predictors,
            ten optimizer starts, and the zero-penalty specification. It uses
            the full pretreatment period for fitting. The working paper also
            considers tuning by a training/validation split.
            [Engelbrektson (2021 working paper), §4 and Table 2](https://arxiv.org/abs/2111.10784)

            SCUL calls the pinned original R functions and glmnet, selecting
            `lambda.median` from fold-specific paths. Its original near-zero-variance
            screening checks the full series and each training window, so data
            admissibility can depend on post-period values; lasso fitting and
            penalty selection use only pretreatment outcomes. Rolling windows
            are stated in code. Basque has only ten pretreatment observations, so
            its validation horizon must be shorter than its 28-period post window.

            ISC calls **MSCMT**, using the costofcare bootstrap workflow with **50
            donor resamples**. 2SISC follows Petrillo et al. (2025): raw nearest-neighbor
            screening followed by nested SCM on temporal predictor means, without
            donor resampling. `n_neighbors=10` counts ten donors; the authors' KDTree
            implementation counts the treated case in its query size. The earlier
            MSCMT/50-draw two-stage variant remains explicitly available through
            `backend='mscmt', n_bootstrap=50`. These aggregate panels
            do not provide the survey-specific demographic predictors, eligibility
            rules, or individual sampling design of the original ISC studies.

            SDID uses post-period donor outcomes to estimate time weights.
            Each model runs in full. The exports include elapsed fitting times
            and optimizer diagnostics alongside the estimates.
        """,
            )
        ),
        code("""
            scul_windows = {'california': (5, 12), 'basque': (3, 5),
                            'germany': (14, 14), 'kansas': (16, 16)}
            initial, horizon = scul_windows[name]
            pre_dates = panel.times[:panel.n_pre].to_numpy()
            predictor_windows = [block.tolist() for block in np.array_split(pre_dates, 3)]
            models = {
                'Outcome SCM': SyntheticControl(),
                'Predictor SCM (three pre-period means)': PredictorSyntheticControl(
                    predictor_windows=predictor_windows,
                ),
                'Ridge-augmented SCM': AugmentedSyntheticControl(),
                'Penalized SCM (0.01)': PenalizedSyntheticControl(lambda_=0.01),
                'Robust SCM (rank 3, ridge 0.1)': RobustSyntheticControl(sv_count=3, lambda_=0.1),
                'Differenced SCM': DifferencedSyntheticControl(),
                'SCUL': SCUL(initial_window=initial, horizon=horizon),
                'ISC (full donor pool)': IndividualSC(),
                '2SISC (nearest 10)': TwoStageIndividualSC(n_neighbors=10),
                'Synthetic DID': SyntheticDID(),
                'DID': DifferenceInDifferences(),
            }
            results, elapsed = {}, {}
            with warnings.catch_warnings(record=True) as comparison_warnings:
                for label, model in models.items():
                    started = time.perf_counter()
                    results[label] = model.fit(panel)
                    elapsed[label] = time.perf_counter() - started
            print(f'Fitted {len(results)} estimators on the common panel.')
            comparison = compare(results)
            comparison['elapsed_seconds'] = pd.Series(elapsed)
            if name == 'california':
                references = matched_references(results['Synthetic DID'], additional={
                    'SC': additional['SC'], 'DID': results['DID'],
                })
            elif name == 'kansas':
                references = matched_references(results['Ridge-augmented SCM'])
            diagnostics = {label: fit.metadata for label, fit in results.items()}
            (output_dir / f'{name}-diagnostics.json').write_text(
                json.dumps(diagnostics, indent=2, default=str) + chr(10)
            )
            display_labels = {
                'ISC (full donor pool)': 'ISC · full donor pool',
                '2SISC (nearest 10)': '2SISC · nearest 10',
                'Predictor SCM (three pre-period means)': 'Predictor SCM · 3 blocks',
                'Penalized SCM (0.01)': 'Penalized SCM · λ = 0.01',
                'Robust SCM (rank 3, ridge 0.1)': 'Robust SCM · rank 3',
            }
        """),
        markdown("""
            ## Average effects and pretreatment fit

            Source comparisons use matched statistics and evaluation windows;
            they are not targets for every estimator. Dots are point estimates without
            confidence intervals.
        """),
        code("""
            fig, axes = plot_comparison(
                results, labels=display_labels,
                title=figure_details['case'],
                subtitle=figure_details['periods'],
                units=figure_details['units'],
                references=references,
                note=figure_details['comparison_note'] + ' Points are estimates without confidence intervals; RMSE is descriptive.',
            )
            comparison.to_csv(output_dir / f'{name}-estimates.csv')
            plt.show()
            comparison_files = export_figure(fig, 'comparison')
            display_warnings('Estimator comparison', comparison_warnings)
            display_audit('Estimator estimates, pretreatment fit and elapsed times', comparison.round(4))
        """),
        markdown("""
            ## Effect paths on a common scale

            Comparison and reference specifications show **observed minus counterfactual** on shared scales.
            Numeric reference guides apply only to their matched estimators.
        """),
        code("""
            fig, axes = plot_effects(
                results, labels=display_labels,
                title=figure_details['case'],
                subtitle='Estimated effect paths · comparison and reference specifications',
                ylabel=figure_details['effect'], xlabel='Year',
                references=references,
                note=figure_details['comparison_note'] + ' Shading marks the post-period, not uncertainty.',
            )
            plt.show()
            effects_files = export_figure(fig, 'effects')
        """),
        markdown(
            "## Pretreatment validation and specification selection\n\n"
            "Common pretreatment holdouts compare prediction performance. "
            "Selection does not establish causal validity; the selected score is optimistic.\n\n"
            + details(
                "Validation design, selection and limitations",
                """
            These are **new analyses of the academic data**, separate from the paper
            replications above. We compare methods and three prespecified 2SISC donor
            counts on common withheld pretreatment blocks. Every actual post-treatment
            row is excluded from validation. Held-out treated outcomes are masked
            during fitting; contemporaneous donor outcomes remain available.

            SCUL keeps its original internal rolling windows, so the outer training
            sample must be long enough to contain them. Short panels consequently
            offer only a few validation folds: this is a transparent illustration,
            not strong evidence that the selected estimator is generally superior.
            Every fit retains the estimator's full optimization and resampling settings.
            SCUL's full-series admissibility screen sees the masked target series here.

            Selection minimizes pooled held-out RMSE, then fits the winner on the
            original panel. The selected score is optimistic after model selection;
            no post-selection confidence interval or causal guarantee is implied.
            Candidate choices and folds must be set before examining treatment effects.
        """,
            )
        ),
        code("""
            candidates = {
                'Outcome SCM': SyntheticControl,
                'SCUL': partial(SCUL, initial_window=initial, horizon=horizon),
                'ISC': IndividualSC,
                'SDID': SyntheticDID,
                'DID': DifferenceInDifferences,
                **{f'2SISC k={k}': partial(TwoStageIndividualSC, n_neighbors=k)
                   for k in (3, 5, 10)},
            }
            validation_start = max(panel.n_pre // 2, initial + horizon)
            validation_horizon = max(1, (panel.n_pre - validation_start) // 3)
            with warnings.catch_warnings(record=True) as validation_warnings:
                selection = select_estimator(
                    panel, candidates, initial_window=validation_start,
                    horizon=validation_horizon,
                )
            validation = selection.validation
            validation.summary.to_csv(output_dir / f'{name}-validation-summary.csv')
            validation.scores.to_csv(output_dir / f'{name}-validation-folds.csv', index=False)
            validation.predictions.to_csv(output_dir / f'{name}-validation-predictions.csv', index=False)
            fig, axes = plot_backtest(
                validation, title=figure_details['case'] + ' · pretreatment validation',
                units=figure_details['units'],
            )
            plt.show()
            validation_files = export_figure(fig, 'validation')
            display_warnings('Pretreatment validation', validation_warnings)
            selected_fit = pd.Series({'Selected estimator': selection.selected_name,
                                      'Full-panel selected ATT': selection.estimate.att}, name='Selection')
            selected_fit.to_csv(output_dir / f'{name}-selection.csv', index_label='statistic')
            display_audit(
                'Validation scores, fold definitions and selected fit',
                validation.summary,
                validation.scores[['model', 'fold', 'training_end', 'validation_start',
                                   'validation_end', 'validation_rmse']],
                selected_fit,
            )
        """),
        markdown(
            "## Dependence on donor eligibility\n\n"
            "Each point removes one eligible donor and fully refits the estimator. "
            "Scenario ranges are not confidence intervals.\n\n"
            + details(
                "Donor exclusions, complete refitting and interpretation",
                """
            Remove **every eligible donor in turn** and completely refit three distinct
            estimator families: outcome SCM, the authors' two-stage ISC specification,
            and SDID. 2SISC repeats nearest-neighbor screening after each exclusion.
            This includes donors whose baseline weight was zero; no old weights are
            simply renormalized. The API accepts the other supported methods and
            named donor groups using the same protocol.

            The figure shows the baseline, every refitted ATT, and the largest absolute
            change. Scenario ranges are **not confidence intervals**. Neither a stable
            result nor a sign reversal establishes whether a donor is causally valid.
            These diagnostics should inform the study design, not selection of a
            donor pool that produces a preferred effect.
        """,
            )
        ),
        code(f"""
            with warnings.catch_warnings(record=True) as sensitivity_warnings:
                sensitivity = donor_sensitivity(panel, {{
                    'Outcome SCM': SyntheticControl,
                    '2SISC k=10': partial(TwoStageIndividualSC, n_neighbors=10),
                    'SDID': SyntheticDID,
                }})
            sensitivity.table.to_csv(output_dir / f'{{name}}-donor-sensitivity.csv', index=False)
            sensitivity.summary.to_csv(output_dir / f'{{name}}-donor-sensitivity-summary.csv')
            fig, axes = plot_donor_sensitivity(
                sensitivity, title=figure_details['case'] + ' · donor sensitivity',
                units=figure_details['units'],{donor_display_names}
            )
            plt.show()
            sensitivity_files = export_figure(fig, 'donor-sensitivity')
            display_warnings('Donor exclusions', sensitivity_warnings)
            display_audit('Donor sensitivity summary', sensitivity.summary)
        """),
        markdown("""
            ## SDID placebo inference and numerical convergence

            The original synthdid placebo variance estimator refits donor and time
            weights in each of 200 resamples. The seed is fixed. These standard errors
            apply to SDID alone; they are not intervals for the other methods.
            The bootstrap and jackknife are undefined with only one treated unit.
            California uses the authors' exact R sampling plan (seed 12345),
            reproducing the paper's standard error 8.4 at its printed precision.
            The other examples use NumPy seed 0 for reproducible new applications.

            The authors' 10,000-iteration cap can stop some draws before their
            objective-decrease criterion is met. We retain that reference result,
            then repeat **the identical 200 resamples** with up to 1,000,000 final-stage
            iterations and require every resampled weight fit to meet the original
            criterion. Penalties, tolerance, sparsification and starting weights
            are unchanged. The larger cap is a numerical extension of the reference
            software, not its exact finite-iteration replication.

            The table compares both standard errors and capped-draw counts. The
            original-cap SDID point estimate remains fixed for this comparison;
            the strict check applies to resampled weight fits. Meeting the original
            objective-decrease rule is not a global optimality certificate; dual
            gaps and draw-level diagnostics are exported for inspection. Reference-cap
            warnings remain visible; an unresolved extended fit stops the notebook.
        """),
        code("""
            with warnings.catch_warnings(record=True) as inference_warnings:
                if paper_uncertainty is not None:
                    np.testing.assert_allclose(results['Synthetic DID'].att, reference.att, atol=1e-12)
                    reference_uncertainty = paper_uncertainty
                else:
                    reference_uncertainty = SyntheticDID().inference(
                        panel, method='placebo', replications=200, random_state=0,
                        estimate=results['Synthetic DID'],
                    )
                uncertainty = SyntheticDID().inference(
                    panel, method='placebo', replications=200,
                    estimate=results['Synthetic DID'],
                    resamples=reference_uncertainty.resamples,
                    max_iter=1_000_000, require_convergence=True,
                )
            np.testing.assert_array_equal(uncertainty.resamples, reference_uncertainty.resamples)
            assert uncertainty.metadata['capped_draws'] == 0
            budget_comparison = pd.DataFrame([
                {
                    'Run': label,
                    'Final-stage iteration cap': fit.metadata['max_iter'],
                    'Capped draws': fit.metadata['capped_draws'],
                    'Placebo standard error': fit.standard_error,
                    'SE change from reference cap': fit.standard_error - reference_uncertainty.standard_error,
                    'Largest absolute draw change': np.max(np.abs(fit.draws - reference_uncertainty.draws)),
                }
                for label, fit in [('Reference cap', reference_uncertainty), ('Extended cap', uncertainty)]
            ]).set_index('Run')
            capped = reference_uncertainty.metadata['capped_draws']
            if capped:
                print(f'Original SDID iteration cap reached in {capped} of '
                      f'{len(reference_uncertainty.draws)} draws before the stopping rule.')
            display(budget_comparison)
            display_warnings('SDID inference', inference_warnings)
            budget_comparison.to_csv(output_dir / f'{name}-sdid-budget-comparison.csv')
            convergence = pd.concat({
                label: pd.json_normalize(fit.metadata['solver_diagnostics'])
                for label, fit in [('Reference cap', reference_uncertainty), ('Extended cap', uncertainty)]
            }, names=['Run', 'Draw']).reset_index()
            convergence.to_csv(output_dir / f'{name}-sdid-convergence.csv', index=False)
            inference = pd.Series({
                'SDID ATT': results['Synthetic DID'].att,
                'Placebo standard error': uncertainty.standard_error,
                '95% normal interval': uncertainty.confidence_interval(),
                'Replications': 200,
                'Final-stage iteration cap': uncertainty.metadata['max_iter'],
                'Capped resample fits': uncertainty.metadata['capped_draws'],
            }, name='SDID inference')
            display(inference)
            inference.to_csv(output_dir / f'{name}-sdid-inference.csv')
            np.savetxt(output_dir / f'{name}-sdid-resamples.csv', uncertainty.resamples, delimiter=',', fmt='%d')
            np.savetxt(output_dir / f'{name}-sdid-draws.csv', uncertainty.draws, delimiter=',')
            np.savetxt(output_dir / f'{name}-sdid-reference-draws.csv', reference_uncertainty.draws, delimiter=',')
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
            weights = weights.loc[weights.abs().max(axis=1).sort_values(ascending=False).index]
            weights.to_csv(output_dir / f'{name}-donor-weights.csv', index_label='donor')
            pd.DataFrame(warning_rows, columns=['Section', 'Category', 'Message', 'Count']).to_csv(
                output_dir / f'{name}-warnings.csv', index=False,
            )
            weights.head(15).round(4)
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

            `pysynth.placebo_test` provides in-space RMSPE-ratio diagnostics and a
            rank statistic. The default untreated-only design has different donor-pool sizes. The
            reassignment design uses all other units for each fit. A randomization
            interpretation requires a defensible assignment mechanism and a sharp null.
            It is not synthdid's placebo
            variance estimator. The original SCUL scanner-data and ISC/costofcare
            survey studies are not reproduced here. Re-run from a fresh kernel to
            reproduce the source checks, estimates and figures together.
        """),
    ]
    if name == "california":
        cells[-1:-1] = [
            markdown("""
                ## Could another state have produced California’s gap?

                Abadie, Diamond and Hainmueller assigned Proposition 99 to each of the 39 states
                in turn. The other 38 states supplied the donors each time. This section uses their
                smoking data with covariates. For each assignment we fit the predictor weights
                and donor weights afresh using the original pretreatment years. The historical
                comparison runs the original R Synth package. A second analysis runs the
                checked Python predictor optimizer on the same rows, states and years.
                Both use the authors’ predictor and donor order.
                [Study, Figures 4–8](https://economics.mit.edu/sites/default/files/publications/Synthetic%20Control%20Methods.pdf#page=9)
            """),
            code("""
                from pysynth import placebo_test, plot_placebo_gaps
                from pysynth.datasets import load_dataset
                from pysynth.replication import california_tobacco_estimator

                tobacco = load_dataset("california_tobacco")
                tobacco_panel = DATASETS["california_tobacco"].panel(tobacco)
                assignments = placebo_test(
                    partial(california_tobacco_estimator, tobacco, backend="r-synth"), tobacco_panel,
                    donor_pool="reassign", n_jobs=4,
                )
                reference_versions = pd.Series({
                    key: assignments.treated.metadata[key]
                    for key in ("reference_package", "reference_version", "r_version", "rpy2_version")
                })
                reference_versions.to_csv(output_dir / f'{name}-placebo-reference-versions.csv')
                assert reference_versions["reference_version"] == "1.1-10", reference_versions
                python_states = placebo_test(
                    partial(california_tobacco_estimator, tobacco, backend="python"), tobacco_panel,
                    donor_pool="reassign", n_jobs=4,
                )
                python_states.gaps.to_csv(output_dir / f'{name}-python-placebo-gaps.csv')
                python_states.table.to_csv(output_dir / f'{name}-python-placebo-ranks.csv')
                assignments.gaps.to_csv(output_dir / f'{name}-placebo-gaps.csv')
                assignments.table.to_csv(output_dir / f'{name}-placebo-ranks.csv')
                checks = []
                for multiplier, published_count in [(None, 38), (20, 34), (5, 29), (2, 19)]:
                    screened = assignments.with_pre_mspe_cutoff(multiplier)
                    python_screen = python_states.with_pre_mspe_cutoff(multiplier)
                    checks.append({
                        "pre_MSPE_screen": "All" if multiplier is None else f"{multiplier}×",
                        "paper_placebos": published_count,
                        "r_synth_placebos": int(screened.table.eligible.sum()) - 1,
                        "r_synth_rank": screened.rank, "r_synth_rank_fraction": screened.p_value,
                        "python_placebos": int(python_screen.table.eligible.sum()) - 1,
                        "python_rank": python_screen.rank, "python_rank_fraction": python_screen.p_value,
                    })
                assert assignments.rank == 1 and len(assignments.table) == 39
                checks = pd.DataFrame(checks)
                checks.to_csv(output_dir / f'{name}-placebo-screens.csv', index=False)
                print(f'Figure 8 rank: paper 1/39, R Synth {assignments.rank}/39, Python {python_states.rank}/39')
                for label, result in [("R Synth", assignments), ("Python", python_states)]:
                    print(f'{label} post/pre MSPE: {result.table.loc["California", "mspe_ratio"]:.1f} (paper ≈ 130)')
                    print(f'{label} year-2000 gap: {result.treated.effects.loc[2000]:.2f} (paper ≈ −26 packs)')
                display(checks)

                # Independently reconstruct California from the rounded donor weights in Table 2.
                printed = pd.Series({"Colorado": 0.164, "Connecticut": 0.069, "Montana": 0.199,
                                     "Nevada": 0.234, "Utah": 0.334})
                sales = tobacco.pivot(index="year", columns="state", values="cigsale")
                printed_gap = sales["California"] - sales[printed.index] @ printed
                printed_gap.to_csv(output_dir / f'{name}-table-2-reconstructed-gap.csv', header=["gap"])
                print(f'Published rounded weights alone give {printed_gap.loc[2000]:.2f} packs in 2000.')
            """),
            code("""
                fig, ax = plot_placebo_gaps(
                    assignments.with_pre_mspe_cutoff(5), treated_label="California",
                    title="California’s cigarette-sales gap among state placebos",
                    xlabel="Year", ylabel="Gap in annual packs per capita",
                    note="ADH (2010), Figures 6 and 8. At the same cutoff the paper retained 29 placebo states. "
                         "The published and R Synth unscreened ranks are both 1/39.",
                )
                gap_2000 = assignments.treated.effects.loc[2000]
                ax.text(0.025, 0.065, f"California in 2000   ·   paper ≈ −26   ·   R Synth {gap_2000:.2f}",
                        transform=ax.transAxes, fontsize=8, fontfamily=ax.xaxis.label.get_family(),
                        color=ax.lines[-1].get_color(),
                        va="bottom", ha="left")
                save_figure(fig, output_dir / f'{name}-placebo-paths')
                plt.show()
            """),
            markdown("""
                The cutoff compares **mean squared** pretreatment gaps. A fivefold cutoff on
                root-mean-squared gaps would allow a 25-fold difference on the paper’s scale.
                Screening changes the set of assignments in the denominator. California
                adopted the policy through a ballot initiative, so the all-state rank describes
                its extremeness under hypothetical uniform assignment. The figure and CSV
                report the fitted counts alongside the authors’ counts because predictor
                optimization can give different synthetic controls for some states. For the
                stated predictor years, the Stata companion file gives a 15–24 age share
                of 17.35% in California. Table 1 prints 17.40%. The authors’ MATLAB
                data give the same 17.35%. Their table weights donor-state means by population.
                Both backends use the authors’ predictor and state order. The Python default uses a
                single equal start for predictor importance. R Synth uses equal and regression-based
                starts and evaluates multiple optimizers against pretreatment outcome loss.
                [Original MATLAB example](https://www.mit.edu/~jhainm/synthpage.html)
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
