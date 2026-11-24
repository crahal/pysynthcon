# References and attribution

1. Abadie, A. and Gardeazabal, J. (2003). The Economic Costs of Conflict: A Case
   Study of the Basque Country. *American Economic Review*, 93(1), 113–132.
   [DOI](https://doi.org/10.1257/000282803321455188).
2. Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic Control Methods
   for Comparative Case Studies: Estimating the Effect of California's Tobacco
   Control Program. *JASA*, 105(490), 493–505.
   [DOI](https://doi.org/10.1198/jasa.2009.ap08746).
3. Abadie, A., Diamond, A. and Hainmueller, J. (2015). Comparative Politics and
   the Synthetic Control Method. *AJPS*, 59(2), 495–510.
   [DOI](https://doi.org/10.1111/ajps.12116).
   [2025 erratum](https://economics.mit.edu/sites/default/files/2025-09/Comparative%20Politics%20and%20the%20Synthetic%20Control%20Method%20erratum.pdf):
   the empirical GDP series is PPP-adjusted current US dollars.
4. Hollingsworth, A. and Wing, C. (2020). Tactics for design and inference in
   synthetic control studies: An applied example using high-dimensional data.
   [Working paper](https://doi.org/10.31235/osf.io/fc9xt),
   [SCUL documentation](https://hollina.github.io/scul/).
5. Vagni, G. and Breen, R. (2021). Earnings and Income Penalties for Motherhood:
   Estimates for British Women Using the Individual Synthetic Control Method.
   *European Sociological Review*, 37(5), 834–848.
   [Article](https://academic.oup.com/esr/article/37/5/834/6325498),
   [costofcare](https://github.com/centre-for-care/costofcare),
   [author code](https://github.com/giacomovagni/IndividualSyntheticControl).
6. Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S.
   (2021). Synthetic Difference-in-Differences. *AER*, 111(12), 4088–4118.
   [DOI](https://doi.org/10.1257/aer.20190159),
   [synthdid](https://github.com/synth-inference/synthdid).
7. Ben-Michael, E., Feller, A. and Rothstein, J. (2021). The Augmented Synthetic
   Control Method. *JASA*, 116(536), 1789–1803.
   [DOI](https://doi.org/10.1080/01621459.2021.1929245).
8. Abadie, A. and L'Hour, J. (2021). A Penalized Synthetic Control Estimator for
   Disaggregated Data. *JASA*, 116(536), 1817–1834.
   [DOI](https://doi.org/10.1080/01621459.2021.1971535).
9. Amjad, M., Shah, D. and Shen, D. (2018). Robust Synthetic Control.
   *Journal of Machine Learning Research*, 19(22), 1–51.
   [Paper](https://jmlr.org/papers/v19/17-777.html).
10. Petrillo, M., Valdenegro, D., Rahal, C., Zhang, Y., Pryce, G. and Bennett,
    M. R. (2025). Estimating the Cost of Informal Care with a Novel Two-Stage
    Approach to Individual Synthetic Control. *Sheffield Economic Research
    Paper Series*, 2025004.
    [Paper](https://eprints.whiterose.ac.uk/id/eprint/224118/1/Binder2025004.pdf),
    [code](https://github.com/centre-for-care/costofcare).
11. Rho, S., Tang, A., Bergam, N., Cummings, R. and Misra, V. (2025).
    ClusterSC: Advancing Synthetic Control with Donor Selection.
    *Proceedings of AISTATS*, PMLR 258, 109–117.
    [Paper and software](https://proceedings.mlr.press/v258/rho25a.html).

12. Engelbrektson, O. (2021). Why Synthetic Control estimators are biased and
    what to do about it: Introducing Relaxed and Penalized Synthetic Controls.
    Working paper. [Preprint](https://arxiv.org/abs/2111.10784).

The two-stage approach is attributed to Petrillo et al.; the default follows
their Python mean-predictor fitting specification. An explicitly selected
MSCMT/resampling variant remains available.
ClusterSC is related work, not an estimator implemented by this release.

Software foundations: [pysyncon](https://github.com/sdfordham/pysyncon), by
Stiofán Fordham and contributors; [SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods),
by Oscar Engelbrektson and contributors. Cite the method papers and the
software used, not only this wrapper. Repository citation metadata are in
`CITATION.cff`; copied/adapted permissive-source notices are in
`THIRD_PARTY_NOTICES.md`.
