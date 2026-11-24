API reference
=============

Panel and results
-----------------

.. automodule:: pysynth.panel
   :members:
   :undoc-members:

.. automodule:: pysynth.results
   :members: Estimate, Estimator, compare

Core and reference methods
--------------------------

.. automodule:: pysynth.classic
   :members:

.. automodule:: pysynth.scul
   :members:

.. automodule:: pysynth.individual
   :members:
   :inherited-members:

.. automodule:: pysynth.sdid
   :members:

Published and working-paper SCM specifications
----------------------------------------------

.. automodule:: pysynth.backends
   :members: PredictorSyntheticControl, AugmentedSyntheticControl, PenalizedSyntheticControl, RobustSyntheticControl, DifferencedSyntheticControl

Backend compatibility
---------------------

.. autoclass:: pysynth.backends.PySyncon
   :members: fit

.. autoclass:: pysynth.backends.SCM
   :members: fit

Publication figures
-------------------

.. automodule:: pysynth.plotting
   :members: LiteratureBenchmark, ReferenceComparison, reference_comparison_table, plot_paths, plot_effects, plot_comparison, publication_style, publication_title_font, save_figure

Inference
---------

.. automodule:: pysynth.inference
   :members:

Validation, selection and sensitivity
----------------------------------------

.. automodule:: pysynth.validation
   :members: backtest, select_estimator, BacktestResult, SelectionResult

.. automodule:: pysynth.sensitivity
   :members: donor_sensitivity, DonorSensitivityResult

.. automodule:: pysynth.evaluation_plotting
   :members: plot_backtest, plot_donor_sensitivity

Data and replications
---------------------

.. automodule:: pysynth.datasets
   :members:

.. automodule:: pysynth.replication
   :members:
