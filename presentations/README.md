# Project presentations

`bo_project_explanation_en.pptx` is an English explanation of the current
Bayesian optimization project. It begins with the research problem and covers:

- the objective, GP surrogate, acquisition, stopping, and recommendation;
- the full sequential BO loop;
- a worked GP posterior calculation;
- a numerical UCB example;
- the effects of kernel, length scale, output scale, noise, mean, and ARD;
- marginal-likelihood fitting and calibration diagnostics;
- stopping-rule families and a minimal BoTorch structure;
- the Phase 1 completion and next implementation plan.

The fuller derivations and diagnostic discussion are in
`docs/bo_gp_stopping_rules_study_guide.md`.

`bo_stopping_rules_part_ii_phases_2_3_en.pptx` is the ten-slide continuation.
It covers the Phase 2 benchmark bank, the 3 x 3 Ising MRF, the 540-trajectory
Phase 3 characterization prototype, the MRF pilot, stopping-rule diagnostics,
and the next implementation steps.

`bo_stopping_rules_complete_update_en.pptx` is the 30-slide combined update.
It preserves the two earlier ten-slide explanations, removes the Chinese
speaker notes from the oldest version, appends the Phase 2 and Phase 3 results,
and uses continuous slide numbering from 01 through 30.
