# Project presentation

`bo_project_explanation_en.pptx` is a ten-slide explanation of the current
Bayesian optimization project. It begins directly with the research problem
and covers:

- the objective, GP surrogate, acquisition, stopping, and recommendation;
- the full sequential BO loop;
- a worked GP posterior calculation;
- a numerical UCB example;
- the effects of kernel, length scale, output scale, noise, mean, and ARD;
- marginal-likelihood fitting and calibration diagnostics;
- stopping-rule families and a minimal BoTorch structure;
- the Phase 1 completion and next implementation plan.

The deck includes speaker notes and source links. The fuller derivations and
diagnostic discussion are in
`docs/bo_gp_stopping_rules_study_guide.md`.
