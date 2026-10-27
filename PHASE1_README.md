# Phase 1: BO foundations and worked example

Phase 1 establishes the shared concepts and a reproducible one-dimensional Bayesian Optimization (BO) loop. It is a learning and implementation milestone—not evidence that any acquisition strategy or stopping rule is generally reliable.

## Materials

- docs/bo_concept_map.md — term relationships and a worked numerical example.
- docs/bo_primer.md — introduction to BO, Gaussian processes, acquisition, recommendation, and stopping.
- docs/bo_gp_stopping_rules_study_guide.md — detailed GP calculation, hyperparameters, diagnostics, and stopping rules.
- docs/reading_list.md — annotated ten-paper reading list.
- docs/phase1_results.md — verified output, limitations, and questions passed to the next phase.
- docs/handwritten_note_english.md — cleaned English outline of the handwritten notes.
- examples/01_bo_intro.ipynb and examples/bo_intro.py — visual notebook and runnable script for the same BO loop.
- examples/02_lengthscale_sensitivity.py — diagnostic showing how GP length-scale assumptions change predictions.
- tests/test_bo_intro.py — correctness checks.
- results/phase1/ and assets/ — saved trace, summary, and regenerated figures.
- presentations/ — project slide decks and the bilingual speaker guide.

## Suggested order

1. Read docs/bo_concept_map.md, then docs/bo_primer.md.
2. Work through docs/bo_gp_stopping_rules_study_guide.md and consult docs/reading_list.md for original sources.
3. Run examples/bo_intro.py and examples/02_lengthscale_sensitivity.py.
4. Review docs/phase1_results.md for numerical results and limitations.

## Run

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python examples/bo_intro.py
python examples/02_lengthscale_sensitivity.py
python -m pytest -q
~~~

The fixed-seed example uses three initial observations and six LogEI decisions (nine objective evaluations total). It returns x=0.7494546, with dense-grid reference simple regret about 0.0004961. The early best-so-far plateau later improved, so stagnation alone is not proof that the search is done. See docs/phase1_results.md for what this does—and does not—show.
