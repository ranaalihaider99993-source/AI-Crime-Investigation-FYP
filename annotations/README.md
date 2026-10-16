# Evidence-to-scenario annotations

`scenarios_template.csv` is a header-only template, not real ground truth.
`synthetic_scenarios.csv` contains 12 deliberately synthetic pipeline examples:
8 train, 2 validation, 2 test. No example is presented as a DCSASS narrative.

For real data, use one row per human-reviewed evidence/scenario pair:

- `source_id`: stable original source video identifier; all related clips and paraphrases stay in one split.
- `split`: train, val, or test, assigned by original source before annotation expansion.
- `evidence`: a consistent serialization of visible observations, classifier outputs (with uncertainty), detected objects (when a trained detector exists), and investigator text.
- `scenario`: a short cautious hypothesis supported by the evidence; identify uncertainty and avoid unsupported intent or identity claims.
- `provenance`: human_reviewed for reviewed real examples; synthetic for artificial examples.
- `reviewer`, `review_status`: document human review; training should use approved real examples only.

Before real training, verify licenses/consent, remove identifying details where appropriate,
check contradictions and near-duplicate scenes across splits, and obtain adequate independent
source coverage. This implementation checks exact evidence duplicates and source overlap;
it cannot automatically detect paraphrases or visually duplicate scenes. The vocabulary is
built on training rows only, with `<unk>` for unseen words. Targets are truncated to 46 words.
The tiny synthetic test is a wiring test, not a language-quality or investigation benchmark.

Inspect all actual DCSASS annotation files in notebook 01. Category/temporal labels do not
supply scenario targets or weapon bounding boxes. Availability is currently unverified.
If no boxes exist, choose a separate appropriately licensed Kaggle/Roboflow dataset with
box coordinates and class names, review its split integrity and object coverage, then plan
randomly initialized detector training in a separate task. No YOLO checkpoints are used.
