# Validation evidence — canonical four-class preprocessing (schema v2)

Executed locally in Codex Cloud using Python 3.12.14 and PyTorch 2.14.1+cpu.

- `python -m pytest -q`: **11 passed**.
- `python -m compileall -q scripts tests`: passed.
- `git diff --check`: passed.

Regression coverage includes:

- Actual temporary MJPEG videos inside original-source directories named `.mp4`.
- The exact mirrored wrapper pattern `DCSASS Dataset/` versus
  `dcsass dataset/DCSASS Dataset/`. The generated fixture contains 52 physical video files;
  inspection retains **26 canonical clips / 13 original sources**, excludes 26 hash-verified
  mirror copies, and reports two unmapped Arson clips separately. These are fixture counts,
  **not measurements of DCSASS**.
- Repeatable manifests and exclusion reports; deterministic split selection despite reversed
  manifest row order; different source videos with equal segment basenames remain distinct.
- Equal-size but different-byte duplicate collisions fail and produce a conflict report.
- Duplicate canonical clips, missing files, legacy manifests, invalid class counts, and
  source leakage are rejected.
- A fixture with 10 Burglary / 12 Stealing / 8 Fighting / 9 Vandalism sources is capped at
  eight per class: actual split reports show **6 train / 1 validation / 1 test per class**
  and empty source-overlap lists. Only one clip per source is selected.
- Four-output randomly initialized CNN and GRU forward passes, optimizer updates,
  two-epoch CPU training/checkpoint reload/evaluation on synthetic test inputs.
- Changed canonical split membership and old five-class checkpoints fail evaluation.
- Synthetic annotation leakage rejection, all four notebook schemas, every notebook code
  cell's Python syntax, and execution of notebook 02's actual preparation/guarded cells on
  the mirrored fixture. Both review/training flags remain false; no training is launched
  by those notebook preparation cells.

`reports/smoke_metrics.json` retains the **historical five-class synthetic CPU experiment**
from the earlier implementation. It is explicitly labeled historical, not evidence for
this corrected four-class pipeline or real crime performance. No values were invented or
reinterpreted as real-data results. Cloud package versions are in
`requirements-cloud-tested.txt`.

## User-reported real-data observations

The user reported 33,278 raw manifest rows, 16,639 unique clips, and 520 original sources
following canonicalization. The mapped counts were Burglary 1,504 clips / 47 sources,
Stealing 2,048 / 64, Fighting 256 / 8, and Vandalism 928 / 29; 11,903 clips were unmapped.
These observations motivate the fix but are **not independently verified in this machine**.
The generated inspection report computes current counts rather than hard-coding them.

## Colab steps still required

No real DCSASS download, corrected full-dataset inspection, Colab execution, GPU training,
Drive authorization, or real-data accuracy/F1/confusion matrix was measured here.

1. Reopen notebook 01 and run its setup to update a clean checkout and clear stale module
   imports, or start a fresh Colab runtime. Resolve any local-edit warning without discarding
   your work. Run inspection and review `manifest.csv`, `excluded_duplicates.csv`, and
   `inspection.json` under `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2`.
   Hashing all duplicate copies reads their full contents and can take time.
2. Inspect actual annotation schemas and canonical source IDs. Canonical names do not prove
   that unrelated IDs never depict the same incident; review near duplicates.
3. In notebook 02, set only `SOURCE_MAPPING_REVIEWED=True` to generate the split report.
   Review actual per-class/per-split source and clip counts and zero source intersections.
4. Only after review, enable the small smoke flag. Defaults are three sources per class,
   one representative clip each, two epochs: **12 clips total**, split 1/1/1 per class.
   Normal remains unmapped. Old duplicated manifests and old five-class checkpoints must
   not be reused. The new Drive subfolder isolates corrected artifacts.
5. Report the measured results with their small-sample limits. Do not enlarge to expensive
   full training without separate approval. No conclusion about generalization can rest
   on a single held-out Fighting source. GRU real-data training is outside this change.

Checkpoints now record a hash of canonical clip/source/label/split membership; evaluation
rejects changed membership even if each new split is individually disjoint. Physical cache
path relocation alone does not change that hash, but missing physical files still fail.
