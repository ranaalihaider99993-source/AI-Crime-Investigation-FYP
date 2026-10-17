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

## Subsequent automatic-report and reviewed-smoke launcher update

The latest local run has **22 tests passed**. The prior 11-test result above describes
schema-v2 preprocessing; this update adds coverage for the report/launcher workflow.

- Notebook 02 now reuses notebook 01 artifacts instead of repeating whole-dataset inspection.
  It creates an unapproved candidate split while **both flags remain False**.
- The report clearly separates automatic checks, pending human review, hardware availability
  and no training execution. It includes current readable source counts, per-class split
  counts, source overlap, a mapped-source review CSV and selected-frame review sheet.
- Automatic checks bind physical paths to canonical clip/source IDs, validate saved count
  and duplicate-audit consistency, reject insufficient/unrecognized source names, decode
  actual selected frames and reject byte-identical selected videos across splits.
  Full-dataset duplicate hashes are recorded inspection evidence, not rehashed by preparation.
- Before a reviewed launch, inspection hashes, split/config and selected video bytes are
  checked again. Missing GPU, pending human approval, failed validation, changed inputs,
  and configuration expansion beyond two epochs/three sources per class all block training.
- Local tests exercised fresh two-epoch CNN training and actual evaluation **on generated
  temporary videos only**, using an explicit CPU test override. Unique run directories
  preserve prior checkpoints; a simulated disconnection cannot create a completed pointer.
- The same selected-frame preflight exposed repeated-seek failures on one-frame clips.
  The loader now caches decoded sample positions so repeated frame indices work correctly.
- Notebook 02 default execution was tested without its Colab/UI/authentication cells.
  Rendering was stubbed; actual preparation and guarded launch/result cells ran. Neither
  flag became True, no model trained, and a candidate/report was created as intended.
- A fresh actual Chromium/Playwright navigation to notebook 02's Colab URL failed with
  `net::ERR_TUNNEL_CONNECTION_FAILED`. No Colab runtime or Google Drive session was accessible.

The user reports successfully running notebook 01 and notebook 02 on a T4 runtime, with
both launch/review flags still False. These are user-reported executions. Their Drive
artifacts have **not been read or independently validated from this cloud machine**.
No real-data GPU training, Drive checkpoint, real accuracy, macro F1 or per-class results
were produced in this update. The new report must be run on those actual files in Colab.

Current next action supersedes steps 3–4 above: reopen the latest notebook 02 from GitHub,
Run all with flags False, review the automatically generated report and source sheet,
then enable both flags only after human review and run the launch and result cells.
Actual successful runs are stored under `classifier/runs/<timestamp>-<id>/`; notebook 04
now reads the latest completed run pointer. Full training remains blocked by launcher bounds.

## Dataset-root/path validation fix

Latest complete suite: **34 tests passed** in the cloud CPU environment.
`python -m compileall -q scripts tests` and whitespace checks also pass.

Root cause: notebook 02 compared persisted absolute manifest paths directly with the
current `DATA` directory. A valid path from the cache version's primary `DCSASS Dataset/`
copy could lie outside a selected inner `dcsass dataset/DCSASS Dataset/` directory. Across
sessions the current KaggleHub cache version or mounted path can also differ. The user's
exact selected `DATA` value was not available to this agent; both mismatch cases are
reproduced by the regression tests.

Changes and verified coverage:

- Both notebooks normalize recognized package/wrapper layouts and print returned versus
  verified roots. No arbitrary common-ancestor expansion or containment bypass is used.
- The exact reported Abuse path is reproduced under a temporary
  `root/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1` tree, with the inner
  mirror selected. Validation passes without moving files or changing source IDs.
- Legacy schema-v2 snapshots (without root metadata) relocate safely from cache version 1
  to version 2 using their recorded duplicate SHA-256 evidence. A test verifies that old
  out-of-bound video bytes are never read even when those files still exist.
- A recorded custom root can relocate after the old download is deleted. Kaggle mounted
  input layouts are supported. Canonical split signatures remain identical before/after.
- Rebinding preserves original artifacts byte-for-byte and writes a separate runtime
  inspection snapshot and relocation audit. Original and resolved artifact hashes are
  both bound to the existing guarded training launcher.
- Missing files, altered bytes (including equal-size changes), unknown external path
  provenance, wrong relative identities, missing hash evidence, traversal, category-root
  selection and file/wrapper symlink escapes fail closed. No partially rebound manifest
  is saved on these failures.
- New inspections record the verified `dataset_root` and SHA-256 for every retained clip,
  including single-copy clips, so later relocation has content evidence.
- The human-review/training flags remain False. Existing four-class, duplicate, source
  leakage, preflight, GPU-gate and CPU model tests continue to pass.

No real Colab GPU training or real-data metrics were executed or observed in this fix.
For a location-only failure, reopen updated notebook 02, restart its session and Run all
with both flags False. Notebook 01 is unnecessary when saved identities/copy counts and
content hashes match; rerun it only when the explicit report says evidence/content differs
or cannot be verified. Hash verification of relocated copies may take time.

## Temporal selected-clip review update

Complete local suite: **42 tests passed**. Existing preprocessing/root resolution and model
smoke tests remain intact. New tests use generated temporary MJPEG videos, not DCSASS.

- A 12-clip temporal review samples 12 distinct positions through each synthetic 24-frame
  sequence, shows relative timestamps and renders self-contained HTML/JSON/CSV reports.
- Similar synthetic sequences written with different container timing are flagged as
  **uncertain** scene matches; the technical checks pass without human approval.
- Byte-identical videos under distinct source IDs cause technical FAIL. Source overlap,
  incorrect class assignment, outside-root paths and unreadable video fail safely.
- One-frame/low-texture clips report limited evidence instead of inventing temporal coverage.
- Human review and semantic/incident independence remain PENDING/UNVERIFIED, training remains
  unexecuted, and the notebook defaults leave both flags False.
- Guarded launch rejects temporal reports for another split or with technical failures.
  Reported clip SHA-256 values must also match the preflight's selected video hashes.
- Tests exercise the actual new notebook review cell with rendering stubbed locally, together
  with existing preparation and disabled launch/result cells. No Colab execution is implied.

A fresh browser attempt again failed with `net::ERR_TUNNEL_CONNECTION_FAILED`. The user's
successful T4 automatic validation is user-reported; the actual selected clips/Drive report
were inaccessible here. No actual DCSASS temporal findings, human approval, GPU training,
accuracy or new real checkpoint are claimed.

A synthetic demonstration report was generated outside the checkout at
`/workspace/artifacts/dcsass-review-demo/review.html`. It contains random generated videos
and one deliberately repeated visual sequence. Its findings are **not** DCSASS findings.
For the real report, reopen notebook 02 from GitHub, run all with both flags False, and read
`smoke_preparation/temporal_review/review.html` in Drive. Resolve technical failures and
investigate uncertain cases before explicit human approval; only then is the existing
12-clip/two-epoch GPU smoke the next permitted training experiment.

The self-contained synthetic HTML was rendered locally with Chromium: 12 cards and 12 embedded contact sheets were present, with the synthetic-only scope visible. This offline rendering check is not evidence of Colab access.

## Real-review summary and export update

Complete available local suite: **44 tests passed**. Python compilation and whitespace
checks passed. These are regression checks on temporary generated videos, not a real-data
review or GPU training run.

The existing temporal review now additionally writes a concise `review_summary.md`, embeds
structured summary counts/pair findings in `review.json`, records every selected label and
incident relationship as pending in `human_review_queue.csv`, and creates a bounded
`review_artifacts.zip` export. The bundle includes HTML, JSON, summary, case CSV, pending
queue, selected-manifest snapshot and file-integrity metadata. Raw videos, checkpoints,
credentials and synthetic demonstration files are not exported.

Tests verify that even zero-alert clips retain pending semantic and incident reviews,
uncertain cross-split pairs appear by actual source ID in the summary, confirmed byte
copies are technical blockers, ZIP member contents/hashes match the current report, and
an invalid rerun cannot export an old selected-manifest snapshot. The notebook's actual
export cell is exercised with downloading disabled; both training flags remain False.

A fresh Colab browser attempt failed with `net::ERR_TUNNEL_CONNECTION_FAILED`. No Drive
connector was available. The user's 8/8 automatic checks and T4 availability remain
user-reported. The agent did not access the real selected clips, generate their real report
in Drive, list actual suspicious pairs, approve semantics/incidents, or train a model.
No real accuracy or checkpoint results are claimed. No new synthetic demonstration was
presented as fulfilling the real-data request.

To generate the real artifacts, reopen latest notebook 02 and Run all with both flags False.
The temporal cell uses the actual `AUTOMATIC_REPORT` selection and verified dataset root.
Output: `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/temporal_review/`.
Download `review_artifacts.zip` from Drive, or set only `DOWNLOAD_REAL_REVIEW=True` in the
export cell. Read the summary/full HTML and resolve human-review cases before considering
any later experiment; this request does not authorize training.

## Existing-original playback and bounded package update

Latest complete local suite: **58 tests passed**. Python compilation and whitespace checks
also pass. New tests use generated temporary videos; quota values are test fixtures, not a
measurement of the user's Drive space.

- New helpers read the existing saved candidate/preflight only. Regression tests check that
  candidate bytes remain identical and that the export workflow never invokes the splitter.
- All 12 HTML5 players embed exact original bytes and show source, dataset class, split and
  original filename. The eight user-specified low-motion identifiers are matched ignoring
  `_x264`, and flagged as user-reported rather than semantically verified.
- Repeated video basenames remain unchanged under split/source subdirectories. A package
  contains exactly 12 original video members plus five explicit metadata/checklist/player
  files. Tests compare every archived video byte and SHA-256 with its source.
- Missing/changed candidate assignments/content, escaped paths, oversized clips/total,
  inline overflow, stale temporal-selection binding, unknown/insufficient quota and low
  filesystem space block the corresponding operation. An existing ZIP is preserved.
- A content change during copying fails integrity checks and removes the incomplete archive.
- Default execution of the actual new notebook cells is covered with UI rendering stubbed:
  all 12 players are prepared, export/download opt-ins remain False, both training flags
  remain False, and no training/package is started by default.
- Real export requires live Google Drive API storageQuota metadata for the same account as
  mounted MyDrive, plus output/local filesystem space. It is not inferred from FUSE disk
  usage. Unknown/non-finite quota blocks export. Authentication/quota/playback on the user's
  actual Colab runtime were not available or independently verified here.

No actual DCSASS original clips were viewed/copied here, no real Drive package was generated,
and no human label/incident approval, CNN training or accuracy is claimed. In Colab reopen
notebook 02, run setup/Drive mounting only, then `original-video-players` directly. Do NOT
rerun Automatic preparation/Run all for this workflow. Export with `export-original-videos`
only when requested; preserve the same saved candidate. Destination:
`MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/original_video_review/original_video_review.zip`.

### Source-context review regression coverage

The additive `scripts/context_review.py` workflow binds to the existing passing
validation report, candidate-file hash/split signature, full inspection artifact
hashes, and saved clip SHA-256 evidence. It does not invoke the splitter or trainer.
It selects at most one nearest numeric predecessor and successor per source;
unknown/ambiguous ordering gets no guessed neighbors. Gaps/boundaries are explicit.
Path containment and canonical identities are checked before reading context
content, and containment/content hashes are checked again before playback.

Local tests cover numeric ordering (1, 2, 10), source boundaries, unchanged candidate
bytes, matching source/split metadata, pending checklists, unique output folders,
stale inspection artifacts, altered video content and symlink escapes. The actual
notebook context cell also runs against the duplicated-root test fixture with both
training flags False. These are local generated-video tests, not real DCSASS review
or Colab execution. Semantic labels and incident independence remain unverified.

### Consolidated review dashboard

The dashboard reads existing context metadata and the human checklist, revalidates
saved candidate/context/inspection identities and video hashes, and imports only
matching temporal-review evidence. It does not regenerate the candidate or perform
semantic inference. Reports keep approval PENDING irrespective of checklist answers.
Exact full source references in checklist relationships receive attributed cross-split
links; ambiguous references require human investigation. Source 056's supplied
observation remains user-reported and unresolved when not uniquely identifiable.
HTML is escaped; spreadsheet formula prefixes in CSV are neutralized. Input checksums
are retained in report JSON. Every output is a new snapshot; inputs stay untouched.
Regression coverage includes notes preservation, low-motion flags, identity tampering,
ambiguous source 056, HTML escaping, existing temporal cross-split findings, stale
evidence rejection, and actual default notebook-cell execution with training disabled.
These checks do not constitute actual DCSASS human review or Colab execution.
