# AI-Based Crime Investigation and Scenario Generation System Using Machine Learning and Blockchain

Final Year Project using Machine Learning and Blockchain.

This phase provides reproducible **from-scratch PyTorch training pipelines** and Google
Colab notebooks. The original repository description was “AI-Based Crime Investigation
and Scenario Generation System.” Frontend, live camera, object-detector training,
multimodal fusion, blockchain, and investigation reports are future work.
There are no pretrained weights, external AI inference calls, or fabricated accuracy results.

## Start in Google Colab

1. Ensure this commit is pushed to the existing GitHub repository (a local commit alone is not visible to Colab).
2. Go to **https://colab.research.google.com/**, click **File → Open notebook → GitHub**,
   enter `ranaalihaider99993-source/AI-Crime-Investigation-FYP`, and open
   **`notebooks/01_Dataset_Inspection.ipynb` first**.
3. Click **Runtime → Run all**, then authorize Google Drive mounting when prompted.
   GPU is optional for inspection. For training choose **Runtime → Change runtime type → T4 GPU**.
4. The notebook installs requirements, checks the device, and downloads
   `mateohervas/dcsass-dataset` using KaggleHub. It does not access a Windows drive or require a ZIP upload.
   If Kaggle requires authentication, use KaggleHub's supported login or secure Colab secret handling.
   Never paste credentials into committed cells.
5. Review the generated `inspection.json` and `manifest.csv` in
   `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/inspection`. Inspect **all** candidate annotation files;
   the notebook previews only the first 20. Reports contain observed counts, not assumed statistics.
6. If notebook 01 already produced the schema-v2 files, reopen the latest
   [notebook 02 in Colab](https://colab.research.google.com/github/ranaalihaider99993-source/AI-Crime-Investigation-FYP/blob/main/notebooks/02_Crime_Classification_Training.ipynb),
   choose a T4 runtime, and **Run all with both flags False**. Notebook 02 now reuses the
   saved inspection, automatically prepares a candidate split, and displays a PASS/FAIL
   report and selected-frame review sheet. It does not approve mappings or start training.
7. Review `smoke_preparation/validation.md`, `source_review.csv`, and `review_sheet.jpg`.
   The added temporal-review cell then displays `smoke_preparation/temporal_review/review.html`,
   with up to 12 distinct positions per clip, relative timestamps and highlighted cases.
   Check class assignments, original-source conventions and related incidents across source
   IDs. Compare suspicious pairs in the temporal report and inspect original videos if needed.
   If a technical failure or unresolved semantic/incident concern remains, leave both flags False.
8. Only after your review, set **both** `SOURCE_MAPPING_REVIEWED=True` and
   `RUN_PROTOTYPE_TRAINING=True` in the launch cell, then run that cell and the result cell.
   The GPU-required launcher permits only **2 epochs / 3 sources per class**, creates fresh
   random weights, and evaluates held-out sources. It rejects changed reports/data, missing
   GPUs and expanded training configurations. This 12-clip experiment checks execution only.
9. Actual checkpoint paths, accuracy, macro F1, confusion matrix and per-class results are
   printed after successful execution. Set `DOWNLOAD_CHECKPOINT=True` in the result cell
   to download `crime_classifier.pth` for later Django integration. Notebook 04 can repeat
   evaluation using the latest successfully completed smoke run.
10. Notebook 03 is an **optional, separate synthetic GRU pipeline test**. It is unnecessary
    for CNN training/evaluation and provides no real scenario-generation evidence.

Run cells from top to bottom. Colab is interactive and Drive mounting requires your action.
If Drive mounting fails, resolve that error before training; the notebook does not silently
fall back to ephemeral storage in Colab. Outside Colab it uses ignored `artifacts/`.

## Repository layout

```text
notebooks/01_Dataset_Inspection.ipynb
notebooks/02_Crime_Classification_Training.ipynb
notebooks/03_Scenario_GRU_Training.ipynb
notebooks/04_Model_Evaluation.ipynb
scripts/data.py           # real-file inspection and grouped splitting
scripts/dataset_paths.py  # verified KaggleHub roots and hash-checked path rebasing
scripts/preprocess.py     # preprocessing CLI
scripts/classifier.py     # CNN, video sampling, training and evaluation
scripts/smoke.py          # automatic report, human gate and bounded GPU launch
scripts/review.py         # temporal visual report and uncertain similarity flags
scripts/original_review.py # original HTML5 players and bounded 12-video export
scripts/scenario.py       # GRU encoder-decoder, training and inference
scripts/evaluate.py       # held-out evaluation CLI
scripts/common.py         # reproducibility settings
configs/prototype.json
annotations/              # human annotation template + synthetic examples
models/                   # model documentation; weights ignored
reports/                  # validation evidence and remaining checks
tests/                    # temporary video and CPU model tests
requirements.txt
requirements-cloud-tested.txt
```

## Data and classifier design

DCSASS: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

Inspection checks `is_file()` before accepting video extensions, counts class-mapped and
unmapped videos, records extension counts and annotation candidates, and decodes the first
frame of each clip. This is not a full corruption scan: training raises an error if any
sampled frame is unreadable. Counts and actual annotation availability remain unknown until
notebook 01 executes against the dataset. Unmapped classes are excluded from classifier training.

Current prototype classes: **Burglary, Stealing, Fighting, Vandalism**.
`Normal` and all other categories are unmapped and excluded from training for now.
The four-class CNN rejects checkpoints with the old five-class mapping. The three-layer CNN
starts with random weights and averages logits over three uniformly sampled RGB frames
(resized to 64×64 and scaled to [0,1]). It is an intentionally limited clip classifier,
not a temporal detector; category labels may include normal frames within crime clips.

Inspection strips repeated leading `DCSASS Dataset` wrappers case-insensitively and
lowercases the remaining relative clip key. Thus `DCSASS Dataset/...` and
`dcsass dataset/DCSASS Dataset/...` identify the same clip. It preserves category and
source directory components: equal segment basenames from different sources are distinct.
Among copies, it retains the shortest relative path, then case-insensitive/exact lexical
order. SHA-256 verifies every excluded copy; conflicting bytes stop inspection and write
`duplicate_conflict.json`. New inspections hash every retained clip and duplicate copy, which can take time; decoding
runs only for the retained copies. This is path-identity deduplication, not detection of
similar scenes stored under unrelated names. The new selected-clip preflight also rejects
byte-identical videos across different source IDs/splits, but cannot detect re-encoded duplicates. `excluded_duplicates.csv` preserves each
excluded path, selected path, canonical key, hash, and reason. No dataset files are deleted.

Original `.mp4` parent directory basenames identify source videos; flat
`*_x264_N.mp4` names group by `*_x264`. IDs are canonical lowercase, independent of dataset
root and category directories. Conflicting source labels, duplicate canonical/physical
clips, missing files, invalid class counts, legacy manifests, and cross-split source leakage
fail explicitly. Unknown naming conventions or near-duplicate incidents still need review.
Regenerate **all** old manifests and splits: schema v2 requires `canonical_clip`.

The prototype uses one seeded representative clip per original source, equal source counts
per class. Sources and clip candidates are sorted before seeded selection, so manifest row
order cannot change the split. Selection is capped at the least available readable class.
Each class needs at least three independent sources; otherwise it fails. The split policy
assigns `max(1, floor(0.2*n))` sources each to validation/test and the rest to training.
Frames are decoded after splitting; test data are not used for early stopping.

`candidate_splits.report.json` in notebook 02 (or `<split-file>.report.json` from the CLI)
records available readable clips/sources, exclusions, selected
counts **per class and per split**, and the actual source-intersection lists (all must be
empty). The notebook smoke uses 3 sources per class: **1 train / 1 validation / 1 test**,
12 clips total. If 8 sources per class are selected, counts are **6 / 1 / 1**. A held-out
Fighting clip from only one source cannot establish reliable performance. Enlarging other
classes does not create additional independent Fighting sources.

### User-reported DCSASS inspection (not independently measured here)

The user found **33,278 raw rows, 16,639 unique clips, and 520 canonical original sources**.
The duplicate dataset-root copies explain the doubled rows.

| Mapped category | Unique clips | Original sources |
| --- | ---: | ---: |
| Burglary | 1,504 | 47 |
| Stealing | 2,048 | 64 |
| Fighting | 256 | 8 |
| Vandalism | 928 | 29 |

The remaining **11,903 unique clips are unmapped**. These observations are not hard-coded
assertions about future Kaggle versions. Regenerated `inspection.json` reports the actual
current raw/unique counts, all-source counts, mapped counts, unmapped source/clip totals and
unmapped category-name candidates separately. No real-data accuracy is claimed.

Training uses Adam, cross entropy, validation-loss early stopping, and best-checkpoint saving
as `crime_classifier.pth`. Epoch histories include real loss, accuracy, macro precision,
recall/F1, confusion matrices and per-class reports. The config and class order are saved
with the checkpoint. A canonical split-membership hash prevents evaluation against changed
clip/source/label/split assignments, even if physical cache paths change. Record the dataset version, reviewed manifest and package versions
for each experiment. Exact reproducibility across GPU/platform/library versions is not guaranteed.

## Neural scenarios and detection feasibility

The randomly initialized embedding + GRU encoder-decoder uses teacher forcing during
training and greedy decoding for inference. Its vocabulary is learned only from training
pairs. The included synthetic examples validate execution, not useful scenario generation.
See `annotations/README.md` for the real paired annotation workflow. A sufficiently large,
human-reviewed evidence-to-scenario dataset is required before reliable generation can
be evaluated. Outputs can hallucinate or fail to express uncertainty and always need review.
This phase serializes hypothetical evidence as text; it does not implement visual/text fusion.

DCSASS bounding-box and narrative annotation availability is **unverified in this environment**.
Notebook 01 records annotation candidates for schema review. If boxes are unavailable,
a separate annotated detection dataset is required; do not fabricate boxes. No detector is trained.

## Checkpoint persistence

The notebook 02 launcher writes each reviewed run under
`MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/classifier/runs/<timestamp>-<id>/`.
The best checkpoint, epoch history, exact split snapshot, experiment metadata, review
attestation and actual test metrics are retained there. A `latest_completed.json` pointer
is updated **only after training and evaluation both succeed**. Interrupted/failed runs
cannot become the latest completed experiment. Earlier successful runs are not overwritten
or displayed as a new result after a failed invocation. Training always starts from scratch.
The legacy CLI trainer still writes directly to its specified output folder; use a new
folder to preserve previous CLI experiments. GRU outputs remain in the separate `scenario/` folder.
Drive retains files after disconnect; processes and the in-progress epoch do not persist.
Optimizer-state continuation is not implemented. Saved models can be evaluated or used
for inference after reconnecting without retraining. Redownload DCSASS when needed; if its
cache path changes, regenerate/review the manifest with the same seed and **verify identical
source/test membership** before evaluation. Do not silently reassign the test set. The new Drive subfolder isolates corrected
manifests and checkpoints from the old duplicated/five-class artifacts. In a reused Colab
runtime, notebooks 01/02 fast-forward a clean checkout and evict stale project module
imports; local checkout edits cause an error instead of being overwritten.

## Recovering from a dataset-root/path mismatch

Saved manifests contain absolute paths, while `kagglehub.dataset_download()` may return
an inner wrapper directory or a different cache/mounted location in another session.
The old direct containment comparison could therefore reject valid saved dataset paths.
Both notebooks now print the returned path and the **verified dataset package root**.

`resolve_dataset_root` recognizes only this dataset's exact KaggleHub cache identity
(`kagglehub/datasets/mateohervas/dcsass-dataset/versions/<number>`) and Kaggle mounted-input
layouts. It may lift known `DCSASS Dataset` wrappers to that package boundary. It never
selects a filesystem common ancestor or ascends a custom download directory merely because
its name resembles a wrapper. Category/source subdirectories are rejected as download roots.
Resolved video paths, including symlink targets, must remain inside the verified root.

Notebook 02 normally **does not require rerunning notebook 01** for a location-only mismatch:

1. Reopen the latest GitHub notebook 02 (an already-open/Drive copy may have older cells).
2. Restart the Colab session to clear old variables/imports, retain/select the T4 runtime,
   and **Run all**. The setup updates a clean checkout. Authorize Drive mounting as prompted.
3. Leave both training/review flags **False**. Read the root-resolution check in the report.
   It records the selected path, verified root, relocation count and SHA-256 rechecks.
4. If the download location changed, notebook 02 writes a separate verified snapshot under
   `smoke_preparation/resolved_inspection/`, including `path_relocations.csv`. The original
   `inspection/` files are unchanged. Canonical clip/source IDs, labels, readability evidence,
   duplicate hashes and canonical split membership are preserved.

For relocation, current files must match the saved canonical relative identities and copy
counts, exist within the verified current root, and match recorded SHA-256 content evidence.
Old files outside that root are treated only as historical path strings and are never read.
The legacy two-copy inspection already records a hash for each clip in its duplicate audit;
new notebook 01 inspections also record `dataset_root` and `clip_sha256` for single-copy clips.
Rechecking relocated copies streams their bytes and can take time.

**Rerun notebook 01 only if instructed by a failed check**: missing content/hash evidence,
a changed dataset release or copy layout, differing hashes, or invalid canonical identities
cannot be repaired by trusting paths alone. An unrecognized old root without recorded
provenance also requires fresh inspection. Do not disable containment or manually replace
path prefixes. Neither resolving paths nor a PASS report grants source-mapping approval
or enables training. Source review and the two-epoch launch remain separate user actions.

## What the automatic report does and does not verify

`smoke_preparation/validation.json` and `validation.md` separate **automatic PASS/FAIL**,
**human mapping review PENDING**, and **no training executed**. Preparation checks physical
paths against canonical IDs, unique clips, the four-class mapping and source counts,
saved duplicate-audit consistency, at least three readable recognized sources per class,
zero canonical source overlap, all sampled frame decoding and byte-identical leakage
among the selected clips. Very short clips reuse decoded frames for repeated sample indices.

The existing notebook 01 duplicate SHA-256 records are checked for consistency with counts,
current paths and file sizes; unchanged-location files are not all rehashed. Relocation rechecks every moved copy
against saved SHA-256 evidence. If its contents changed,
rerun notebook 01. Only the 12 selected clips are hashed again; their hashes, candidate split
and inspection artifacts are rechecked before training to bind review to unchanged inputs.
Unknown source naming fails automatically rather than assuming independence. The review
sheet helps visual inspection but cannot certify labels or identify all related incidents.
Preparation never sets an approval flag, saves trained weights or creates evaluation metrics.

## Temporal review of the 12 selected smoke clips

Notebook 02 adds a report after the existing automatic preparation; working inspection,
root resolution, source grouping, SHA-256 deduplication and training guards are retained.
Reopen the latest GitHub notebook 02 and Run all with both flags **False**. No rerun of
notebook 01 is needed just to add this review if its artifacts still validate.

The report is saved in Drive under
`AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/temporal_review/`:

- `review.html`: self-contained images, class naming assignments, source IDs, split labels,
  sample frame positions/times, and linked suspicious pairs. Download/open it in a browser
  if Drive preview or Colab output truncates the full report. No external scripts are needed.
- `review.json`: technical checks, sample coverage, SHA-256, uncertain matching evidence,
  limits, and explicit `human_review=PENDING`, `semantic_status=UNVERIFIED`,
  `incident_independence=UNVERIFIED`, `training_executed=false`.
- `review_cases.csv`: per-clip warnings/errors for review.

The current utility also generates:

- `review_summary.md`: concise counts, technical blockers, prioritized suspicious pairs
  (cross-split cases first), and all selected sources with pending label/incident review.
  If there are more than ten pairs, the summary says how many remain in the full report.
- `human_review_queue.csv`: every selected clip remains `PENDING` for both its semantic
  label and incident/source independence, including clips with no automatic alert.
- `selected_clips.csv`: the exact bound selection snapshot, not the complete dataset.
- `export_info.json`: scope, generation time, source-manifest hash and export-file hashes.
- `review_artifacts.zip`: those metadata files plus the HTML/JSON/case report. It contains
  representative frame images inside HTML, but **no raw videos or model checkpoints**.

### Generate and export the real review (no training)

This agent cannot inspect your Drive session when Colab access is blocked. Successful
preparation reported by you is not a substitute for reading the actual review artifacts.
No real suspicious pair names, verified labels or incident decisions are claimed until the
real files are available. The notebook uses its actual validated KaggleHub selection and
video paths; it does not load the synthetic demonstration used in earlier local tests.

1. Reopen the latest GitHub notebook 02, keep/select the T4 runtime, and **Run all** with
   `SOURCE_MAPPING_REVIEWED=False` and `RUN_PROTOTYPE_TRAINING=False`. Mount Drive when prompted.
2. The temporal-review cell displays the concise summary and visual report and saves them
   under this exact Drive folder:

   ```text
   MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/temporal_review/
   ```

3. Download `review_artifacts.zip` from that Drive folder. Alternatively, in the new export
   cell set **only** `DOWNLOAD_REAL_REVIEW=True` and rerun that cell. This export flag is
   unrelated to training approval; leave both training/review flags **False**.
4. Extract the ZIP. Read `review_summary.md`, open `review.html` locally to see the embedded
   temporal frames, and inspect `review_cases.csv` and `human_review_queue.csv`. Compare
   suspicious original videos where frames do not resolve uncertainty. Technical failures
   require correction; uncertain labels and incident relationships require human judgment.

Colab's absolute folder is `/content/drive/MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/temporal_review/`.
Notebook 01 does not need rerunning solely for summary/export support if its saved artifacts
still pass validation. Editing a pending checklist does not grant approval or enable training.

Up to **12 unique positions** are sampled across each clip's reported frame range, using
available videos instead of only the original three-frame sheet. Short clips report the
actual smaller sample count. Unknown-length videos get bounded sampling of at most the
first 240 decoded frames and a partial-coverage warning. This is not full corruption scanning.

Technical checks validate canonical identities, naming-based class mapping, containment,
unique selected clips and zero canonical source overlap. Byte-identical files under different
source IDs are a confirmed content duplicate and make technical review **FAIL**, including
within one split. The report does not call their crime labels or incident identities verified.

Classical dHash (64-bit) plus grayscale mean absolute error compares temporal samples with
one-to-one matches, allowing offsets. At least three matches covering half the smaller sample
set, Hamming distance <=6, grayscale MAE <=12, and texture standard deviation >=12 trigger an
**uncertain shared-scene** alert. These fixed thresholds are uncalibrated heuristics, not
probabilities, semantic models or learned embeddings. Shared cameras/static backgrounds can
cause false positives. Crops, re-encoding, camera differences, other temporal locations and
related incidents outside these 12 clips can evade detection. Low texture, little motion,
too few samples and decoding failures are highlighted. No alerts does not prove independence.

**Before approval:** investigate highlighted pairs and watch the original clips where samples
are inconclusive. Verify source conventions and dataset labels, especially Fighting's small
source pool. Correct grouping/splits if footage is duplicated or related across holdouts;
regenerate preparation and this report after any change. A technical PASS never approves the
mapping. Only an explicit human decision may set the existing review flag. The updated notebook
launch additionally binds the report's split/file hashes and rejects technical failures.

Standalone generation from an already prepared and path-resolved split:

```bash
python -m scripts.review /path/to/candidate_splits.csv /verified/dataset/root /path/to/temporal_review
```

The command exits nonzero on technical failure but still writes the report. It never trains,
loads pretrained weights, calls inference APIs, edits dataset files or changes approval flags.

## Watch/export the 12 ORIGINAL videos (existing candidate only)

For the original-video workflow, **do not Run all or rerun Automatic preparation**.
That earlier workflow prepares candidates; this workflow must reuse your already saved one.
Reopen the latest notebook 02 and run only its setup/Drive-mount cell, then the new
**Play the 12 existing original videos** cell (`original-video-players`). It reads the existing
`smoke_preparation/validation.json` and its unchanged candidate CSV. It never calls the splitter.
If resolved Kaggle paths are no longer accessible, stop and restore/rebind the same canonical
selection; do not create another split just to make playback work.

All 12 originals appear as HTML5 players with class naming assignment, canonical source ID,
split, original filename and byte size. These clips are embedded from the resolved dataset
paths, not inaccessible browser file URLs. The eight user-reported low-motion sources
`burglary041`, `stealing060`, `stealing054`, `stealing002`, `fighting005`, `fighting002`,
`vandalism019`, and `vandalism043` are highlighted when present (`_x264` is ignored for matching).
Any listed source absent from the current candidate is reported; no clip is substituted.
Low motion is a user-reported attention flag, not an inferred crime or label decision.

To create the downloadable original-video package, run **Export originals**
(`export-original-videos`) with `EXPORT_ORIGINAL_VIDEOS=True`. Leave
`SOURCE_MAPPING_REVIEWED=False` and `RUN_PROTOTYPE_TRAINING=False`; both new cells explicitly
keep them False. Set `DOWNLOAD_ORIGINAL_PACKAGE=True` in the same export run if you also want
an immediate browser download. For a later download-only run, set export False and download
True, or download the existing ZIP directly from Drive.

Package location:

```text
MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/original_video_review/original_video_review.zip
```

The archive contains **exactly 12 unchanged original videos**, preserving basenames under
`clips/<split>/<source>/`, plus `selected_clips.csv`, `review_metadata.json`,
`human_checklist.csv`, `README.txt`, and an offline `index.html` with all 12 relative video
players. Extract the entire ZIP before opening it. No re-encoding, cropping or content
changes occur; each streamed original is SHA-256 checked and archive CRC verified.
Class labels, full-video-watched status and incident independence remain **PENDING** in the
checklist. Editing it does not enable training. Existing ZIPs are never overwritten.

Bounds/checks before export:

- Default maximum: 64 MiB per video, 256 MiB total for the 12, plus at most 1 MiB metadata.
- Inline Colab embedding additionally caps the total at 128 MiB to bound notebook output.
  The plan remains available for the ZIP/offline players if inline playback exceeds this.
- The export cell requests Google Drive metadata authorization if needed. Select the
  **same account used for the mounted MyDrive**. Actual account storage is read through
  Drive API `about.storageQuota`; FUSE/local free space is never treated as Drive quota.
- Available Drive bytes must cover twice the estimated archive size plus a 16 MiB margin
  for rename/copy behavior. Output filesystem and local Colab cache space are also checked.
  Unknown/unavailable/non-finite quota or insufficient space blocks export, without fallback.
- Input paths remain inside the verified dataset root; candidate file/signature, video
  sizes/hashes and any existing temporal report binding must match. Content changes during
  export fail and remove the incomplete generated archive.

HTML5 playback depends on the original codec being supported by the browser. If unsupported,
use a compatible offline player on the packaged original; the utility deliberately does not
transcode evidence. These outputs are generated in your authorized Colab/Drive runtime;
local generated-video tests do not establish that your real clips played or were exported.

## Local CPU smoke tests

Python 3.12 was used in Codex. To reproduce this cloud CPU environment:

```bash
python3 -m venv /workspace/crime-venv
/workspace/crime-venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
/workspace/crime-venv/bin/pip install -r requirements-cloud-tested.txt
/workspace/crime-venv/bin/python -m pytest -q
/workspace/crime-venv/bin/python -m compileall -q scripts tests
```

`requirements-cloud-tested.txt` records installed versions for the cloud CPU run. Colab uses
`requirements.txt` ranges to preserve its GPU-compatible PyTorch installation. Export its
actual package versions alongside each real experiment before interpreting results.

CLI examples, after manually reviewing a real inspection manifest:

```bash
python -m scripts.preprocess inspect /path/to/dataset artifacts/inspection
python -m scripts.preprocess split artifacts/inspection/manifest.csv artifacts/splits.csv --per-class 3
python -m scripts.classifier artifacts/splits.csv --output artifacts/classifier
python -m scripts.evaluate cnn artifacts/splits.csv artifacts/classifier/crime_classifier.pth artifacts/test.json
python -m scripts.evaluate gru annotations/synthetic_scenarios.csv artifacts/scenario/scenario_gru.pth artifacts/gru-test.json
```

Do not run expensive full training without confirmation. Tests use generated temporary videos,
not real crime footage. See `reports/validation.md` for actual execution status.
