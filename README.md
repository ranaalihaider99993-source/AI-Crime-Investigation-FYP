# AI-Based Crime Investigation and Scenario Generation System Using Machine Learning and Blockchain

Final Year Project using Machine Learning and Blockchain.

This phase provides reproducible **from-scratch PyTorch training pipelines** and Google
Colab notebooks. The original repository description was “AI-Based Crime Investigation
and Scenario Generation System.” Frontend, live camera, object-detector training,
multimodal fusion, blockchain, and investigation reports are future work.
There are no pretrained weights, external AI inference calls, or fabricated accuracy results.

## Resume an existing review in one Colab cell

Open [Notebook 02](https://colab.research.google.com/github/ranaalihaider99993-source/AI-Crime-Investigation-FYP/blob/main/notebooks/02_Crime_Classification_Training.ipynb)
and run its **first code cell: Launch OpenCV-assisted saved review**
(`launch-saved-review-dashboard`). Authorize Drive when prompted, review the
videos, then click **Save review decisions to Google Drive** and wait for the
saved path. CPU is sufficient. Do not Run all or run any other setup/preparation
cell for this existing-review workflow.

The launcher mounts Drive, safely fast-forwards the clean Git checkout to main,
installs dependencies, and loads existing review artifacts. It prefers a folder
with human annotations/saved decisions over a newer blank review and prints its
selection. Set `REVIEW_CONTEXT_FOLDER` only if you need a particular existing
folder. The launcher itself never downloads a dataset, inspects data, prepares/resplits candidates,
or starts training. A dirty/diverged checkout stops without resetting local work.

Previously saved explicit human decisions and notes are restored after identity
checks. New controls remain Pending; metadata and heuristics cannot confirm them.
A live summary lists source IDs by selected-clip label decision (Pending,
Confirmed, Uncertain, Rejected), plus watched/independence counts. Restored
answers are attributed to the prior human submission, not new automatic review.
If saved inputs changed, the launcher stops rather than discarding annotations.

Every save uses a new `decisions-<id>/review_decisions_<UTC>_<id>.csv` plus provenance
inside the printed Drive context-review folder. Original checklists and earlier
versions are preserved. Save before disconnecting or rerunning; unsaved browser
edits are not durable. Both training flags remain False regardless of decisions.

If Kaggle files disappeared in a new session, the launcher first looks in known
Colab/KaggleHub caches. It accepts a relocated package only after saved canonical
identities, physical duplicate counts and SHA-256 content match. It builds a
temporary path-only view; original manifests and review metadata stay untouched.
The active default review is `review-bd132d7b024a`; no new review is created.
If no valid copy exists, run **Restore exact saved-review dataset** immediately
below the launcher **with its defaults**. No path/version editing is normally
needed. It reads bound version evidence; a unique recorded version is requested
exactly. If the version cannot be established, the cell authorizes ONE current
DCSASS release as an untrusted candidate and accepts it only if the COMPLETE saved
canonical inventory, physical copy counts and SHA-256 hashes match. A dataset name,
clip count or "latest" version number alone is never enough.

This may be a large full-dataset download. The cell prints available local disk
and authentication instructions. If Kaggle requires access, sign in/accept dataset
access and run `import kagglehub; kagglehub.login()` in Colab, then retry. Do not
hardcode keys/passwords. Mismatches stop without modifying the original review;
then recover an original full backup or known version. The exported 12 clips are
not a full dataset substitute. Do not rerun inspection, preparation, splitting or
the correction merely to resume. Neither training flag changes.

## Initial dataset/training preparation in Google Colab

### Persistent dataset backup and isolated experiments

After complete version-1 recovery, Notebook 02 offers a separate opt-in persistent
backup cell. It checks actual Google Drive quota, stores the complete physical
dataset layout in a new `canonical-four-class-v2/dataset_backups/<id>/` archive,
and saves SHA-256 and provenance. The saved-review launcher automatically restores
a matching published archive when runtime data is missing, then rechecks every
saved canonical identity, duplicate-copy count and video hash. It never downloads
from Kaggle automatically. Fresh sessions still need Drive authorization and local
space to extract the archive. No original manifest, split or decision is replaced.

The separate experimental CNN entrypoint requires explicit authorization and a T4.
It keeps production flags False, uses random weights for two epochs, filters the
existing selection without replacing sources, and saves checkpoint, configuration,
logs and exploratory metrics under `experimental_cnn_smoke/`. By default it requires
saved Confirmed labels and watched videos. Including Uncertain dataset labels
requires a separate explicit opt-in; such results are UNVALIDATED pipeline evidence.
Rejected, Pending and unwatched examples are always excluded. Incident independence
is never granted by code, and this experiment does not claim validated performance.

The steps below are for initial preparation, not resuming an existing review.

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

### Review selected clips with same-source context (no training)

Open the latest [Notebook 02](https://colab.research.google.com/github/ranaalihaider99993-source/AI-Crime-Investigation-FYP/blob/main/notebooks/02_Crime_Classification_Training.ipynb).
For an existing 12-clip preparation, run **setup/Drive mount (`cell-1`)**, then
**Source-context review players (`source-context-players`)**. Do not Run all or
rerun Automatic preparation. The setup cell updates the code from GitHub main.
The saved resolved dataset files must still be accessible; if paths are missing,
stop and restore that validated dataset rather than selecting a new candidate.
Notebook 01 does not need rerunning for this feature.

Each selected original is shown alongside the nearest available numeric segment
before/after it from the same canonical source. The candidate CSV and all split
assignments remain unchanged; neighbors are context only. Numbered filenames
suggest order, not verified timing or incident continuity. Missing/ambiguous
ordering and gaps are reported. Source names and class mappings are technical
identities, not human approval of labels or incident independence.

A unique folder is printed under
`/content/drive/MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/smoke_preparation/source_context_review/review-<id>/`.
It contains `context_metadata.json`, `human_checklist.csv`, and `README.md`.
Watch all displayed videos, record whether the selected clip itself shows the
crime or only neighboring context does, and compare sources for related incidents.
Record uncertain/absent labels and related source IDs. Every decision starts
PENDING; prior checklists are preserved. The existing original-video ZIP remains
bounded to its original 12 clips. Context videos are played without transcoding,
not copied into that ZIP or added to training. Inline playback has a 128 MiB total
and 64 MiB per-file bound. Browser codec support may vary.
Both `SOURCE_MAPPING_REVIEWED` and `RUN_PROTOTYPE_TRAINING` remain False.

### Consolidate existing human-review notes

In the latest Notebook 02, run **setup/Drive mount (`cell-1`)**, then
**Consolidated review dashboard (`consolidated-review-dashboard`)**. Do not rerun
preparation or context generation just to consolidate an existing checklist.
The cell uses the current session's context folder or prints/selects the newest
saved context metadata folder. Set `CONTEXT_DIRECTORY` to a specific review folder
if your notes are in an older one. Dataset files must still match the saved hashes.

Edit `notes` / `review_notes`, label-review answers, and `possible_related_sources`
in that folder's `human_checklist.csv`; rerun the dashboard to consolidate them.
Use full canonical source IDs for automatic cross-split links from the checklist;
short numeric/free-text references remain for manual investigation. No checklist
answer enables training or automatically approves semantic labels/independence.

A new `dashboard-<id>/` folder beside the checklist contains `review_dashboard.html`,
`review_report.csv`, and `review_report.json`. The HTML displays grouped filenames,
roles/order caveats, source/class/split, warnings, attributed notes, and pair findings.
It imports existing temporal heuristics if available; it does not run semantic
video analysis or infer crime labels. Exact-byte duplicates are technical evidence,
while scene similarity and reported relationships need human investigation.
Missing findings never establish incident independence. Original videos remain in
the existing context playback cell, with no new dataset or candidate modifications.

The user's observation for source **056** is included verbatim and attributed as
unverified. It is associated with a source only when that numeric ID is unique in
the selection; absent/ambiguous IDs stay unresolved in the dashboard. Existing
checklists/notes and prior dashboard snapshots are preserved. Both training flags
remain False. Generating real Drive reports requires running the cell in your
Drive-authorized Colab session; local tests are generated-video fixtures only.

### Review videos and save decisions without editing CSV

The **Consolidated review dashboard** cell now provides an interactive form in
Colab. Open the latest Notebook 02, run setup/Drive mount (`cell-1`), then run
`consolidated-review-dashboard`. Use the existing context folder; do not rerun
preparation. Confirm the printed folder or set `CONTEXT_DIRECTORY` explicitly.

Each source card includes the selected original and available preceding/following
videos, flags, and controls for selected-clip label visibility, watching all shown
videos, incident independence, reviewer name, and notes. All decision controls
start **Pending**. Choose **Confirmed**, **Uncertain**, **Rejected**, or **Pending**
for the stated claim; playback events never automatically mark videos watched.
Fighting and low-motion Vandalism cards receive closer-inspection highlights.
The expandable evidence section retains technical findings and attributed notes.

Click **Save review decisions to Google Drive** below the 12 cards and wait for
the receipt containing the CSV path. Each click writes a fresh
`decisions-<id>/review_decisions.csv` and `provenance.json` inside the chosen context
folder. The original checklist, dataset, candidate and splits stay unchanged.
Save before rerunning the cell or leaving the runtime: unsaved browser form edits
are not durable. Saved decisions do not automatically prefill a new form or change
the training gates. Both training flags remain False even if every decision is
Confirmed. Decisions are attributed human submissions, not automatically verified
labels or proof of independent incidents.

Drive saving uses a callback in the live authorized Colab runtime. Opening the
saved HTML outside that runtime cannot save to Drive. If the callback or codec
fails, the form reports the problem; do not treat an error as a successful save or
an unplayable video as watched. The existing original-video export can be used
with a compatible player. The report contains embedded original video bytes,
subject to existing per-file/total bounds, with no transcoding.

### Automatic OpenCV-assisted review (same single launcher cell)

The first Notebook 02 cell now analyzes all 12 selected clips and available
preceding/following context (at most 36 videos) before opening the dashboard.
No pretrained weights, learned models, external inference APIs or training are used.
It reuses the existing canonical paths, source split and saved human annotations.
Both training approval flags remain unchanged at their disabled defaults.

For each original video, the analyzer takes up to 12 evenly spaced temporal samples
and adjacent-frame pairs at those positions. It reports adjacent-frame mean absolute
pixel difference, changed-pixel fraction (>10 gray levels), median Farneback flow
in pixels/frame at 160×90, longer-gap sampled frame difference, brightness/dark/bright
fractions, grayscale texture and Laplacian sharpness variance. Unknown length gets
an explicitly bounded prefix and WARNING; this is not a full decode or event detector.
Contact sheets retain frame indices and timestamps where reported by the decoder.

Every cross-source clip pair, including context, is checked for SHA-256 equality
and uncertain scene resemblance using the existing one-to-one dHash/grayscale-MAE
matches. The report also includes grayscale-histogram Bhattacharyya distance as a
background appearance cue, not an incident classifier. Same-source neighbors are
not treated as independent sources or added to training. Cross-split findings
identify the roles and source/split IDs for investigation.

Technical status meanings:

- **PASS**: the named technical check passed, or no configured threshold was hit;
  it does not establish a crime label or incident independence.
- **WARNING**: limited sampling, low motion/quality or uncertain shared-scene evidence.
- **FAIL**: integrity/decode failure or exact video bytes shared across distinct sources.

Thresholds are recorded in `analysis.json`: low adjacent change <1 gray level,
sharpness variance <30, texture std <12, dark/bright fractions >0.8, and native
resolution below 160×90. Scene matches use dHash Hamming ≤6 and gray MAE ≤12 for
textured samples, at least 3 one-to-one matches and ≥50% of the smaller sample set.
These are transparent heuristics, not calibrated crime-label/incident accuracy.
Camera motion, exposure, compression, static scenes and common backgrounds can
mislead them. Sparse samples can miss events; no warnings does not prove independence.

Every launcher run creates a fresh `dashboard-<id>/automatic_analysis/` containing
`analysis.json`, `clips.csv`, `pairs.csv`, `priority_queue.csv` and contact-sheet JPEGs.
The consolidated HTML embeds these alongside original players and editable controls.
Cards are ordered by technical severity, cross-split suspicion, warning count and
Fighting scarcity; the score is review priority, not crime confidence. Low-motion
Vandalism is highlighted as well. All automatic semantic/independence fields stay
PENDING. Prior explicitly saved human answers may be restored with attribution,
without granting new approval. New technical outputs never overwrite old reports,
checklists or decisions, and do not invalidate otherwise matching human snapshots.

Real DCSASS analysis is performed only when you run the launcher in your authorized
Colab/Drive session. Local generated-video tests are not real dataset findings.

### Preview and approve saved decisions → human checklist

Run the existing first-cell **Launch OpenCV-assisted saved review** in Notebook 02.
At the dashboard's **Synchronize saved human decisions to checklist** panel:

1. Save any current video-form edits first. Sync reads saved CSVs, not unsaved controls.
2. Click **Preview latest saved decisions**. Review the exact-source field diff,
   before/proposed counts, and remaining Pending/Uncertain sources with reasons.
3. Only if the diff is correct, tick **I reviewed this preview and explicitly
   approve these checklist changes**, then click **Apply approved checklist changes**.
4. Wait for the checklist, backup and refreshed report paths. Rerun the launcher
   before further video-form saves, since old callbacks bind to the prior checklist.

Preview is read-only. Each proposed label/watched/independence value is copied
from an explicit saved human decision, never from motion, filenames, hashes or
similarity. Pending and Uncertain stay as recorded unless the saved human answer
explicitly differs and you approve that exact change. Confirmed is not a training
approval. Neither training flag is changed by synchronization.

The latest decision in the **selected context folder** is chosen by its recorded
timezone-aware save timestamp, not mutable Drive file timestamps. Equal latest
timestamps, missing CSV/provenance, duplicate/missing/nonexact source IDs, changed
split identities, unknown options and changed context/evidence are rejected.
Different review folders are not merged. Check the printed folder or use the
existing explicit folder override if you have multiple saved reviews.

Existing `notes` and other checklist columns are retained. Saved reviewer names
and attributed notes are appended without removing earlier entries; repeated sync
of the same snapshot is a no-op. Prior decision files/provenance and all previous
Drive reports remain untouched. Approved writes create a unique
`sync-<UTC>-<id>/human_checklist.backup.<UTC>.csv` containing the original bytes,
`sync_audit.json`, and `post_sync_report.json`. The refreshed report lists per-source
answers, Pending/Uncertain reasons and Rejected cases; overall training approval
remains PENDING. An approved-sync audit permits resuming older decision snapshots
against the intentionally updated checklist without rewriting their provenance.

If you manually added checklist notes after saving decisions and the launcher
stops on its stale-input guard, run the optional **Preview and approve checklist
synchronization** cell (`preview-approve-checklist-sync`) after launcher setup/Drive
mounting. It allows checklist-only edits into a NEW explicit preview while keeping
candidate/context/evidence checks strict. It preserves those newer notes. Changes
since preview, newer decisions, concurrent syncs or a modified preview require a
fresh preview and approval. If validation/reporting fails after our own write, the
original checklist is restored when its bytes are still ours; the backup remains.
Do not remove a stale sync lock blindly after a runtime crash—inspect backups and
the current checklist first. This workflow requires your authorized live Colab;
local tests do not imply access to your Drive or application to your checklist.

### Correct the vandalism019_x264 label without changing other reviews

Open the latest Notebook 02. Run the first launcher cell for the code update and
Drive mount, then run **Correct vandalism019_x264 only — preview, then explicit
approval** (`correct-vandalism019-review`). Check the printed context folder;
set the launcher's `REVIEW_CONTEXT_FOLDER` if you need another existing review.
Do not rerun preparation, splitting or blanket checklist synchronization for this
single-source correction. If the launcher stops after setup because of a stale
checklist, the dedicated correction cell can still preview checklist-only edits.

Click **Preview source correction**. Verify the exact source, proposed Uncertain
label, unchanged watched value and Pending incident independence, plus the
checklist/decision note changes. Tick **I explicitly approve only this source
correction**, then click **Apply approved source correction**. Wait for the backup,
new decision version and audit paths; rerun the launcher before further form saves.

The supplied user observation is appended, with attribution, preserving old notes:

> A person is visibly writing on a wall, but authorization is unknown. The footage
> does not conclusively establish vandalism, so the label remains uncertain.

Only this exact source's CSV record is patched. All other records and header bytes
remain byte-for-byte unchanged, including quoting and embedded newlines. Existing
watched status is never promoted or reset. Independence must already be Pending;
a conflicting newer state or a non-Confirmed/non-Uncertain label stops for review.
The workflow does not certify whether the writing was authorized.

A `correction-<UTC>-<id>/` folder contains the timestamped original checklist backup,
`correction_audit.json` and `post_correction_report.json`. A new timestamped saved
human-decision CSV patches only the same target record, so the launcher or a future
sync cannot restore its obsolete Confirmed label. Every previous decision version
remains untouched; all non-target decision records are byte-identical. No checklist
schema is regenerated; old checklists without `review_notes` retain their schema
and use their existing `notes` column. Repeating an already applied correction is
a no-op. Stale previews, missing/ambiguous source IDs or changed inputs are rejected.
Neither training flag changes and no training starts. The code prepares a Colab
workflow; it does not mean your actual Drive files were modified from this workspace.

### Saved Kaggle paths in fresh Colab runtimes

The saved-review reader uses `scripts/review_paths.py` to bind missing Kaggle video
paths to a valid local package. Discovery covers the recorded root, standard
KaggleHub caches (including `KAGGLEHUB_CACHE`) and bounded named Colab/project dataset
directories. Use `REVIEW_DATASET_ROOT` for a custom full-package location. Every
relocated canonical clip and duplicate physical copy must match the saved
inspection SHA-256 and copy counts. Selected source assignments, split signature,
context roles, filenames, sizes and hashes must still match the existing review.
Nothing selects new clips or replaces the candidate.

Only temporary local inspection/manifest path views are written. No existing Drive
metadata, manifest, checklist, decision CSV, correction backup or audit is rebased
or rewritten. Original artifact hashes remain the human-decision provenance inputs,
so the approved `vandalism019_x264` Uncertain correction resumes unchanged. Missing
artifact paths are located only at bounded project-relative locations, with their
saved hashes/signatures checked; lost original artifacts must be recovered from
backup rather than regenerated. Subsequent callbacks validate the original inputs
and the temporary view/current selected/context clip hashes again.

Root relocation verifies the whole saved dataset package and can take time; it is
not dataset inspection or model training. The launcher prints verification progress
and accepted root, then opens the existing dashboard. Recovery prefers a uniquely recorded
`mateohervas/dcsass-dataset/versions/N`. The explicitly run recovery cell can instead
verify one current-release candidate against the same COMPLETE saved identities,
copy counts and hashes; no requirement is weakened. If Kaggle requires credentials,
authorize KaggleHub in Colab through its supported login. No authentication or actual
Drive/dataset access is implied by the local regression tests.

### Automatic recovery when the original version number is unknown

Investigation found unversioned `kagglehub.dataset_download('mateohervas/dcsass-dataset')`
configuration. Historical diagnostics/test fixtures mentioning version 1 are not
bound proof of the version used for this saved review. The recovery cell examines
only the actual pinned Drive context, validation/inspection JSON (including nested
root-resolution records) and original manifest paths for version evidence. Numeric
schema/model/OpenCV versions and repository test examples are not dataset evidence.
A single recorded version can be used; conflicting or absent versions remain unknown.

Because the original complete canonical inventory and video SHA-256 evidence are
saved, a candidate CAN identify the original video content without establishing
its historical Kaggle release number. The recovery cell explicitly opts into this
one-candidate download; the Python API remains strict unless `allow_candidate=True`
is supplied. It first requires complete saved per-clip hashes (including valid
legacy duplicate-audit evidence). It does not search/download releases repeatedly,
regenerate manifests, infer labels, or bypass duplicate/source/path checks.

A matching download writes a new `dataset-recovery-<UTC>-<id>/recovery_receipt.json`
inside the existing context folder and opens the existing dashboard. The receipt
separates recorded historical version evidence from the downloaded cache version,
records complete-manifest verification and input fingerprints, and explicitly
keeps historical identity unknown when appropriate. It never edits old records,
review notes, decision files, the corrected vandalism019 label, or training flags.
Source artifacts are checked again across the long download/verification interval.
A mismatch or unavailable original SHA evidence stops safely and requires the
original backup/version rather than treating a new release as the original.

These changes do not mean the real dataset has been restored by the coding agent.
The one-time remaining action is to run the updated recovery cell in the user's
Drive-authorized Colab session, authorize Kaggle if prompted, and wait for the
complete verification and accepted-root receipt. If it rejects the candidate,
recover the original dataset backup or original release number before proceeding.
