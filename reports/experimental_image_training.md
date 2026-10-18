# Actual image CNN experiment — UNVALIDATED

Trained from random initialization on **Tesla T4**, 2026-10-18. No pretrained
weights or external AI inference APIs. Dataset backup restored from existing
Drive archive and all 16,639 canonical clips / 33,278 physical copies reverified
against saved SHA-256 evidence in this runtime before selection.

## Data, review and leakage audit

Used the existing selected source/split assignments, filtered without reassignment:
2 Confirmed + 7 Uncertain watched sources with original weak video categories.
Three Rejected sources excluded. All incident-independence answers remain Pending.
These categories are **not verified image observations**. No human decisions,
canonical manifests/splits, prior checkpoints or production flags were changed.

Uniformly sampled up to 8 frames per selected clip: 72 candidates. Global exact
preprocessed-pixel SHA-256 and perceptual duplicate filtering removed 58,
including similar frames within sources. Retained **14 frames / 9 sources**:

| Split | Frames | Sources | Class support |
|---|---:|---:|---|
| Train | 5 | 4 | Burglary 2, Stealing 1, Fighting 1, Vandalism 1 |
| Validation | 3 | 3 | Burglary 1, Stealing 1, Vandalism 1 |
| Test | 6 | 2 | Burglary 5, Fighting 1 |

No source ID crosses splits. Near duplicates are dropped when dHash Hamming
distance ≤4 and 8×8 RGB mean absolute difference <6; exact pixel duplicates are
always dropped. Remaining frames pass this algorithm, but the heuristic cannot
prove incident independence or eliminate all visual similarity. Frames from a
source remain correlated. This is a tiny technical experiment, not a benchmark.

## Model and actual metrics

`WeakFrameCNN-v1`: Conv 3→16→32→64, ReLU/pooling, adaptive average pooling,
Linear 64→4. Class order: Burglary, Stealing, Fighting, Vandalism.
EXIF transpose → RGB → bilinear aspect-preserving black pad 128×128 → CHW
float32 /255. Adam .001, batch 8, seed 42; **15 epochs**, best validation-loss
checkpoint selected at **epoch 5**. Test evaluated after selection.

- Test frame loss: **1.2588208914**.
- Frame accuracy: **5/6 = 83.33%**, four-class macro F1 **0.22727**.
- Source accuracy (mean frame logits): **1/2 = 50%**, macro F1 **0.16667**.
- Every test prediction was **Burglary**. Both accuracies equal the majority
  baseline. No useful generalization or crime accuracy is established.
- Stealing/Vandalism absent from test; Fighting absent from validation.
- Full confusion matrices, class supports and predictions: companion JSON.

## Saved and integrated artifacts

Drive: `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/experimental_image_cnn/20261018T094503Z-bb078083/`.
Contains frame PNGs, frame/source provenance and removed-frame audit, human-label
status snapshots, config, history, execution log, metrics, completion record and
`experimental_image_cnn.pth`. Saved checkpoint and records were read back and
SHA-256 verified during local transfer. Checkpoint SHA-256:
`e95b7ebf04510e78bbbde4e7716408cbdd1666291a7406ed42dcf26878e647a8`.

The local checkpoint is ignored by Git; source code pins its hash. See
`models/README.md` to install it in another checkout. Existing video weights are
separate and unchanged. `/api/images/model` exposes availability/configuration;
`/api/images/analyze` runs classification per image (1–6) after bounded decoding.
The UI shows the top weak category, collapsible four-class scores, an uncalibrated
inconclusive heuristic and repeated UNVALIDATED warnings. Scores are not crime
probabilities. Shared scenarios continue to abstain from unsupported facts.

The reviewed attribute trainer and its gates are unchanged. Notebook 05 now has a
separate disabled-by-default weak-label launcher; no flag is silently enabled.

## Verification and next step

Local API/model/decoder/training regression checks and real browser image/video
tests are recorded with this commit. Desktop/mobile screenshots accompany this
report. Synthetic fixtures test plumbing only; the metrics above are the actual
saved T4 run, not test-fixture training results.

Final checks: **43 Python tests passed** and **7 JavaScript score-policy tests
passed**. The real source-video-derived image produced a 29.9% top relative
Burglary score, displayed as **Inconclusive / Low confidence**. Single and two-image
browser submissions succeeded; six-image inference passed through the API.
Video inference still returned three sampled frames and its existing inconclusive
result. At mobile 390×844, no horizontal overflow and the actual footer was
visible at the bottom. Screenshots: `trained-image-desktop.png` and
`trained-image-mobile.png`. The T4 runtime was released after persistence and
artifact readback; the local Flask server remains available on port 8000.

Next: expand genuinely reviewed image/action evidence with independent sources,
all-class held-out coverage and normal/out-of-domain examples. Resolve incident
independence and calibrate abstention on validation data. More epochs on these
14 weak frames would not establish useful scene understanding.
