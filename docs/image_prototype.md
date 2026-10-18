# Image evidence prototype

The image workflow is separate from video inference. No trained semantic image
checkpoint or verified image-level annotations were found in this repository.
The saved four-class video CNN is not an image evidence model. Synthetic scenario
examples are not evidence of real scenes.

## Working now

Single/multiple JPEG, PNG and WebP uploads; local previews; bounded decoding;
EXIF orientation correction; dimensions, byte count and SHA-256; deterministic
evidence briefs separating measured observations, hypotheses and unknowns.
Uploads remain in memory and are not saved. No review files are read or written.
No objects, actions or crimes are recognized. No semantic confidence is supplied.
The brief generator is a safe template foundation, not a trained language model.

Limits: six images, eight MB and sixteen million pixels per image; no animation.
Preprocessing contract: EXIF transpose, RGB, bilinear aspect-preserving resize
and black pad to 128×128, future CHW float32 /255. Server-generated thumbnails
are returned for inspection; browser previews show the selected original.

## Separate from-scratch training design (not executed)

Use a new, explicitly reviewed image evidence dataset, not canonical video crime
labels. Start with observable attributes such as person, vehicle, visible smoke
and visible flame. Use multilabel targets with unknown labels masked out, not
assumed negative. Every image needs original-file SHA-256, licensed provenance,
source/incident group, reviewer, review status and attribute annotations. Images
from one incident/source must remain in one split. Deduplicate before freezing
new image-specific splits; preserve existing canonical video manifests/splits.
Human reviewers must define each attribute and resolve ambiguous evidence.

Train a small CNN initialized randomly (three convolution/ReLU/pooling blocks,
global average pooling and an attribute head), using masked BCE loss. No external
weights, AI APIs or synthetic crime narratives. Match the preprocessing contract
above. Save architecture, ordered attributes, seed, manifests' hashes, split
groups, preprocessing and optimizer configuration alongside each checkpoint.
Do not reuse the video checkpoint or production training flags.

Colab can run this independent training pipeline after data review. First run
CPU manifest/decoder checks; request a GPU only for the actual approved training
experiment. Evaluate on incident-disjoint held-out data, report per-attribute
precision/recall and calibration with uncertainty. Select abstention thresholds
on validation data only. Include out-of-domain and normal images. Do not claim
crime detection accuracy from attribute metrics.

A future scenario generator may cite only reviewed human observations and
evaluated attribute outputs, each linked to an image ID and source. Keep model
suggestions distinct from observed facts. Missing evidence must remain unknown;
never infer suspects, intent, motive, chronology or crime occurrence. Production
integration remains blocked until suitable data, model and evaluation exist.

## Run

`python -m pip install -r requirements-web.txt`

`python -m web.app` → http://127.0.0.1:8000

Next step: curate and human-review the separate image attribute dataset. This
prototype deliberately does not provide a training launcher for unreviewed data.
