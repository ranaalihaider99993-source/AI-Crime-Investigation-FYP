# Combined incident prototype

Home page: one multiple-file upload and Analyze Incident button. Limits: 6 images
(8 MB and 16 million pixels each), 3 videos, 9 files, 60 MB total. Processing is
sequential on local CPU; one incident request at a time. Uploads are temporary
and deleted after the request. A corrupt file rejects the entire incident result.
Missing image weights reject image-containing incidents rather than silently
omitting images. Videos retain the original frame sampler and pinned checkpoint.
Individual tools and APIs remain available at `/individual` and their old URLs.

## Fixed aggregation rule

1. Hash original file bytes. Exact duplicates within one modality count once and
   are not run twice. Re-encoded/edited duplicates are not automatically detected.
2. Use each checkpoint's four softmax scores in canonical order: Burglary,
   Stealing, Fighting, Vandalism. Video scores retain mean-frame-logit aggregation.
3. Arithmetic mean across unique files within each modality.
4. If both modalities exist, mean the two modality vectors with equal weights.
   Otherwise use the single modality mean. This prevents image file count alone
   from overpowering videos; it does not establish independence or reliability.
5. Rank all four classes, break ties in checkpoint order, display the top three.
   Scores are not renormalized over the three displayed alternatives.

No ensemble fitting, retraining or calibration. Models have correlated biases
and different training/preprocessing; equal weights are an experimental display
heuristic, not statistically justified evidence fusion. Source/incident
independence remains unresolved, and the image model collapsed on its tiny test.
The same inconclusive heuristic is retained: top score <50% or top-two margin
<10 percentage points. Passing it never confirms a crime.

## Meaning of scenarios

The three "scenarios" are explicitly unverified category hypotheses. Their
conditional explanations are separate from the CNN scoring basis; they never
assert objects, acts, chronology, suspects or motives as observations. The renderer abstains from
event reconstruction. User-supplied grouping does not certify a common incident.
Neither model has Normal/Unknown, so even non-crime evidence gets forced scores.
Percentages are uncalibrated relative model scores, not crime probabilities.

The current cards expand those hypotheses into category-conditioned, human-readable
templates: possible property entry, unauthorized property removal, physical
confrontation or property damage. Each explanation begins with a conditional
human-verification requirement, not an asserted event. Each card separates its
model basis, hypothetical explanation and specific uncertainty. `observed_events`
is always empty; these templates never extract observed actions, people or time
sequences. This is deterministic text generation, not a trained scenario model.
Scores, ranking, checkpoints and aggregation are unchanged. Repetitive caveats,
modality contributions and technical records remain in collapsed Advanced Details.

Hashes, model architecture, preprocessing, file-level predictions, full four-class
scores and aggregation details are collapsed in Advanced Details. No review,
training flag, model checkpoint or canonical dataset split is modified.
