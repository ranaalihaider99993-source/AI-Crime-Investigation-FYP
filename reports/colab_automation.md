# Authorized Colab automation evidence

Recovered `mateohervas/dcsass-dataset/versions/1` through the authenticated
Playwright-controlled Colab session. The existing verifier accepted all 16,639
canonical clips, 33,278 physical video copies and saved SHA-256 evidence. It
resumed `review-bd132d7b024a` without changing decisions or regenerating splits.

Persistent backup under `MyDrive/AI-Crime-Investigation-FYP/canonical-four-class-v2/`:

- `dataset_backups/20261018T052115Z-9ac4d688/dcsass-v1.tar`
- Archive bytes: 1,595,750,400; original file bytes: 1,536,142,028.
- Files: 33,304, including 33,278 videos and 26 other original files.
- SHA-256: `edcc13b843139af0cadd110c89df8fe649a95ba09587b4cfb2851073565fa775`.
- New checksum and `provenance.json` accompany the archive; no old backup replaced.
- Actual Drive API quota before export: 429,150,968,054 bytes available.
- Full source verification repeated before and after archive publication; copied
  archive SHA-256 reread from Drive and matched.

The resumed explicit human decisions are 2 Confirmed, 7 Uncertain, 3 Rejected;
all 12 groups watched, all incident-independence answers Pending. The prior
`vandalism019_x264` correction remains Uncertain. The user separately authorized
an UNVALIDATED technical experiment with the 2 Confirmed plus 7 Uncertain clips,
excluding all 3 Rejected clips. That authorization does not alter human decisions,
certify labels or incident independence, or enable either production flag.

Final full regression run: **145 tests passed in 75.70 seconds** on Colab CPU.
The real Drive archive was extracted into a fresh runtime directory and all
canonical clips, physical copy counts and SHA-256 values were reverified:
`REAL_ARCHIVE_RESTORE_VERIFIED /tmp/crimescope-dcsass-v1-m922ap1l`.
No validated crime classification accuracy is established by dataset verification
or these tests.

On the replacement **Tesla T4** runtime, the first launcher automatically restored
the Drive archive into `/tmp/crimescope-dcsass-v1-qbiger9f`, fully reverified the
dataset and resumed dashboard `dashboard-ea142bd9fbc9`. No second Kaggle download
was needed.

## First isolated CNN experiment — UNVALIDATED

Completed two epochs from random initialization on Tesla T4 using code commit
`d51cd1124b1a4e7e7a8f72a17a69a1a08acf96a7`. The existing selection was filtered
without reassignment: 4 train, 3 validation, 2 test clips. All three Rejected
examples were excluded; the user explicitly authorized use of the seven Uncertain
dataset labels for this technical execution experiment. All incident independence
decisions remain Pending. These are not validated classification results.

Drive run folder, relative to `canonical-four-class-v2/`:
`experimental_cnn_smoke/20261018T052957Z-84209c36/`.

- Test loss: 1.399735927581787; accuracy: 0.5; four-class macro F1: 0.16666666666666666.
- Test has only Burglary and Fighting, one clip each; Stealing and Vandalism have
  no test support. Both predictions were Burglary. No generalization claim follows.
- Best validation-loss checkpoint: epoch 1; two epochs executed.
- `experimental_cnn.pth` (108,171 bytes), SHA-256:
  `a31794fd43f0a736d63702806a0c838af266b3536867a65da4cd74b4082b5ab6`.
- Configuration, exact experimental subset, provenance, execution log, epoch
  history, full per-class metrics, completion record and artifact verification
  checksums are saved alongside the checkpoint.
- Checkpoint reloaded from Drive with finite weights; saved metrics reread and
  matched; canonical split signature and saved human-decision hash unchanged.
- Both production flags remain False. No pretrained model or external AI API used.
