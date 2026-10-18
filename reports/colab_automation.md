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

Initial full regression run: 144 tests passed on Colab CPU. No validated crime
classification accuracy is established by dataset verification or these tests.
