# Experimental inference UI update

The top experimental result is prominent by default. All four raw scores,
including the other three categories, are available inside the initially closed
**View all model scores** disclosure. Each new result closes it again.

Display-only rule: **Inconclusive / Low confidence** if the top relative score is
below 0.50, **or** the top-minus-runner-up margin is below 0.10. Equality passes
each boundary. Both must pass before the category becomes the main heading.
These conservative heuristic thresholds are experimental, uncalibrated and not
validated. They are not probabilities of crime or evidence of a confirmed incident.
The highest-scoring category remains secondary information for inconclusive results.

## Local verification

- Seven Node policy tests passed: real scores, strong separated scores, close
  scores, weak scores, threshold boundaries, ties, and malformed input.
- All 13 existing Windows inference/model tests passed unchanged (47.98 seconds).
- Real sample upload through the local browser retained its original raw output:
  Burglary score 0.264125794172287; runner-up Stealing 0.2547009289264679.
  The UI showed **Inconclusive / Low confidence** for both weak and close scores.
- Verified disclosure closed by default, expanded to four score rows, and reset
  closed after another inference.
- A **browser-only test fixture** with scores 0.70, 0.15, 0.10, 0.05 verified the
  top-category display. This was not actual CNN output or performance evidence.
  The fetch override was removed and actual inference rerun afterward.
- Desktop and 390-pixel mobile layouts showed no horizontal overflow.
- Existing warnings remain visible. No CNN, checkpoint, preprocessing, API,
  production flag, training pipeline, human review decision or existing test changed.

Evidence:

- [Actual sample, collapsed inconclusive result](inference_ui_inconclusive.png)
- [Actual sample, mobile layout](inference_ui_mobile.png)

Launch with `.\.venv\Scripts\python.exe -m web.app` and open
http://127.0.0.1:8000. The owned local QA server is stopped after verification.
