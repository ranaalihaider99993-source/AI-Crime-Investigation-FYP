# Colab execution access check

- GitHub main initially pointed to c34bb74; cdb9147 was local only.
- Pushed cdb9147 to the existing repository's main branch without force.
- Verified main matched the full cdb9147 SHA and all four notebook raw URLs returned HTTP 200 with valid JSON.
- Playwright and /usr/bin/chromium are installed. An actual headless browser navigation to the dataset inspection Colab URL failed with `net::ERR_TUNNEL_CONNECTION_FAILED`.
- No interactive Colab runtime, Google account session, GPU allocation, or Drive authorization was established.
- No DCSASS download, real-data training, held-out real-data metrics, or real trained checkpoint was produced here. Synthetic CPU smoke results from the earlier task remain clearly labeled.

Historical follow-up fixes before schema v2: notebook 02 defaulted to a 15-source-clip / two-epoch real-data smoke
experiment (three independent original source videos per class), prints actual training
device, evaluates held-out sources, and offers an explicit checkpoint download. Notebook 04
no longer requires a GRU checkpoint for CNN evaluation. Six local tests passed after these
changes; notebook validation does not establish that Colab execution works.

Launch notebook 01 in an authorized Colab browser session, mount Drive, and inspect actual
data before reviewing source IDs and enabling notebook 02's two training/review flags.
If no GPU is allocated, stop before training when GPU execution is required. The first
smoke result has only one held-out source per class; it tests the pipeline, not generalization.
Full-data training must wait for a successful real-data smoke experiment and user approval.

Schema v2 supersedes that default: four classes and 12 clips, still with launch/review flags disabled. See validation.md for current regression results.

A subsequent browser access attempt for notebook 02 again failed with
`net::ERR_TUNNEL_CONNECTION_FAILED`. Browser automation is installed, but no authenticated
Colab session, Drive access, GPU allocation, or real-data execution was available to the
agent. The user reports running inspection and default notebook preparation on a T4 runtime;
those runs were not performed or observed by the agent. Notebook 02 now automatically
prepares a candidate/report without approval and offers a separately guarded two-epoch launch.
