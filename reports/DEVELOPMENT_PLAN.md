# 48-hour execution schedule

Hours are relative to the beginning of the student's development session; no fictional elapsed training time is asserted.

| Window | Work | Exit evidence |
|---|---|---|
| 0–4 hours | Verify access/licenses, inspect actual assets, download a manageable subset | Pinned revision, at least 100 aligned decodable videos, schema audit |
| 4–10 hours | Extract shared MediaPipe features, verify disjoint splits, run simple baseline | Audit JSON, baseline eligibility/results, real model forward pass |
| 10–24 hours | Two-epoch verification then expand to 1,000–3,000 samples and train | Real checkpoint, loss curves, held-out generated examples |
| 24–32 hours | Upload inference and webcam recording through same weights | Real predictions on both paths, errors understood |
| 32–36 hours | gTTS, audio download/reset, complete tests | Actual audio playback and browser-device verification |
| 36–44 hours | Evaluation, report, real screenshots, presentation | Actual metrics and prediction table; failure analysis |
| 44–48 hours | Rehearse, archive code/checkpoints and verify clean startup | Reproducible viva demonstration |

Current access block: HF and Kaggle are denied by environment policy. Review/save the dataset network additions in environment settings or run the provided notebook in Colab. Data access must come before any accuracy claim.

If the main dataset remains inaccessible for several hours, inspect ISL-CSLTR via authorized Kaggle access. Only build its closed-set fallback after verifying real video/label files and license. Re-extract raw features, group source/signers and train/evaluate a classifier. Label the fallback explicitly; the present repository does not claim that unavailable fallback data has been loaded.

Defer live streaming until both required recording modes and real evaluation are complete.
