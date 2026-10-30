# Presentation outline — 12 slides

1. **Title and author:** SignVoice; MCA project; institute/student details to be filled by the author.
2. **Problem:** connected ISL to English; accessibility motivation; no claim that ISL is English gesture encoding.
3. **Scope:** upload and webcam recording; text and speech; completed-recording processing.
4. **Related work:** ISLTranslate and landmark-based translation; distinguish external results from ours.
5. **Dataset audit:** actual source, license, revision, usable counts and rejection reasons. Currently pending video access.
6. **Preprocessing:** 94 points, both hands, body and face; sampling, missing values, shoulder normalization; shared train/inference schema.
7. **Architecture:** BiLSTM, masked attention, GRU decoder; teacher forcing versus autoregression.
8. **Training:** Colab GPU, split policy, optimizer, clipping, scheduling, early stopping and resumable checkpoints.
9. **Baseline:** most-frequent sentence; eligible closed-set BiLSTM classifier with separate split/coverage.
10. **Results:** insert actual curves, BLEU/WER/exact match and reference/prediction table. Leave clearly “not yet measured” until trained.
11. **Live demonstration:** actual upload and webcam recording; use current-state screenshots if no model, with an explicit limitation label.
12. **Limitations and future scope:** signer variation, vocabulary/domain limits, detector errors, hallucinations and future streaming evaluation.

Speaker notes: Explain what each metric measures and include one real failure case. A working UI and synthetic smoke test are engineering evidence, not evidence of sign-language accuracy.
