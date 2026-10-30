# Model artifacts

- `smoke.pt`: locally generated synthetic engineering checkpoint; rejected by application inference.
- `best.pt`: place your actual real-data trained best checkpoint here after Colab training.
- `last.pt`: resumable state, kept with best.pt during training.

Binary weights are ignored by Git. Checkpoints bundle model config, vocabulary, tokenization, preprocessing metadata and training status. Only load weights from trusted sources. Evaluation additionally records the checkpoint SHA-256 to prevent displaying mismatched metrics.
