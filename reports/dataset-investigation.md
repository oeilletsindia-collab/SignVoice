# Dataset investigation

Observed during development on 30 October 2026. Network failures below occurred through the platform's configured HTTPS proxy; they are not evidence that a dataset is private or unavailable globally.

| Candidate | Verified evidence | Access/compatibility status | Decision |
|---|---|---|---|
| iSign, Exploration-Lab/iSign | ISLTranslate's official README links this host and describes raw-video and MediaPipe pose archives. | HF metadata requests returned proxy `403 Forbidden`. Current asset inventory, schema version, gate, pose format, and license could not be inspected directly. | Preferred source of **raw** ISLTranslate videos, conditional on verified access. Do not claim its current precomputed poses match our extractor. |
| GloSL-ISL, pedroodb/glosl-isl | Dataset URL supplied by user. | HF metadata blocked by proxy 403. License, sentence annotations, pose layout and videos remain unverified. | No invented statistics or format assumptions; inspect in Colab before considering an adapter. |
| ISLTranslate, exploration-lab/isltranslate | Official GitHub README and `data/ISLTranslate.csv` downloaded successfully. CSV has columns `uid,text`, 31,222 rows, all nonempty UID/text. Source group/segment UID examples were inspected. README describes continuous sentence/phrase videos, MediaPipe archives, and CC BY-NC 4.0 with citation. | Annotation access verified. Video download points to blocked HF iSign host. README license paragraph refers to CISLR by name, so confirm current dataset terms on the hosting card before redistribution. No raw videos or poses loaded here. | Provisional target: extract raw videos using our own shared pipeline. |
| ISL-CSLTR, drblack00/isl-csltr-indian-sign-language-dataset | Kaggle URL supplied by user. Public API search attempted. | Kaggle request returned proxy 403. Exact contents, splits and license remain unverified. | Fallback cannot be declared working until actually downloaded and audited. No automatic conversion of sentences into gloss labels. |

Reference implementation ISLiM was cloned and inspected. It documents a 94-landmark MediaPipe representation and has a raw-video extraction script, but the checked checkout contained **zero `.npy` files** despite its README mentioning samples. It is not evidence of available aligned training samples. Its published architecture/results are external claims, not SignVoice results.

## Immutable source evidence

- ISLTranslate: `https://github.com/exploration-lab/isltranslate`, commit `1b599eb2f9c552d165958a95827507169c008016`.
- ISLiM: `https://github.com/ahmedshoaib/ISLiM`, commit `ec50943b928c9db32b0a9bd30d74b7eb43d81a5d`.
- SignVoice's landmark subset follows the documented MediaPipe anatomical indices; no ISLiM learned weights or claimed metrics are used. SignVoice's normalization, visibility handling and sampling are its own versioned protocol; even ISLiM arrays cannot simply be imported as normalized SignVoice input.

## Required next validation

1. Run `scripts/inspect_datasets.py` after HF access is enabled, or in Colab.
2. Inspect actual current file inventory, README, license and gate requirements.
3. Download a manageable set of raw videos and corresponding English annotations at a pinned revision.
4. Verify exact UID/video matching, decoding and landmark quality on at least 100 examples.
5. Record prepared manifest counts, rejection reasons, source grouping, preprocessing metadata and archive revision.
6. Run training and held-out raw-video inference before claiming video translation works.

**Current real aligned video/pose examples loaded: 0.** Annotation-only rows do not satisfy that milestone.
