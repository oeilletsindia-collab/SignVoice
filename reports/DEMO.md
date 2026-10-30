# Five-minute demonstration script

## Current implementation demonstration

Say: “This build demonstrates the training system and application. Real-data model training is pending dataset access; I am not claiming translation accuracy.”

1. Show the four UI tabs and missing-checkpoint information.
2. Show the versioned preprocessing configuration and shared `Translator.translate` function.
3. Run `python -m pytest -q` and explain which tests use synthetic fixtures or mocked speech.
4. Show `reports/smoke.json`: a real optimization/checkpoint test on synthetic input, not a translation result.
5. Open the Colab notebook and identify the actual dataset selection/training cells and current access blocker.

## Final demonstration after real training

Prepare beforehand: locally installed dependencies and ffmpeg; `models/best.pt`; matching evaluation JSON; known held-out signing video; camera permission; audio playback; Internet for gTTS. Verify the whole path once before your viva. Keep an honest offline plan: if speech fails, show recognized text and the actual speech-service error.

1. (0:00–0:40) Explain continuous connected signing and the distinction from alphabet classification.
2. (0:40–1:40) Upload a held-out video, preview, translate and compare the **actual** output with its annotation. Play and download speech if successful.
3. (1:40–2:40) Record a short signing sequence, stop, translate, play speech, and reset. Do not claim browser recording is live translation.
4. (2:40–3:40) Show real Colab loss curves and test BLEU/WER/exact match, number of evaluated samples and dataset revision.
5. (3:40–4:30) Show a correct example and a failure case. Discuss signer/domain shift and missing detections.
6. (4:30–5:00) Explain architecture and next improvements. State any unverified capabilities clearly.

Never rename videos to influence output, substitute fixed predictions, or use training examples as held-out test evidence. Capture screenshots only from this real run.
