# Viva questions and defensible answers

1. **What is continuous sign language translation?** It maps connected signing over time into a spoken-language sentence, accounting for temporal context and different grammar. It is more than alphabet or isolated-sign classification.
2. **What has actually been completed?** Code, notebook, temporal model, preprocessing, training/evaluation mechanics, UI and engineering tests. Real-data training and video translation remain pending. Update this answer only after obtaining actual evidence.
3. **Why use a BiLSTM?** It encodes past and future visual context in a completed recording. It is lightweight enough for an initial Colab experiment. This encoder is not causal and therefore is not itself a live streaming model.
4. **Why attention?** English output tokens need different portions of the visual sequence. Masked attention lets the decoder weight valid visual frames without attending to padding.
5. **Why not CTC on English labels?** CTC assumes a monotonic alignment of output labels with input time. English word order differs from signing and English words are not annotated glosses. Direct sentence annotations support sequence-to-sequence training.
6. **Are the outputs unrestricted English?** No. The word vocabulary is fitted on training text. The model can generate new combinations of known tokens; unseen words become UNK.
7. **How are train/test leaks prevented?** Source-video segments and identical normalized text are grouped before the main split; content hashes, UID, group and text are checked across partitions. This does not prove signer independence.
8. **Why not trust a pose archive with 376 columns?** Column count cannot identify landmark order, units, missing-value handling, sampling or normalization. SignVoice re-extracts raw videos through its own versioned pipeline.
9. **How are missing hands handled?** Bounded internal gaps are interpolated. Other missing points remain zero with zero visibility. Videos with no detected hands are rejected; no invented trajectories are used to replace completely missing hands.
10. **What is teacher forcing?** The decoder receives the reference previous token while training. Evaluation receives its own predicted token, which exposes real autoregressive errors.
11. **Why ignore padding in the loss?** Padding is a batching artifact, not a target word. Encoder packing and attention masks also prevent temporal padding from influencing predictions.
12. **What is BLEU?** Corpus n-gram overlap with brevity penalty. The implementation records its sacreBLEU signature. BLEU alone does not establish grammatical correctness or semantic fidelity.
13. **What is WER?** Word substitutions, deletions and insertions divided by reference word count. It can exceed 100% when insertions are large.
14. **What is the classifier baseline?** A BiLSTM mapping a sequence to one whole-sentence class, only if repeated classes exist. It has a separate source-group split and explicit test coverage. It is not open sentence translation.
15. **Why can a model output a fluent but wrong sentence?** Language priors and scarce data can dominate weak visual evidence. Reference/prediction examples and failure cases are essential; no fake confidence score is used.
16. **Does webcam recognition stream live?** No. Gradio records a video, then the shared pipeline translates it after recording stops. Real-time sliding windows are future work.
17. **How is resumable training supported?** Last and best weights, optimizer, scheduler, vocabulary, preprocessing, RNG state and dataset content digest are saved. Resume checks the dataset and architecture match.
18. **What requires Internet?** Initial installation, dataset download and gTTS speech. Recognition can run locally after weights and dependencies are installed; gTTS failure preserves text.
19. **What are your results?** No real-data translation metrics have been measured yet. Engineering pass counts are recorded separately. After Colab training, answer using the actual saved evaluation file, not paper results.
20. **How would you improve it?** Increase verified aligned data, use signer-disjoint tests, inspect landmark quality, compare pretrained decoders and evaluate semantics with qualified ISL users before adding complexity.
