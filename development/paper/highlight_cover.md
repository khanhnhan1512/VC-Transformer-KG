# Highlights

**Highlights:**
- BiDecT eliminates the intermediate encoder, projecting multimodal features directly into a bidirectional decoder for video captioning.
- A GOP-based pipeline extracts complementary appearance, semantic, and motion features from structured video segments.
- A dual-purpose scheme leverages a pre-trained vision-language model for both appearance and semantic features, replacing explicit knowledge graphs.
- A pseudo reverse caption strategy prevents information leakage between backward and forward decoders during bidirectional training.
- BiDecT achieves state-of-the-art CIDEr scores of 138.0 on MSVD, 64.8 on MSR-VTT, and 78.2 on VATEX, surpassing prior methods by up to 10.8 points.

# Cover Letter

Dear editors and reviewers,

We are pleased to submit our manuscript, "BiDecT: Encoder-free Bidirectional Decoder Transformer for Video Captioning", for consideration as a research article in [Journal/Conference Name]. This work is original, has not been published previously, and is not currently under consideration for publication elsewhere.

This manuscript addresses two common limitations in Transformer-based video captioning: the reliance on intermediate encoder stages that refine visual features before decoding, and the restriction of caption generation to strictly left-to-right decoding. We propose BiDecT, an encoder-free framework that projects multimodal features directly into a bidirectional decoding system. By removing the intermediate encoder, BiDecT reduces the dominant computational cost from quadratic to linear in the number of input video segments. A backward decoder generates right-to-left context, which the forward decoder incorporates alongside the multimodal representation to produce the final caption. A pseudo reverse caption strategy prevents information leakage between the two decoders during training without requiring additional annotations.

BiDecT represents each video as a sequence of Groups of Pictures (GOPs), from which three complementary feature types are extracted: appearance features from a pre-trained vision-language model, semantic features derived from the descriptions it generates, and motion features from a video classification model. This design transfers rich visual and linguistic knowledge from pre-trained models to video captioning, providing semantic grounding without explicit knowledge graph construction. Experiments on MSVD, MSR-VTT, and VATEX demonstrate that BiDecT achieves state-of-the-art CIDEr scores of 138.0, 64.8, and 78.2, outperforming prior methods by up to 10.8 points. Ablation studies confirm the contribution of each design choice, including bidirectional decoding, the pseudo reverse caption strategy, and modality-specific type embeddings.

We appreciate your time and consideration of our manuscript, and we look forward to your feedback.

Sincerely, 

Thanh Le
