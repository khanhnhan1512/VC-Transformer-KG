## 4.6. Optimization

The BiDecT model is trained by jointly optimizing both decoders via cross-entropy losses. For a given video $X$ with its unified multimodal embeddings $E=\{\overleftarrow{E}, \overrightarrow{E}\}$, let $\overrightarrow{Y}=[\overrightarrow{y_1},\dots,\overrightarrow{y_L}]$ be its ground-truth forward caption, and $\overleftarrow{Y}=[\overleftarrow{y_1},\dots,\overleftarrow{y_{L'}}]$ be its paired pseudo reverse caption (detailed below). The cross-entropy losses for the backward and forward decoders are defined as:

$$ \mathcal{L}_{BD} = -\sum_{t'=1}^{L'} \log P(\overleftarrow{y_{t'}} \mid \overleftarrow{Y_{<t'}}, \overleftarrow{E};\; \theta_{BD}), $$

$$ \mathcal{L}_{FD} = -\sum_{t=1}^{L} \log P(\overrightarrow{y_{t}} \mid \overrightarrow{Y_{<t}}, \overrightarrow{E}, \overleftarrow{H};\; \theta_{BD}, \theta_{FD}), $$

where $\theta_{BD}$ and $\theta_{FD}$ denote the trainable parameters of the backward and forward decoders, respectively. Notably, $\mathcal{L}_{FD}$ explicitly depends on $\theta_{BD}$ because the FD incorporates the global backward context $\overleftarrow{H}$ produced by the BD. Consequently, gradients from $\mathcal{L}_{FD}$ flow back through the BD, ensuring that the backward context $\overleftarrow{H}$ is jointly optimized for both reverse and forward caption generation.

The overall loss $\mathcal{L}$ is formulated as a weighted combination of both objectives:

$$ \mathcal{L} = (1-\lambda)\,\mathcal{L}_{BD} + \lambda\,\mathcal{L}_{FD}, $$

where the hyperparameter $\lambda \in [0,1]$ balances the contribution of the two decoders. During training, we also apply label smoothing to mitigate model overconfidence.

**Pseudo Reverse Captions.** A critical challenge in bidirectional decoding is the prevention of information leakage. If the reverse caption used to supervise the BD were simply the exact word-for-word reversal of the forward reference caption, the FD could trivially exploit the future answers implicitly encoded in $\overleftarrow{H}$. In such a scenario, the FD would degenerate into a simple copying mechanism, significantly limiting the model’s ability to learn meaningful multimodal alignments and robust bidirectional contexts.

To address this, we adopt a randomization strategy inspired by prior works in bidirectional decoding [$\sout{CITE}$-BiTransformer | BTKG](). Specifically, we leverage the fact that each video in the dataset typically contains multiple reference captions. For a given video, we first apply a word-for-word reversal to all of its associated forward captions to form a video-specific backward pool. As illustrated in Figure [$\sout{???}$](), these candidates are then randomly shuffled and pseudo-paired with the forward captions. This randomization strategy forces the FD to utilize $\overleftarrow{H}$ for building a global contextual understanding rather than exploiting trivial word-level mappings.

![](figures/Pseudo-Reverse-Captions.svg)<br>
Figure 4. The construction process of pseudo reverse captions. For a given video, the reference forward captions undergo a word-for-word reversal to form a video-specific backward pool. These candidates are then randomly shuffled and pseudo-paired with the original forward captions. This intra-video randomization prevents information leakage by forcing the FD to leverage the global backward context rather than exploiting trivial word-level mappings.
<br><br>
