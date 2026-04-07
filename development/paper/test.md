## 4.6. Training and Inference

**Joint Training Objective.** The BiDecT model is trained by jointly optimizing both decoders via cross-entropy losses. For a given video $X$ with its unified multimodal embeddings $E=\{\overleftarrow{E}, \overrightarrow{E}\}$, let $\overrightarrow{Y}=[\overrightarrow{y_1},\dots,\overrightarrow{y_L}]$ be its ground-truth forward caption, and $\overleftarrow{Y}=[\overleftarrow{y_1},\dots,\overleftarrow{y_{L'}}]$ be its paired pseudo reverse caption (detailed below). The cross-entropy losses for the backward and forward decoders are defined as:

$$ \mathcal{L}_{BD} = -\sum_{t'=1}^{L'} \log P(\overleftarrow{y_{t'}} \mid \overleftarrow{Y_{<t'}}, \overleftarrow{E};\; \theta_{BD}), $$

$$ \mathcal{L}_{FD} = -\sum_{t=1}^{L} \log P(\overrightarrow{y_{t}} \mid \overrightarrow{Y_{<t}}, \overrightarrow{E}, \overleftarrow{H};\; \theta_{BD}, \theta_{FD}), $$

where $\theta_{BD}$ and $\theta_{FD}$ denote the trainable parameters of the backward and forward decoders, respectively. Notably, $\mathcal{L}_{FD}$ explicitly depends on $\theta_{BD}$ because the forward decoder incorporates the global backward context $\overleftarrow{H}$ produced by the backward decoder. This structural coupling ensures that gradients from the forward objective flow back through the backward decoder, encouraging both components to learn complementary representations.

The overall loss $\mathcal{L}$ is formulated as a weighted combination of both objectives:

$$ \mathcal{L} = (1-\lambda)\,\mathcal{L}_{BD} + \lambda\,\mathcal{L}_{FD}, $$

where the hyperparameter $\lambda \in [0,1]$ balances the contribution of the two decoders. During training, we also apply label smoothing to mitigate model overconfidence.

**Pseudo Reverse Captions.** A critical consideration in bidirectional decoding is the prevention of information leakage. If the reverse caption were simply the exact word-for-word reversal of the forward reference caption, the forward decoder could trivially exploit the future answers encoded in $\overleftarrow{H}$, significantly diminishing the model's incentive to learn genuine bidirectional semantic representations.

To address this, the reverse sequences used during training are pseudo reverse captions. Each video in the dataset typically contains multiple reference captions. For a given video, we first reverse all of its associated forward captions to form a localized pool of backward candidates. As illustrated in Figure [$\sout{???}$](), these reversed captions are then randomly shuffled via intra-video randomization and pseudo-paired with the forward captions. Consequently, the backward decoder is never fed the exact reversal of its paired forward caption. This randomization strategy forces the forward decoder to utilize $\overleftarrow{H}$ for comprehending global semantic intent rather than memorizing exact word reversals.

**Inference.** During inference, caption generation proceeds in two sequential stages, both employing beam search decoding for consistency with the training procedure. In the first stage, the backward decoder generates a reverse caption via beam search, terminating upon prediction of the end-of-sequence marker $\langle \text{S} \rangle$. The resultant hidden states are retained to form the global backward context $\overleftarrow{H}$. In the second stage, the forward decoder applies beam search in a standard left-to-right manner, conditioned on both $\overrightarrow{E}$ and $\overleftarrow{H}$, to produce the final video caption.
