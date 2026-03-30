## 4.3. Multimodal Feature Embedding

The feature extraction process yields three distinct types of features: appearance ($F_A$), semantic ($F_S$), and motion ($F_M$). Because these features are extracted from different pre-trained models, they do not share the same dimensional space. The Multimodal Feature Embedding module serves as an intermediate bridge to project these features into a shared $d_{model}$-dimensional space and enrich them with structural information.

**Token Projection.** We apply separate learnable linear projections to map each feature type into the common space $d_{model}$:

$$F'_A = F_A W_A + b_A,$$
$$F'_S = F_S W_S + b_S,$$
$$F'_M = F_M W_M + b_M,$$

where $W_A \in \mathbb{R}^{d_A \times d_{model}}$, $W_S \in \mathbb{R}^{d_S \times d_{model}}$, and $W_M \in \mathbb{R}^{d_M \times d_{model}}$ are the weight matrices, and $b_A, b_S, b_M \in \mathbb{R}^{d_{model}}$ are the bias vectors. Through this projection, we obtain three sets of feature tokens: $F'_A$, $F'_S$, and $F'_M$, all belonging to $\mathbb{R}^{G \times d_{model}}$.

**Feature Type and Positional Embeddings.** Inspired by the input representation mechanism in BERT [$\sout{CITE}$](), we introduce two auxiliary components to enrich the projected tokens: learnable feature type embeddings and unlearnable positional embeddings. During text generation, the decoders will attend to all three types of features simultaneously. Therefore, to help the model distinguish the origin of each token, we add a modality-specific feature type embedding ($T_A, T_S, T_M \in \mathbb{R}^{d_{model}}$). Simultaneously, we add a shared unlearnable positional embedding ($P \in \mathbb{R}^{G \times d_{model}}$) to help the model recognize the temporal order of the GOPs. Finally, a modality-specific Layer Normalization is applied to stabilize the training process:

$$E_A = \text{LayerNorm}_A(F'_A + T_A + P),$$
$$E_S = \text{LayerNorm}_S(F'_S + T_S + P),$$
$$E_M = \text{LayerNorm}_M(F'_M + T_M + P).$$

After this step, we collect three sets of normalized embedding tokens: $E_A$, $E_S$, and $E_M \in \mathbb{R}^{G \times d_{model}}$.

**Interleaved Concatenation.** To form the final input sequence for the decoder, we do not simply append the entire feature sequences end-to-end. Instead, we physically interleave the tokens from each GOP to ensure that the appearance, semantic, and motion information of the exact same time-step stay close together. The final multimodal input sequence $E$ is constructed as:

$$E = [e_A^{(1)}, e_S^{(1)}, e_M^{(1)}, \dots, e_A^{(G)}, e_S^{(G)}, e_M^{(G)}] \in \mathbb{R}^{3G \times d_{model}},$$

where $e_A^{(g)}, e_S^{(g)}, e_M^{(g)}$ denote the $g$-th token from $E_A, E_S,$ and $E_M$, respectively.

**Dual Embedding Module Strategy.** As illustrated in Figure 3, our bidirectional architecture consists of a Forward Decoder and a Backward Decoder. Because generating text from left-to-right and right-to-left involves entirely different decoding objectives, we construct two separate Multimodal Feature Embedding modules. These modules share the identical mathematical operations mentioned above but maintain completely independent learnable weights. This uncoupled design allows each decoder to learn a specialized visual representation that is strictly optimized for its specific decoding direction. Consequently, we obtain two distinct multimodal sequences: $\overleftarrow{E}$ (for the Backward Decoder) and $\overrightarrow{E}$ (for the Forward Decoder). 
