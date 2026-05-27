## 4.3. Multimodal Feature Embedding

The feature extraction process yields three distinct types of features: appearance ($F_A$), semantic ($F_S$), and motion ($F_M$). Because these features are extracted from different pre-trained models, they do not share the same dimensional space. The multimodal feature embedding module serves as an intermediate bridge to project these features into a shared $d_{model}$-dimensional space and enrich them with structural information.

**Token Projection.** We apply separate learnable linear projections to map each feature type into the common space $d_{model}$:

$$F'_A = F_A W_A + b_A,$$
$$F'_S = F_S W_S + b_S,$$
$$F'_M = F_M W_M + b_M,$$

where $W_A \in \mathbb{R}^{d_A \times d_{model}}$, $W_S \in \mathbb{R}^{d_S \times d_{model}}$, and $W_M \in \mathbb{R}^{d_M \times d_{model}}$ are the weight matrices, and $b_A, b_S, b_M \in \mathbb{R}^{d_{model}}$ are the bias vectors. Through this projection, we obtain three sets of feature tokens: $F'_A$, $F'_S$, and $F'_M$. All of them now have the same shape of $\mathbb{R}^{G \times d_{model}}$.

**Feature Type and Positional Embeddings.** Inspired by the input representation mechanism in BERT [$\sout{CITE}$](), we incorporate two types of auxiliary information to enrich the projected tokens: learnable feature type embeddings and fixed positional embeddings. **During text generation, the decoders will attend to all three types of features at the same time. Therefore, to help the model distinguish the origin of each token, we add a modality-specific feature type embedding $(\text{TE}_A, \text{TE}_S, \text{TE}_M \in \mathbb{R}^{d_{model}})$.** Simultaneously, we add a shared fixed positional embedding $(\text{PE} \in \mathbb{R}^{G \times d_{model}})$ to help the model recognize the temporal order of the GOPs. Finally, a learnable modality-specific normalization layer is applied to stabilize the training process:

$$E_A = \text{Norm}_A(F'_A + \text{TE}_A + \text{PE}),$$
$$E_S = \text{Norm}_S(F'_S + \text{TE}_S + \text{PE}),$$
$$E_M = \text{Norm}_M(F'_M + \text{TE}_M + \text{PE}).$$

After this step, we collect three sets of normalized embedding tokens: $E_A$, $E_S$, and $E_M \in \mathbb{R}^{G \times d_{model}}$.

**Interleaved Concatenation.** To construct the unified representation for the decoders, we do not simply concatenate the modality-specific embedding sequences consecutively. Instead, we interleave the tokens from each GOP to ensure that the appearance, semantic, and motion information of the exact same time step stay close together. The unified multimodal embeddings $E$ are formulated as:

$$E = [e_A^{(1)}, e_S^{(1)}, e_M^{(1)}, \dots, e_A^{(G)}, e_S^{(G)}, e_M^{(G)}] \in \mathbb{R}^{3G \times d_{model}},$$

where $e_A^{(g)}, e_S^{(g)}, e_M^{(g)}$ denote the $g$-th token from $E_A, E_S,$ and $E_M$, respectively.

**Dual Embedding Module Strategy.** As illustrated in Figure [$\sout{???}$](), our bidirectional architecture consists of a backward decoder and a forward decoder. To support their distinct but complementary roles, and to provide each decoder with a dedicated representational space, we construct two separate multimodal feature embedding modules. These modules share the aforementioned mathematical operations but maintain **independent learnable weights**. This decoupled design allows each decoder to learn a specialized multimodal representation **tailored to** its specific role of reverse or forward caption generation. Consequently, this strategy yields two distinct sets of unified multimodal embeddings: $\overleftarrow{E}$ and $\overrightarrow{E}$ for the backward and forward decoders, respectively.

