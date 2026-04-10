# 3. Background

## 3.1. Group of Pictures (GOP) Structure

In modern video compression standards, such as H.264 and H.265, reducing temporal redundancy between consecutive frames is a key mechanism for optimizing storage. Based on data dependencies and coding techniques, frames within a compressed video sequence are generally categorized into three primary types [$\sout{CITE}$-Compressed Video Contrastive Learning]():

1. **I-frame (Intra-coded frame):** An I-frame, also known as a keyframe, is encoded independently using intra-prediction. It does not rely on any other frames for reconstruction. As a result, an I-frame contains complete appearance information, capturing the full spatial context and original objects within a scene.
2. **P-frame (Predictive frame):** A P-frame uses uni-directional inter-prediction. To save storage space, it only records the visual differences relative to a preceding reference frame (which can be either an I-frame or a previously decoded P-frame). These differences are efficiently encoded using motion vectors and residual errors.
3. **B-frame (Bi-predictive frame):** A B-frame uses bi-directional inter-prediction. It references information from both past and future frames to find the best matching blocks, maximizing storage efficiency. Similar to P-frames, its content is represented by motion vectors and residuals.

**Group of Pictures (GOP) Structure.** The I-frames, P-frames, and B-frames are not arranged randomly; rather, they follow a periodic, repeating pattern known as the Group of Pictures (GOP) structure. As illustrated in Figure [$\sout{???}$](), each GOP strictly begins with an I-frame and includes all subsequent P-frames and B-frames until the next I-frame appears. 

<br>
<figure style="align: left; text-align:center;">
    <img src="figures/New-GOP-Structure.svg" >
    <figcaption>Figure 1. The Group of Pictures (GOP) structure in compressed video. In each GOP, the first frame is always an I-frame, which is then followed by several P/B-frames until the next I-frame appears. The arrows indicate the reference dependencies for motion compensation.</figcaption>
</figure>
<br><br>

Typically, a GOP serves as an independently decodable unit within the video bitstream (often referred to as a closed GOP). This means that frames within one specific GOP do not reference any frames located in adjacent GOPs. Because of this structural independence, we can naturally view a compressed video as a continuous sequence of GOPs rather than a sequence of individual frames, treating each GOP as a distinct structural “unit of information”.

**Discussion.** In traditional video captioning frameworks, models often process densely sampled individual frames. However, this dense sampling strategy is highly computationally expensive and often introduces massive redundant visual information, which can easily overwhelm the video captioning network. By shifting the perspective and utilizing the GOP structure as the fundamental input unit, we can effectively eliminate temporal redundancy while preserving the most critical spatio-temporal dynamics required to generate accurate captions.

## 3.2. Transformer Building Blocks

The standard Transformer architecture was originally designed with an encoder and a decoder, each composed of a stack of identical layers. Each layer typically contains multiple functional sub-layers, each built around either the attention mechanism or the position-wise feed-forward network (FFN), paired with a residual connection and layer normalization.

### 3.2.1. Attention Mechanism

**Scaled Dot-Product Attention.** At the core of the Transformer is the scaled dot-product attention mechanism, which computes attention scores by mapping a set of query $(Q)$, key $(K)$, and value $(V)$ representations. The general mathematical formulation is expressed as follows:

$$
\begin{align}
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V, 
\end{align}
$$

where $\sqrt{d_k}$ is a scaling factor based on the dimension of the keys. As $d_k$ increases, the variance of the dot products between $Q$ and $K$ tends to grow, resulting in extremely large values. These large values push the softmax function into saturated regions with extremely small gradients. By scaling down the raw attention scores by $\sqrt{d_k}$, we effectively normalize the inputs to a reasonable range, which prevents “softmax saturation” and ensures stable gradients during the training process.

**Multi-Head Attention.** Instead of performing a single attention function, Transformer models typically use multi-head attention (MHA). This mechanism allows the model to jointly attend to information from different representation subspaces. The MHA is computed as follows:

$$ 
\begin{align}
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O, 
\end{align}
$$

$$ 
\begin{align}
\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V),
\end{align}
$$

where $W_i^Q$, $W_i^K$, and $W_i^V$ are learned projection matrices for each head $i$, $W^O$ is the learned output projection matrix to combine the gathered information, and $h$ is the number of heads. Multiple heads enable the model to capture different types of relationships in parallel.

**Self-Attention and Cross-Attention.** Based on the origin of $Q$, $K$, and $V$, attention modules in the Transformer can be categorized into two primary types: self‑attention and cross‑attention.
- **Self-Attention:** In a self‑attention mechanism, $Q$, $K$, and $V$ are all derived from the same input sequence (e.g., the hidden states of previously generated tokens). Self‑attention enables the model to effectively capture internal dependencies among elements within the same sequence.
- **Cross-Attention:** Cross‑attention occurs when $Q$, $K$, and $V$ come from different sources. In this case, $Q$ is taken from one representation (e.g., the current decoder states), while $K$ and $V$ are taken from another (e.g., encoder outputs). Cross‑attention naturally acts as a routing mechanism that helps the attention module gather relevant context from various information sources.

### 3.2.2. Position-Wise Feed-Forward Network (FFN)

The position-wise feed-forward network (FFN) typically consists of two linear transformations separated by a non-linear activation function. In traditional Transformer architectures, the most commonly used activation function is ReLU. However, in this study, we replace ReLU with the Gaussian Error Linear Unit (GELU) [$\sout{CITE}$-GAUSSIAN ERROR LINEAR UNITS]() activation, following the successful practices of well-known models like Google BERT [$\sout{CITE}$-BERT]() and OpenAI GPT. The computation of the FFN with a GELU activation is as follows:

$$
\begin{align}
\text{FFN}(X) = \text{GELU}(X W_1 + b_1) W_2 + b_2,
\end{align}
$$

where $W_1$, $W_2$, $b_1$, and $b_2$ are learnable parameters. GELU is defined as

$$
\begin{align}
\text{GELU}(x) = x\Phi(x),
\end{align}
$$

where $\Phi(x)$ denotes the cumulative distribution function (CDF) of the standard Gaussian distribution. A commonly used practical approximation is

$$
\begin{align}
\text{GELU}(x)\approx 0.5x\big(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3)]\big).
\end{align}
$$

### 3.2.3. Layer Normalization Strategies

During the training of deep Transformer models, the choice of layer normalization (LN) placement plays a critical role in controlling gradient stability and convergence speed. Historically, two main strategies, Post-LN and Pre-LN, have been widely adopted despite their limitations in large-scale training.

**Post-LN.** In this strategy, normalization is applied after adding the module’s output to the residual stream [$\sout{CITE}$-Attention is all you need]():

$$y_l = \text{Norm} \big(x_l + \text{Module}(x_l)\big), $$

where $x_l$ and $y_l$ represent the input and output hidden states of the $l$-th sub-layer, respectively. Here, $\text{Module}$ denotes either the Attention or FFN module in the Transformer sub-layer, and $\text{Norm}$ refers to a normalization operation such as LayerNorm or RMSNorm.

Although Post-LN effectively limits the variance of the hidden states, it often weakens gradient signals as they propagate backward. This leads to the vanishing gradient problem in deep networks, resulting in slower and unstable convergence [$\sout{CITE}$-On layer normalization in the transformer architecture | Transformers get stable: An end-to-end signal propagation theory for language models]().

**Pre-LN.** To address the gradient flow issue, Pre-LN applies normalization to the hidden state before it enters the module [$\sout{CITE}$-On layer normalization in the transformer architecture]():

$$y_l = x_l + \text{Module}\big(\text{Norm}(x_l)\big). $$

Although Pre-LN significantly improves gradient propagation during early training [$\sout{CITE}$-On layer normalization in the transformer architecture](), it leaves the main residual path completely unnormalized. As a result, the variance of hidden states can accumulate exponentially across successive sub-layers, causing “massive activations” that can severely destabilize the optimization process [$\sout{CITE}$-Massive activations in large language models]().

**Peri-LN: An Enhanced Normalization Strategy.** To overcome the main weaknesses of both Post-LN and Pre-LN, recent studies have begun to adopt a third strategy termed **Peri-LN** [$\sout{CITE}$-Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](). Essentially, Peri-LN can be viewed as an improved version of Pre-LN, where an additional normalization layer (Output-LN) is placed immediately after the module’s output:

$$y_l = x_l + \text{Norm}\Big(\text{Module}\big(\text{Norm}(x_l)\big)\Big). $$

By simultaneously normalizing both the inputs and the outputs of the core computational module, Peri-LN acts as a robust self-regularizing mechanism. It regulates the variance from both ends of the computation, effectively introducing a damping factor that prevents sudden spikes in gradients even when the module produces extremely large activation values. For clarity, the placements of the normalization layers in the Post-LN, Pre-LN, and Peri-LN architectures are visually compared in Figure [$\sout{???}$]().

<br>
<figure style="align: left; text-align:center;">
    <img src="figures/New-LN-Strategies.svg" >
    <figcaption>Figure 2. The placements of normalization layers in a Transformer sub-layer. From left to right: the Post-LN, Pre-LN, and Peri-LN strategies.</figcaption>
</figure>
<br><br>

By achieving balanced variance growth and stable gradient flow, Peri-LN guarantees high convergence stability. Driven by these clear benefits, we universally adopt this normalization strategy, paired with a standard residual connection, across every sub-layer in our proposed video captioning model. As this structural pattern is strictly consistent throughout the entire network, we purposely omit the explicit notation of Peri-LN and residual connections from both the mathematical formulas and the architecture diagrams in subsequent sections. This abstraction helps avoid unnecessary repetition and keeps the model description concise and easy to follow.

# 4. Method

## 4.1. Overview

Our proposed architecture, named BiDecT, focuses on fully leveraging bidirectional context during the video caption generation process. The core structural difference between BiDecT and previous approaches [$\sout{CITE}$-BiTransformer | BTKG]() is its encoder-free design. Specifically, our model completely removes the conventional intermediate encoder and directly integrates multimodal features into a Transformer-based bidirectional decoder system. This design is motivated by the observation that multimodal features extracted from large pre-trained models already possess robust representational power. Consequently, passing these features directly into the decoders not only reduces computational overhead but also minimizes the potential information loss typically encountered when propagating features through multiple intermediate encoder layers.

As shown in Figure [$\sout{???}$](), instead of processing individual video frames, we optimize the computational cost by treating the input video as a sequence of GOPs (as mentioned in Section [$\sout{???}$]()). The core processing pipeline of BiDecT consists of four main components. First, the **Multimodal Feature Extraction** uniformly obtains three types of complementary information (namely appearance, semantic, and motion features) across all GOPs in the video. Next, the **Multimodal Feature Embedding** transforms these extracted features into the model dimension and effectively combines the three independent streams into one unified representation for the decoders. Then, the **Backward Decoder (BD)** acts as a context predictor. Utilizing the multimodal embedding tokens from the video, it performs the caption generation process in a reverse direction (from right to left) to establish a global backward context $\overleftarrow{H}$ for the main decoder. Finally, the **Forward Decoder (FD)** serves as the primary caption generator. It simultaneously uses both the unified multimodal embeddings from the video and the global backward context $\overleftarrow{H}$ provided by the BD to generate the final video caption in the standard left-to-right direction. The detailed design of each component will be introduced in the following sections.

![](figures/New-Architecture.svg)<br>
Figure 3. An overview of the proposed BiDecT architecture for video captioning. To reduce the computational cost, we treat the input video as a sequence of GOPs. From this sequence, the Multimodal Feature Extraction uniformly obtains appearance, semantic, and motion features across all GOPs. These features are then projected into a shared dimensional space and combined into a unified representation through the Multimodal Feature Embedding. Next, leveraging this unified representation, the Backward Decoder (BD) generates a global backward context $\overleftarrow{H}$ by predicting the caption from right to left. Finally, the Forward Decoder (FD) uses both the unified multimodal embeddings and the global backward context $\overleftarrow{H}$ to generate the final caption from left to right.
<br><br>

## 4.2. Video Representation and Multimodal Feature Extraction

**Video Representation.** As mentioned in Section [$\sout{???}$](), to improve computational efficiency and eliminate redundant visual information, we process the video based on its compression structure rather than frame-by-frame. Given an input video $X$, we represent it as a sequence of Groups of Pictures (GOPs):

$$X = \left[ \text{GOP}^{(1)}, \dots, \text{GOP}^{(G)} \right],$$

where $G$ is the maximum number of sampled GOPs per video. Each GOP strictly begins with an I-frame, followed by a series of predictive frames (P/B-frames):

$$\text{GOP}^{(g)} = \left[\text{I}^{(g)}, \text{P/B}^{(g, 1)}, \dots, \text{P/B}^{(g, \text{KeyInt}-1)}\right].$$

Here, $\text{KeyInt}$ (Keyframe Interval) is a hyperparameter defining the maximum distance between two consecutive I-frames during the video compression process. Therefore, a single GOP contains at most $\text{KeyInt}-1$ consecutive P/B-frames.

**Feature Extraction Process.** After segmenting the video into GOPs, the next critical step is extracting robust feature representations from each GOP. Historically, using appearance and motion features has been the standard and widespread practice for building multimodal video representations. While effective, relying only on these visual signals often leaves a “semantic gap” between low-level visual content and high-level natural language descriptions. To bridge this gap and enhance semantic understanding, we introduce a third modality: the semantic feature.

To ensure data consistency, we apply the feature extraction process uniformly across every GOP. For a given $\text{GOP}^{(g)}$, we extract three different types of information.

**Appearance and Semantic Features (from I-frame).** Because the I-frame contains the most representative static visual information within each GOP, it is the ideal candidate to capture not only the physical appearance but also the high-level semantic meaning of the entire GOP. Therefore, we use the I-frame as the single source to extract both features through a collaborative pipeline using pre-trained BLIP-2 [$\sout{CITE}$]() and SRoBERTa [$\sout{CITE}$](). First, we feed the I-frame into the Image Encoder of BLIP-2 and extract the $\text{[CLS]}$ token to serve as our appearance feature vector $a^{(g)}$. Next, the encoder’s outputs are reused in the full BLIP-2 pipeline, passing through its two remaining components (the Querying Transformer and the Large Language Model) to generate a descriptive text caption for that exact I-frame. Finally, this generated caption is processed by SRoBERTa to produce a dense vector, which acts as our semantic feature vector $s^{(g)}$.

**Motion Feature (from GOP sequence).** Although the I-frame provides strong static context, it lacks the temporal dynamics necessary to capture motion and short-term temporal transitions within each GOP. To address this limitation, we use a pre-trained MViTv2 [$\sout{CITE}$]() to process the sequence of frames within the $\text{GOP}^{(g)}$. This model captures these dynamics and outputs a motion feature vector $m^{(g)}$.

By applying this pipeline to all $G$ GOPs in the video, we successfully collect three distinct features. Mathematically, the video $X$ is now represented by:

1. Appearance feature: $F_A = [a^{(1)}, \dots, a^{(G)}]$, where $a^{(g)} \in \mathbb{R}^{d_A}$ and $F_A \in \mathbb{R}^{G \times d_A}$.
2. Semantic feature: $F_S = [s^{(1)}, \dots, s^{(G)}]$, where $s^{(g)} \in \mathbb{R}^{d_S}$ and $F_S \in \mathbb{R}^{G \times d_S}$.
3. Motion feature: $F_M = [m^{(1)}, \dots, m^{(G)}]$, where $m^{(g)} \in \mathbb{R}^{d_M}$ and $F_M \in \mathbb{R}^{G \times d_M}$.

Here, $d_A$, $d_S$, and $d_M$ are the hidden dimensions of the respective pre-trained models. Once extracted, these three distinct features serve as the direct inputs for the multimodal feature embedding module.

## 4.3. Multimodal Feature Embedding

The feature extraction process yields three distinct types of features: appearance ($F_A$), semantic ($F_S$), and motion ($F_M$). Because these features are extracted from different pre-trained models, they do not share the same dimensional space. The multimodal feature embedding module serves as an intermediate bridge to project these features into a shared $d_{model}$-dimensional space and enrich them with structural information.

**Token Projection.** We apply separate learnable linear projections to map each feature type into the common space $d_{model}$:

$$F'_A = F_A W_A + b_A,$$
$$F'_S = F_S W_S + b_S,$$
$$F'_M = F_M W_M + b_M,$$

where $W_A \in \mathbb{R}^{d_A \times d_{model}}$, $W_S \in \mathbb{R}^{d_S \times d_{model}}$, and $W_M \in \mathbb{R}^{d_M \times d_{model}}$ are the weight matrices, and $b_A, b_S, b_M \in \mathbb{R}^{d_{model}}$ are the bias vectors. Through this projection, we obtain three sets of feature tokens: $F'_A$, $F'_S$, and $F'_M$. All of them now have the same shape of $\mathbb{R}^{G \times d_{model}}$.

**Feature Type and Positional Embeddings.** Inspired by the input representation mechanism in BERT [$\sout{CITE}$](), we incorporate two types of auxiliary information to enrich the projected tokens: learnable feature type embeddings and fixed positional embeddings. During text generation, the decoders will attend to all three types of features at the same time. Therefore, to help the model distinguish the origin of each token, we add a modality-specific feature type embedding $(\text{TE}_A, \text{TE}_S, \text{TE}_M \in \mathbb{R}^{d_{model}})$. Simultaneously, we add a shared fixed positional embedding $(\text{PE} \in \mathbb{R}^{G \times d_{model}})$ to help the model recognize the temporal order of the GOPs. Finally, a learnable modality-specific normalization layer is applied to stabilize the training process:

$$E_A = \text{Norm}_A(F'_A + \text{TE}_A + \text{PE}),$$
$$E_S = \text{Norm}_S(F'_S + \text{TE}_S + \text{PE}),$$
$$E_M = \text{Norm}_M(F'_M + \text{TE}_M + \text{PE}).$$

After this step, we collect three sets of normalized embedding tokens: $E_A$, $E_S$, and $E_M \in \mathbb{R}^{G \times d_{model}}$.

**Interleaved Concatenation.** To construct the unified representation for the decoders, we do not simply concatenate the modality-specific embedding sequences consecutively. Instead, we interleave the tokens from each GOP to ensure that the appearance, semantic, and motion information of the exact same time step stay close together. The unified multimodal embeddings $E$ are formulated as:

$$E = [e_A^{(1)}, e_S^{(1)}, e_M^{(1)}, \dots, e_A^{(G)}, e_S^{(G)}, e_M^{(G)}] \in \mathbb{R}^{3G \times d_{model}},$$

where $e_A^{(g)}, e_S^{(g)}, e_M^{(g)}$ denote the $g$-th token from $E_A, E_S,$ and $E_M$, respectively.

**Dual Embedding Module Strategy.** As illustrated in Figure [$\sout{???}$](), our bidirectional architecture consists of a backward decoder and a forward decoder. To support their distinct but complementary roles, and to provide each decoder with a dedicated representational space, we construct two separate multimodal feature embedding modules. These modules share the aforementioned mathematical operations but maintain independent learnable weights. This decoupled design allows each decoder to learn a specialized multimodal representation tailored to its specific role of reverse or forward caption generation. Consequently, this strategy yields two distinct sets of unified multimodal embeddings: $\overleftarrow{E}$ and $\overrightarrow{E}$ for the backward and forward decoders, respectively.

## 4.4. Backward Decoder (BD)

The backward decoder (BD) follows the standard Transformer decoder architecture. However, instead of generating text in the conventional left-to-right manner, it is trained to predict the video caption in reverse order. By processing the unified multimodal embeddings $\overleftarrow{E}$ alongside the previously generated words, the BD explicitly captures critical right-to-left contextual dependencies, serving as a vital structural complement to the conventional left-to-right caption generation.

Mathematically, let $\overleftarrow{\hat{Y}_{<t'}}=[\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{t'-1}}]$ be the sequence of words predicted during the first $t'-1$ time steps, and $\overleftarrow{Z_{<t'}} = [\overleftarrow{z_1},\dots,\overleftarrow{z_{t'-1}}]$ be the corresponding normalized word embeddings. To predict the word $\overleftarrow{\hat{y}_{t'}}$ at time step $t'$, the embedding vector of the immediate previous word $\overleftarrow{z_{t'-1}}$ must pass through the core sub-layers of a decoder layer: a masked self-attention mechanism, a cross-attention mechanism, and a position-wise feed-forward network. The computational workflow for a single BD layer is formulated as follows:

$$ \overleftarrow{u_{t'}} = \text{Masked-Self-Attention}(\overleftarrow{z_{t'-1}}, \overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}), $$
$$ \overleftarrow{q_{t'}} = \text{Cross-Attention}(\overleftarrow{u_{t'}}, \overleftarrow{E}, \overleftarrow{E}), $$
$$ \overleftarrow{k_{t'}} = \text{FFN}(\overleftarrow{q_{t'}}), $$

where $\overleftarrow{u_{t'}}$, $\overleftarrow{q_{t'}}$, and $\overleftarrow{k_{t'}}$ represent the output hidden state vectors of the respective sub-layers at step $t'$. As detailed in Section [$\sout{???}$](), each sub-layer systematically employs the Peri-LN strategy alongside a residual connection.

To capture deep contextual relationships, we construct the complete BD by stacking $N_{BD}$ identical backward decoder layers. The sequence of sub-layer transformations described above is applied iteratively across all layers. After propagating through the entire stack, the output vector from the last FFN sub-layer, denoted as $\overleftarrow{k_{t'}^{(N_{BD})}}$, is passed through an additional normalization layer to obtain the final backward hidden state vector $\overleftarrow{h_{t'}}$:

$$ \overleftarrow{h_{t'}} = \text{Norm}\left(\overleftarrow{k_{t'}^{(N_{BD})}}\right). $$

This normalized hidden state vector $\overleftarrow{h_{t'}}$ is then projected through a linear layer and a softmax function to calculate the probability distribution of the predicted word $\overleftarrow{\hat{y}_{t'}}$:

$$ P(\overleftarrow{\hat{y}_{t'}} \mid \overleftarrow{\hat{Y}_{<t'}}, \overleftarrow{E}) = \text{Softmax}(\text{Linear}(\overleftarrow{h_{t'}})). $$

This autoregressive generation continues iteratively until the model predicts the end-of-sequence marker $\langle \text{S} \rangle$, which terminates the reverse captioning process. The fully generated backward caption is denoted as $\overleftarrow{\hat{Y}}= [\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{T'}},\langle \text{S} \rangle]$.

**Global Backward Context.** In our proposed architecture, the backward hidden states generated across all time steps play an essential role beyond their primary function of predicting the reverse sequence. By preserving these states, we encapsulate the full right-to-left linguistic context of the entire generated caption. As shown in Figure [$\sout{???}$](), this comprehensive representation serves as our global backward context $\overleftarrow{H}$:

$$ \overleftarrow{H} = [\overleftarrow{h_1}, \dots, \overleftarrow{h_{|\overleftarrow{\hat{Y}}|}}]. $$

This sequence-level context $\overleftarrow{H}$ is passed to the forward decoder as an auxiliary guide for the final decoding stage.

## 4.5. Forward Decoder (FD)

The forward decoder (FD) generates the final caption in a standard left-to-right manner. Building upon the core architecture of the BD, the FD incorporates a second cross-attention sub-layer. This modification allows the FD to receive direct guidance from both the unified multimodal embeddings $\overrightarrow{E}$ and the sequence-level backward context $\overleftarrow{H}$. Consequently, at every prediction step, the FD is inherently guided by the anticipated global structure of the entire caption, empowering it to generate highly context-aware descriptions.

Given the word sequence $\overrightarrow{\hat{Y}_{<t}}=[\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{t-1}}]$ generated during the previous $t-1$ time steps, and the corresponding normalized word embeddings $\overrightarrow{Z_{<t}} = [\overrightarrow{z_1},\dots,\overrightarrow{z_{t-1}}]$, the computational workflow of a single FD layer to predict the next word $\overrightarrow{\hat{y}_{t}}$ is formulated as follows:

$$ \overrightarrow{u_{t}} = \text{Masked-Self-Attention}(\overrightarrow{z_{t-1}}, \overrightarrow{Z_{<t}}, \overrightarrow{Z_{<t}}), $$
$$ \overrightarrow{q_{t}} = \text{Cross-Attention}(\overrightarrow{u_{t}}, \overrightarrow{E}, \overrightarrow{E}), $$
$$ \overrightarrow{q'_{t}} = \text{Cross-Attention}(\overrightarrow{q_{t}}, \overleftarrow{H}, \overleftarrow{H}), $$
$$ \overrightarrow{k_{t}} = \text{FFN}(\overrightarrow{q'_{t}}), $$

where the newly introduced output vector $\overrightarrow{q'_{t}}$ acts as a bridge, integrating the attended multimodal representation $\overrightarrow{q_{t}}$ with the backward contextual knowledge from $\overleftarrow{H}$. As in the BD, each sub-layer systematically employs the Peri-LN strategy alongside a residual connection.

After propagating through $N_{FD}$ identical forward decoder layers, the output vector from the last FFN sub-layer, denoted as $\overrightarrow{k_{t}^{(N_{FD})}}$, is passed through an additional normalization layer to obtain the final forward hidden state vector $\overrightarrow{h_{t}}$:

$$ \overrightarrow{h_{t}} = \text{Norm}\left(\overrightarrow{k_{t}^{(N_{FD})}}\right). $$

This normalized hidden state vector $\overrightarrow{h_{t}}$ is then projected through a linear layer and a softmax function to calculate the probability distribution of the predicted word $\overrightarrow{\hat{y}_{t}}$:

$$ P(\overrightarrow{\hat{y}_{t}} \mid \overrightarrow{\hat{Y}_{<t}}, \overrightarrow{E}, \overleftarrow{H}) = \text{Softmax}(\text{Linear}(\overrightarrow{h_{t}})). $$

This left-to-right generation continues iteratively until the model predicts the end-of-sequence marker $\langle \text{S} \rangle$, which concludes the entire video captioning process. The final generated forward caption is denoted as $\overrightarrow{\hat{Y}}= [\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{T}},\langle \text{S} \rangle]$.

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

![](figures/New-Pseudo-Reverse-Captions.svg)<br>
Figure 4. The construction process of pseudo reverse captions. For a given video, the reference forward captions undergo a word-for-word reversal to form a video-specific backward pool. These candidates are then randomly shuffled and pseudo-paired with the original forward captions. This intra-video randomization prevents information leakage by forcing the FD to leverage the global backward context rather than exploiting trivial word-level mappings.
<br><br>
