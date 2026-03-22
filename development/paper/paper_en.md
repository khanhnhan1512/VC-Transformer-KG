# 3. Background

## 3.1. Group of Pictures (GOP) Structure

In modern video compression standards, such as H.264 and H.265, leveraging temporal redundancy between consecutive frames is a crucial mechanism for optimizing storage. Based on data dependencies and coding techniques, frames within a compressed video sequence are generally categorized into three primary types [$\sout{CITE}$-Compressed Video Contrastive Learning]():

1. **I-frame (Intra-coded frame):** An I-frame, also known as a keyframe, is encoded independently using intra-prediction. It does not rely on any other frames for reconstruction. Consequently, an I-frame explicitly contains complete appearance information, capturing the full spatial context and original objects within a scene.
2. **P-frame (Predictive frame):** A P-frame utilizes uni-directional inter-prediction. To save storage space, it only records the visual differences relative to a preceding reference frame (which can be either an I-frame or a previously decoded P-frame). These differences are efficiently encoded using motion vectors and residual errors.
3. **B-frame (Bi-predictive frame):** A B-frame employs bi-directional inter-prediction. It references information from both past and future frames to find the best matching blocks, maximizing storage efficiency. Similar to P-frames, its content is represented by motion vectors and residuals.

**Group of Pictures (GOP) Structure.** The I-frames, P-frames, and B-frames are not arranged randomly; rather, they follow a periodic, repeating pattern known as the Group of Pictures (GOP) structure. As illustrated in Figure [$\sout{???}$](), each GOP strictly begins with an I-frame and encompasses all subsequent P-frames and B-frames until the next I-frame appears. 

<figure style="align: left; text-align:center;">
    <img src="figures/New-GOP-Structure.svg" >
    <figcaption>Figure 1. The Group of Pictures (GOP) structure in compressed video. In each GOP, the first frame is always an I-frame, which is then followed by several P/B-frames until the next I-frame appears. The arrows indicate the reference dependencies for motion compensation.</figcaption>
</figure>

Typically, a GOP serves as an independently decodable unit within the video bitstream (often referred to as a closed GOP). This means that frames within one specific GOP do not reference any frames located in adjacent GOPs. Because of this structural autonomy, we can naturally view a compressed video as a continuous sequence of GOPs rather than a sequence of individual frames, treating each GOP as a distinct semantic "unit of information".

**Discussion.** In conventional video captioning frameworks, models often process densely sampled individual frames. However, this dense sampling strategy is highly computationally expensive and inevitably introduces massive redundant visual information, which can easily overwhelm the video captioning network. By shifting the perspective and utilizing the GOP structure as the fundamental input unit, we can effectively filter out temporal redundancy while preserving the most critical spatio-temporal dynamics needed for generating accurate captions.

## 3.2. Transformer Building Blocks

The standard Transformer architecture was originally designed with an encoder block and a decoder block, each composed of a stack of identical layers. However, in many modern variants, this architecture can be flexibly adapted (e.g., employing a decoder-only structure) to suit specific tasks. The core components within these layers are the attention mechanism and the position-wise feed-forward network (FFN), along with residual connections and layer normalization.

### 3.2.1. Attention Mechanism

**Scaled Dot-Product Attention.** At the core of the Transformer is the Scaled Dot-Product Attention mechanism, which computes attention scores by mapping a set of Query $(Q)$, Key $(K)$, and Value $(V)$ representations. The general mathematical formulation is expressed as follows:

$$
\begin{align}
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V, 
\end{align}
$$

where $\sqrt{d_k}$ is a scaling factor based on the dimension of the keys. As $d_k$ increases, the variance of the dot products between $Q$ and $K$ tends to grow, resulting in excessively large values. These large values push the $\text{Softmax}$ function into saturated regions with extremely small gradients. By scaling down the raw attention scores by $\sqrt{d_k}$, we effectively normalize the inputs to a reasonable range, thereby preventing softmax saturation and ensuring stable gradients during the training process.

**Multi-Head Attention.** Instead of performing a single attention function, Transformer models typically employ Multi-Head Attention (MHA). This mechanism allows the model to jointly attend to information from different representation subspaces. The MHA is computed as follows:

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

where $W_i^Q$, $W_i^K$, $W_i^V$ are learned projection matrices for each head $i$, $W^O$ is the learned output projection matrix to aggregate the gathered information, and $h$ is the number of heads. Multiple heads enable the model to capture different types of relationships in parallel.

**Self-Attention vs. Cross-Attention.** Based on the origin of $Q$, $K$, $V$, attention modules in the Transformer can be categorized into two primary types: self‑attention and cross‑attention.
*   **Self-Attention:** In a self‑attention mechanism, $Q$, $K$, $V$ are all derived from the same input sequence (e.g., the hidden states of previously generated tokens). Self‑attention enables the model to effectively capture internal dependencies among elements within the same sequence.
*   **Cross-Attention:** Cross‑attention occurs when $Q$, $K$, $V$ come from different sources. In this case, $Q$ is taken from one representation (e.g., the current decoder states), while $K$ and $V$ are taken from another (e.g., encoder outputs). Cross‑attention intrinsically acts as a routing mechanism that helps the attention module gather necessary semantic context from various information sources.

### 3.2.2. Feed-Forward Network with GELU Activation

The second crucial component in each Transformer layer is the position-wise feed-forward network (FFN). This network typically consists of two linear transformations separated by a non-linear activation function. In traditional Transformer architectures, the most commonly used activation function is ReLU. However, in this study, we replace ReLU with the Gaussian Error Linear Unit (GELU) [$\sout{CITE}$-GAUSSIAN ERROR LINEAR UNITS]() activation, following the successful practices of well-known models like Google BERT [$\sout{CITE}$-BERT]() and OpenAI GPT. The computation of the FFN with a GELU activation is as follows:

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

where $\Phi(x)$ denotes the Cumulative Distribution Function (CDF) of the standard Gaussian distribution. A commonly used practical approximation is

$$
\begin{align}
\text{GELU}(x)\approx 0.5x\big(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3)]\big).
\end{align}
$$
