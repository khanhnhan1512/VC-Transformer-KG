
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

During the training o