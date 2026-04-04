## 4.4. Backward Decoder (BD)

The backward decoder (BD) follows the standard Transformer decoder architecture. However, instead of generating text in the conventional left-to-right manner, it is trained to predict the video caption in reverse order. By processing the unified multimodal embeddings $\overleftarrow{E}$ alongside the previously generated words, the BD explicitly captures critical right-to-left contextual dependencies, serving as a vital structural complement to the conventional left-to-right caption generation.

Mathematically, let $\overleftarrow{\hat{Y}_{<t'}}=[\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{t'-1}}]$ be the sequence of words predicted during the first $t'-1$ time steps, and let $\overleftarrow{Z_{<t'}} = [\overleftarrow{z_1},\dots,\overleftarrow{z_{t'-1}}]$ be their corresponding normalized word embeddings. To predict the word $\overleftarrow{\hat{y}_{t'}}$ at time step $t'$, the embedding vector of the immediate previous word $\overleftarrow{z_{t'-1}}$ must pass through the core sub-layers of a decoder layer: a masked self-attention mechanism, a cross-attention mechanism, and a position-wise feed-forward network. The computational workflow for a single layer is formulated as follows:

$$ \overleftarrow{u_{t'}} = \text{Masked-Self-Attention}(\overleftarrow{z_{t'-1}}, \overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}), $$
$$ \overleftarrow{q_{t'}} = \text{Cross-Attention}(\overleftarrow{u_{t'}}, \overleftarrow{E}, \overleftarrow{E}), $$
$$ \overleftarrow{k_{t'}} = \text{FFN}(\overleftarrow{q_{t'}}), $$

where $\overleftarrow{u_{t'}}$, $\overleftarrow{q_{t'}}$, and $\overleftarrow{k_{t'}}$ represent the output hidden state vectors of the respective sub-layers at step $t'$. As detailed in Section [$\sout{???}$](), each sub-layer systematically employs the Peri-LN strategy alongside a residual connection.

To capture deep semantic relationships, we construct the complete BD by stacking $N_{BD}$ identical backward decoder layers. The sequence of sub-layer transformations described above is applied iteratively across all layers. After propagating through the entire stack, the output vector from the last FFN sub-layer, denoted as $\overleftarrow{k_{t'}^{(N_{BD})}}$, is passed through an additional normalization layer to obtain the final backward hidden state vector $\overleftarrow{h_{t'}}$:

$$ \overleftarrow{h_{t'}} = \text{Norm}\left(\overleftarrow{k_{t'}^{(N_{BD})}}\right). $$

This normalized hidden state vector $\overleftarrow{h_{t'}}$ is then projected through a linear layer and a softmax function to calculate the probability distribution of the predicted word $\overleftarrow{\hat{y}_{t'}}$:

$$ P(\overleftarrow{\hat{y}_{t'}} \mid \overleftarrow{\hat{Y}_{<t'}}, \overleftarrow{E}) = \text{Softmax}(\text{Linear}(\overleftarrow{h_{t'}})). $$

This autoregressive generation continues iteratively until the model predicts the end-of-sequence marker $\langle \text{S} \rangle$, which terminates the reverse captioning process. The fully generated backward caption is denoted as $\overleftarrow{\hat{Y}}= [\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{T'}},\langle \text{S} \rangle]$.

**Global Backward Context.** In our proposed architecture, the backward hidden states generated across all time steps play an essential role beyond their primary function of predicting the reverse sequence. By preserving these states, we encapsulate the full right-to-left semantic context of the entire generated caption. As shown in Figure [$\sout{???}$](), this comprehensive representation serves as our global backward context $\overleftarrow{H}$:

$$ \overleftarrow{H} = [\overleftarrow{h_1}, \dots, \overleftarrow{h_{|\overleftarrow{\hat{Y}}|}}]. $$

This sequence-level context $\overleftarrow{H}$ is passed to the forward decoder as an auxiliary guide for the final decoding stage.

## 4.5. Forward Decoder (FD)

The forward decoder (FD) generates the final caption in a standard left-to-right manner. Building upon the core architecture of the BD, the FD incorporates a second cross-attention sub-layer. This modification allows the FD to receive direct guidance from both the unified multimodal embeddings $\overrightarrow{E}$ and the sequence-level backward context $\overleftarrow{H}$. Consequently, at every prediction step, the FD is inherently guided by the anticipated global structure of the entire caption, empowering it to generate highly context-aware descriptions.

Given the word sequence generated during the previous $t-1$ time steps $\overrightarrow{\hat{Y}_{<t}}=[\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{t-1}}]$, and their corresponding normalized word embeddings $\overrightarrow{Z_{<t}} = [\overrightarrow{z_1},\dots,\overrightarrow{z_{t-1}}]$, the computational workflow of a single FD layer to predict the next word $\overrightarrow{\hat{y}_{t}}$ is formulated as follows:

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
