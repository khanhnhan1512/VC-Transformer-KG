## 4.4. Backward Decoder (BD)

The backward decoder (BD) follows the standard Transformer decoder architecture. However, instead of generating text in the conventional left-to-right manner, it is trained to predict the video caption in reverse order. By processing the multimodal input sequence $\overleftarrow{E}$ alongside the previously generated words, the BD explicitly captures critical right-to-left contextual dependencies, serving as a vital structural complement to the conventional left-to-right caption generation.

Mathematically, let $\overleftarrow{\hat{Y}_{<t'}}=[\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{t'-1}}]$ be the sequence of words predicted during the first $t'-1$ time steps, and let $\overleftarrow{Z_{<t'}} = [\overleftarrow{z_1},\dots,\overleftarrow{z_{t'-1}}]$ be their corresponding normalized word embedding vectors. To predict the next word $\overleftarrow{\hat{y}_{t'}}$ at time step $t'$, these embeddings must first pass through the core sub-layers of a decoder layer: a masked self-attention mechanism, a cross-attention mechanism, and a position-wise feed-forward network. The computational workflow for a single layer is formulated as follows:

$$ \overleftarrow{U_{<t'}} = \text{Masked-Self-Attention}(\overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}), $$
$$ \overleftarrow{Q_{t'}} = \text{Cross-Attention}(\overleftarrow{U_{<t'}}, \overleftarrow{E}, \overleftarrow{E}), $$
$$ \overleftarrow{K_{t'}} = \text{FFN}(\overleftarrow{Q_{t'}}), $$

where $\overleftarrow{U_{<t'}}$, $\overleftarrow{Q_{t'}}$, and $\overleftarrow{K_{t'}}$ represent the output hidden states of the specific sub-layers. As detailed in Section [$\sout{???}$](), each sub-layer systematically employs the Peri-LN strategy alongside a residual connection.

To capture deep semantic relationships, we construct the complete BD by stacking $N_{BD}$ identical backward decoder layers. The calculation process described above is applied sequentially across all layers. After propagating through the entire stack, the output from the last FFN sub-layer, denoted as $\overleftarrow{K_{t'}^{(N_{BD})}}$, is passed through an additional normalization layer to obtain the final backward hidden state $\overleftarrow{H_{t'}}$:

$$ \overleftarrow{H_{t'}} = \text{Norm}\left(\overleftarrow{K_{t'}^{(N_{BD})}}\right). $$

This normalized hidden state $\overleftarrow{H_{t'}}$ is then projected through a linear layer and a softmax function to calculate the probability distribution of the next predicted word $\overleftarrow{\hat{y}_{t'}}$:

$$ P(\overleftarrow{\hat{y}_{t'}} \mid \overleftarrow{\hat{y}_{<t'}}, \overleftarrow{E}) = \text{Softmax}(\text{Linear}(\overleftarrow{H_{t'}})). $$

This autoregressive generation continues iteratively until the model predicts the end-of-sequence marker $\langle \text{S} \rangle$, which terminates the reverse captioning process. The fully generated backward caption is denoted as $\overleftarrow{\hat{Y}}= [\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{T'}},\langle \text{S} \rangle]$.

**Global Backward Context.** In our proposed architecture, the backward hidden state $\overleftarrow{H_{T'}}$, produced when predicting the final actual word $\overleftarrow{\hat{y}_{T'}}$, plays an essential role. As the final state prior to the end-of-sequence marker, it encapsulates the full right-to-left semantic context of the entire generated caption. As shown in Figure [$\sout{???}$](), we obtain this comprehensive representation to serve as our global backward context $\overleftarrow{H}$:

$$ \overleftarrow{H} = \overleftarrow{H_{T'}}. $$

This preserved context $\overleftarrow{H}$ is directly transferred to the forward decoder, acting as an auxiliary contextual guide for the final decoding stage.

## 4.5. Forward Decoder (FD)

The forward decoder (FD) generates the final caption in a standard left-to-right manner. In addition to the three sub-layers in every BD layer, the FD incorporates an additional cross-attention sub-layer. This modification allows the FD to receive direct guidance from both the multimodal input sequence $\overrightarrow{E}$ and the rich right-to-left semantic context $\overleftarrow{H}$ derived from the BD. Consequently, at every prediction step, the FD is inherently guided by the anticipated global structure of the entire caption, empowering it to generate highly context-aware descriptions.

Given the word sequence generated during the previous $t − 1$ time steps, denoted as $\overrightarrow{\hat{Y}_{<t}}=[\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{t-1}}]$, and their corresponding normalized word embedding vectors $\overrightarrow{Z_{<t}} = [\overrightarrow{z_1},\dots,\overrightarrow{z_{t-1}}]$, the workflow of a single FD layer to predict the word $\overrightarrow{\hat{y}_{t}}$ is formulated as follows:

$$ \overrightarrow{U_{<t}} = \text{Masked-Self-Attention}(\overrightarrow{Z_{<t}}, \overrightarrow{Z_{<t}}, \overrightarrow{Z_{<t}}), $$
$$ \overrightarrow{Q_{t}} = \text{Cross-Attention}(\overrightarrow{U_{<t}}, \overrightarrow{E}, \overrightarrow{E}), $$
$$ \overrightarrow{Q'_{t}} = \text{Cross-Attention}(\overrightarrow{Q_{t}}, \overleftarrow{H}, \overleftarrow{H}), $$
$$ \overrightarrow{K_{t}} = \text{FFN}(\overrightarrow{Q'_{t}}), $$

where the newly introduced output $\overrightarrow{Q'_{t}}$ acts as a bridge, integrating the attended multimodal representations $\overrightarrow{Q_{t}}$ with the backward contextual knowledge $\overleftarrow{H}$. As in the BD, the Peri-LN strategy and residual connections are systematically applied across all sub-layers.

After propagating through $N_{FD}$ stacked forward decoder layers, the output from the last FFN sub-layer, denoted as $\overrightarrow{K_{t}^{(N_{FD})}}$, is passed through an additional normalization layer to obtain the final forward hidden state $\overrightarrow{H_{t}}$:

$$ \overrightarrow{H_{t}} = \text{Norm}\left(\overrightarrow{K_{t}^{(N_{FD})}}\right). $$

Finally, this hidden state is projected through a linear layer and a softmax function to compute the probability distribution of the current word:

$$ P(\overrightarrow{\hat{y}_{t}} \mid \overrightarrow{\hat{y}_{<t}}, \overrightarrow{E}, \overleftarrow{H}) = \text{Softmax}(\text{Linear}(\overrightarrow{H_{t}})). $$

This left-to-right generation iteratively continues until the model outputs the end-of-sequence marker $\langle \text{S} \rangle$, which concludes the entire video captioning process. The final generated forward caption is denoted as $\overrightarrow{\hat{Y}}= [\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{T}},\langle \text{S} \rangle]$.
