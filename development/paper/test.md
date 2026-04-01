## 4.4. Backward Decoder (BD)

The Backward Decoder (BD) follows the standard Transformer decoder architecture. However, instead of generating text in the conventional left-to-right manner, it is trained to predict the video caption in reverse order. By processing the multimodal input sequence $\overleftarrow{E}$ alongside the previously generated words, the BD captures critical right-to-left contextual dependencies.

Mathematically, let $\overleftarrow{\hat{Y}_{<t'}}=[\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{t'-1}}]$ be the sequence of words predicted during the first $t'-1$ time steps, and let $\overleftarrow{Z_{<t'}} = [\overleftarrow{z_1},\dots,\overleftarrow{z_{t'-1}}]$ be their corresponding word embedding vectors. To predict the next word $\overleftarrow{\hat{y}_{t'}}$ at time step $t'$, these embeddings pass through the core sub-layers of a decoder block: masked self-attention, cross-attention, and a feed-forward network (FFN). The computational workflow for a single layer is formulated as follows:

$$ \overleftarrow{U_{<t'}} = \text{Masked-Self-Attention}(\overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}, \overleftarrow{Z_{<t'}}), $$
$$ \overleftarrow{Q_{t'}} = \text{Cross-Attention}(\overleftarrow{U_{<t'}}, \overleftarrow{E}, \overleftarrow{E}), $$
$$ \overleftarrow{K_{t'}} = \text{FFN}(\overleftarrow{Q_{t'}}), $$

where $\overleftarrow{U_{<t'}}$, $\overleftarrow{Q_{t'}}$, and $\overleftarrow{K_{t'}}$ represent the output hidden states of the specific sub-layers. As detailed in Section [$\sout{???}$](), every sub-layer systematically incorporates the Peri-LN strategy and a residual connection.

To capture deep semantic relationships, we construct the complete BD by stacking $N_{BD}$ identical decoding layers. The calculation process described above is repeated recursively across all layers. After propagating through the entire stack, the output from the final FFN sub-layer, denoted as $\overleftarrow{K_{t'}^{(N_{BD})}}$, is passed through an additional normalization layer to stabilize the final representation:

$$ \overleftarrow{H_{t'}} = \text{Norm}\left(\overleftarrow{K_{t'}^{(N_{BD})}}\right). $$

This normalized hidden state $\overleftarrow{H_{t'}}$ is then projected through a linear layer and a softmax function to calculate the probability distribution of the predicted word at the current time step $t'$:

$$ P(\overleftarrow{\hat{y}_{t'}} \mid \overleftarrow{\hat{y}_{<t'}}, \overleftarrow{E}) = \text{Softmax}(\text{Linear}(\overleftarrow{H_{t'}})). $$

This auto-regressive generation continues iteratively until the model predicts the end marker $\langle \text{S} \rangle$, which terminates the reverse caption generation. We express the fully generated sequence as $\overleftarrow{\hat{Y}}= [\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{T'}},\langle \text{S} \rangle]$. Notably, the specific hidden state $\overleftarrow{H_{T'}}$ produced when predicting the final actual word $\overleftarrow{\hat{y}_{T'}}$ (prior to the end marker) is extracted to serve as the global backward context $\overleftarrow{H}$. Crucially, because this final representation strictly encodes the full semantic context in a right-to-left manner, it is preserved and subsequently utilized by the Forward Decoder as an auxiliary context to guide its decoding process.

## 4.5. Forward Decoder (FD)

The Forward Decoder (FD) drives the final stage of our video captioning model by generating the text in a standard left-to-right manner. Its architectural design is highly analogous to the Backward Decoder discussed in Section [$\sout{???}$](). However, to effectively guide the decoding process, the FD incorporates an additional Cross-Attention sub-layer. This crucial modification allows the FD to simultaneously attend to the multimodal input sequence $\overrightarrow{E}$ and the global backward context $\overleftarrow{H}$ provided by the BD. By doing so, the FD continuously grounds its predictions on the rich right-to-left semantic context of the entire video.

Given the word sequence generated during the previous $t − 1$ time steps, denoted as $\overrightarrow{\hat{Y}_{<t}}=[\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{t-1}}]$, and their corresponding word embeddings $\overrightarrow{Z_{<t}}$, the workflow of a single FD layer to predict the word $\overrightarrow{\hat{y}_{t}}$ is formulated as follows:

$$ \overrightarrow{U_{<t}} = \text{Masked-Self-Attention}(\overrightarrow{Z_{<t}}, \overrightarrow{Z_{<t}}, \overrightarrow{Z_{<t}}), $$
$$ \overrightarrow{Q_{t}} = \text{Cross-Attention}(\overrightarrow{U_{<t}}, \overrightarrow{E}, \overrightarrow{E}), $$
$$ \overrightarrow{Q'_{t}} = \text{Cross-Attention}(\overrightarrow{Q_{t}}, \overleftarrow{H}, \overleftarrow{H}), $$
$$ \overrightarrow{K_{t}} = \text{FFN}(\overrightarrow{Q'_{t}}), $$

where the newly introduced output $\overrightarrow{Q'_{t}}$ perfectly acts as a bridge, synergizing the multimodal visual features ($\overrightarrow{Q_{t}}$) with the backward contextual knowledge ($\overleftarrow{H}$). As in the BD, all sub-layers actively utilize the Peri-LN strategy and residual connections.

Similar to the BD, we formulate the complete FD by stacking $N_{FD}$ identical decoding layers. After the representations recursively propagate through the entire network, the output from the final layer's FFN, denoted as $\overrightarrow{K_{t}^{(N_{FD})}}$, is stabilized by an additional normalization layer to extract the ultimate forward hidden state $\overrightarrow{H_{t}}$:

$$ \overrightarrow{H_{t}} = \text{Norm}\left(\overrightarrow{K_{t}^{(N_{FD})}}\right). $$

Finally, this hidden state is projected through a linear layer and a softmax function to compute the probability distribution of the current word:

$$ P(\overrightarrow{\hat{y}_{t}} \mid \overrightarrow{\hat{y}_{<t}}, \overrightarrow{E}, \overleftarrow{H}) = \text{Softmax}(\text{Linear}(\overrightarrow{H_{t}})). $$

This left-to-right generation iteratively continues until the model outputs the end marker $\langle \text{S} \rangle$, which concludes the entire video captioning process. We express it as $\overrightarrow{\hat{Y}}= [\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{T}},\langle \text{S} \rangle]$.
