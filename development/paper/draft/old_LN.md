### 3.2.3. Layer Normalization Strategies

During the training of deep Transformer models, the choice of layer normalization (LN) placement plays a critical role in controlling gradient stability and convergence speed. Historically, two main strategies, Post-LN and Pre-LN, have been widely adopted despite their limitations in large-scale training.

**Post-LN.** In this strategy, normalization is applied after adding the module’s output to the residual stream [$\sout{CITE}$-Attention is all you need]():

$$y_l = \text{Norm} \big(x_l + \text{Module}(x_l)\big), $$

where $x_l$ and $y_l$ represent the input and output hidden states of the $l$-th sub-layer, respectively. Here, $\text{Module}$ denotes either the Attention or FFN module in the Transformer sub-layer, and $\text{Norm}$ refers to a normalization operation such as LayerNorm or RMSNorm.

Although Post-LN effectively limits the variance of the hidden states, it often weakens gradient signals as they propagate backward. This leads to the vanishing gradient problem in deep networks, resulting in slower and unstable convergence [$\sout{CITE}$-On layer normalization in the transformer architecture | Transformers get stable: An end-to-end signal propagation theory for language models]().

**Pre-LN.** To address the gradient flow issue, Pre-LN applies normalization to the hidden state before it enters the module [$\sout{CITE}$-On layer normalization in the transformer architecture]():

$$y_l = x_l + \text{Module}\big(\text{Norm}(x_l)\big). $$

Although Pre-LN significantly improves gradient propagation during early training [$\sout{CITE}$-On layer normalization in the transformer architecture](), it leaves the main residual path completely unnormalized. As a result, the variance of hidden states can accumulate exponentially across successive sub-layers, causing “massive activations” that can severely destabilize the optimization process [$\sout{CITE}$-Massive activations in large language models]().

**Peri-LN: An Enhanced Normalization Strategy.** To overcome the main weaknesses of both Post-LN and Pre-LN, recent studies have begun to adopt a third strategy termed **Peri-LN** [$\sout{CITE}$-Peri-LN | Gemma4](). Essentially, Peri-LN can be viewed as an improved version of Pre-LN, where an additional normalization layer (Output-LN) is placed immediately after the module’s output:

$$y_l = x_l + \text{Norm}\Big(\text{Module}\big(\text{Norm}(x_l)\big)\Big). $$

By simultaneously normalizing both the inputs and the outputs of the core computational module, Peri-LN acts as a robust self-regularizing mechanism. It regulates the variance from both ends of the computation, effectively introducing a damping factor that prevents sudden spikes in gradients even when the module produces extremely large activation values. For clarity, the placements of the normalization layers in the Post-LN, Pre-LN, and Peri-LN architectures are visually compared in Figure [$\sout{???}$]().

<br>
<figure style="align: left; text-align:center;">
    <img src="figures/New-LN-Strategies.svg" >
    <figcaption>Figure 2. The placements of normalization layers in a Transformer sub-layer. From left to right: the Post-LN, Pre-LN, and Peri-LN strategies.</figcaption>
</figure>
<br><br>

By achieving balanced variance growth and stable gradient flow, Peri-LN guarantees high convergence stability. Driven by these clear benefits, we universally adopt this normalization strategy, paired with a standard residual connection, across every sub-layer in our proposed video captioning model. As this structural pattern is strictly consistent throughout the entire network, we purposely omit the explicit notation of Peri-LN and residual connections from both the mathematical formulas and the architecture diagrams in subsequent sections. This abstraction helps avoid unnecessary repetition and keeps the model description concise and easy to follow.
