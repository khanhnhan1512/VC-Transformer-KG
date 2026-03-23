### 3.2.3. Layer Normalization Strategies

During the training of deep Transformer models, the choice of Layer Normalization (LN) placement plays a critical role in controlling gradient stability and convergence speed. Historically, two main strategies, Post-LN and Pre-LN, have been widely adopted despite their limitations in large-scale training.

**A) Post-LN.** In this strategy, normalization is applied after adding the module's output to the residual stream [$\sout{CITE}$-Attention is all you need]().

$$y_l = \text{Norm} \big(x_l + \text{Module}(x_l)\big), $$

where $x_l$ and $y_l$ represent the input and output hidden states of the $l$-th sub-layer, respectively. Here, $\text{Module}$ denotes either the Attention or FFN module in the Transformer sub-layer, and $\text{Norm}$ refers to a normalization operation such as LayerNorm or RMSNorm.

Although Post-LN effectively limits the variance of the hidden states, it often weakens gradient signals as they propagate backward. This leads to the vanishing gradient problem in deep networks, resulting in slower and unstable convergence [$\sout{CITE}$-On layer normalization in the transformer architecture | Transformers get stable: An end-to-end signal propagation theory for language models]().

**B) Pre-LN.** To address the gradient flow issue, Pre-LN applies normalization to the hidden state before it enters the module [$\sout{CITE}$-On layer normalization in the transformer architecture](). 

$$y_l = x_l + \text{Module}\big(\text{Norm}(x_l)\big). $$

Although Pre-LN significantly improves gradient propagation during early training, it leaves the main residual path completely unnormalized. As a result, the variance of hidden states can accumulate exponentially across layers, causing "massive activations" that can severely destabilize the optimization process [$\sout{CITE}$-Massive activations in large language models]().

**Peri-LN: An Enhanced Normalization Strategy.** To overcome the main weaknesses of both Post-LN and Pre-LN, recent studies have begun to adopt a third strategy termed **Peri-LN** [$\sout{CITE}$-Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](). Essentially, Peri-LN can be viewed as an improved version of Pre-LN, where an additional normalization layer (Output-LN) is placed immediately after the module's output. The mathematical formulation is defined as:

$$y_l = x_l + \text{Norm}\Big(\text{Module}\big(\text{Norm}(x_l)\big)\Big). $$

For clarity, the placements of the normalization layers in the Post-LN, Pre-LN, and Peri-LN architectures are visually compared in Figure [$\sout{???}$](). By normalizing both the inputs and the outputs of the Attention and FFN modules, Peri-LN acts as a robust self-regularizing mechanism. It regulates the variance from both ends of the module, effectively introducing a damping factor that prevents sudden spikes in gradients even when the module produces extremely large activation values. 

<figure style="align: left; text-align:center;">
    <img src="figures/New-LN-Strategies.svg" >
    <figcaption>Figure 2. The placements of normalization layers in a Transformer sub-layer. From left to right: the Post-LN, Pre-LN, and Peri-LN strategies.</figcaption>
</figure>

By achieving a more balanced variance growth and a more stable gradient flow, Peri-LN continuously guarantees high convergence stability. Driven by these clear benefits, we directly apply the Peri-LN strategy across all Transformer building blocks in our proposed video captioning model. 

**Implementation Note.** Because the Peri-LN strategy and residual connections are applied consistently to every Attention and FFN module, we omit them from both the mathematical formulas and the overall architecture diagram in the subsequent methodology sections. This simplification helps avoid unnecessary repetition and keeps the model description clear and easy to follow.
