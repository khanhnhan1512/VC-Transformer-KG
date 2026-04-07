# DRAFT

> 4.6. Optimization and Inference

We train the BiDecT model using the cross-entropy losses on both decoders. 
```
The training process is designed to optimize the parameters of both decoders simultaneously, allowing them to generate high-quality captions in both directions.
```
Let $\mathcal{D}$ be the training set of videos with provided ground-truth captions. For each video $X$, its ground-truth caption $\overrightarrow{Y}=[\overrightarrow{y_1},\dots,\overrightarrow{y_l}]$ of length $l$, and the pseudo reverse caption $\overleftarrow{Y}=[\overleftarrow{y_1},\dots,\overleftarrow{y_{l'}}]$ of length $l'$ from the training set $D$, hàm loss tổng thể $\mathcal{L}$ của mô hình BiDecT được định nghĩa như sau:

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{BD} + \lambda\,\mathcal{L}_{FD},$$

$$
\mathcal{L}_{BD} = \sum_{(X,\overleftarrow{Y})\in \mathcal{D}}\;\sum_{t'=1}^{l'} -\log \,P(\overleftarrow{y_{t'}} \mid \overleftarrow{y_{<t'}},\,X; \;\theta_{BD}),
$$

$$
\mathcal{L}_{FD} = \sum_{(X,\overrightarrow{Y})\in \mathcal{D}}\;\sum_{t=1}^{l} -\log \,P(\overrightarrow{y_{t}} \mid \overrightarrow{y_{<t}},\,X;\;\theta_{BD},\,\theta_{FD}),
$$

```
> where $\mathcal{L}_{BD}$ and $\mathcal{L}_{FD}$ denote the cross-entropy losses of the backward and forward decoders, respectively; 
> and $\overleftarrow{y_{t'}}$ and $\overrightarrow{y_{t}}$ denote the reference words for the backward and forward decoders;
```
[v] and $\lambda \in [0,1]$ is a hyperparameter that balances the two terms.  
[v] The trainable parameters of the backward and forward decoders are denoted by $\theta_{BD}$ and $\theta_{FD}$, respectively; each $\theta$ also includes the trainable parameters of the corresponding multimodal feature embedding module.  
The first term, $\mathcal{L}_{BD}$, encourages the model to learn a high-quality right-to-left semantic representation. The second term, $\mathcal{L}_{FD}$, models the forward caption generation process.  
[v] For implementation, we apply **label smoothing** to reduce overconfidence.

**Pseudo Reverse Captions.** Following prior studies on bidirectional decoding, we must avoid trivial information leakage that would arise if reverse candidates were obtained by simply reversing the exact forward ground-truth captions. This leakage occurs when the forward decoder (FD) can exploit future answers provided by the backward decoder (BD) instead of learning to integrate contextual cues.

To prevent this, the reverse captions used during training are **pseudo reverse captions**. Each video in the dataset typically has multiple ground-truth captions. For each video, we first reverse all of its forward captions (i.e., flip the word order) to form a pool of candidate reversed captions specific to that video. These reversed captions are then randomly shuffled and reassigned within the same pool for that video, so that the reverse caption paired with a given video is not necessarily the exact reversal of its corresponding forward caption. Crucially, reversed captions are not exchanged across different videos. In other words, the backward decoder is not fed the exact future ground-truth during training. This intra-video randomization mitigates information leakage and prevents the forward decoder from exploiting memorized exact reversals, thereby encouraging the model to learn genuine bidirectional semantic context.

**Testing / Inference.** Once BiDecT is trained, the caption generation during testing proceeds in two sequential stages. In order to ensure consistency between model training and testing, we apply beam search decoding in both stages. In the first stage, the backward decoder utilizes beam search to sequentially generate a reverse caption and terminates when the end marker $\langle S\rangle$ is predicted with maximum probability, producing the final reverse hidden state. In the second stage, the forward decoder applies beam search in a left-to-right manner to produce the final, best video caption.
