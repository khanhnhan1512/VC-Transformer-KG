# BiDecT

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 67.7 | 45.8 | 82.9 | 138.0 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 48.5 | 31.3 | 65.2 | 64.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 40.9 | 27.0 | 54.6 | 77.7 |

---

# [25-26][][UHCL] Unified hierarchical contrastive learning for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S1566253525009182

- **Published in:** Information Fusion, Volume 127, Part B, March 2026, 103856

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.3 | 40.5 | 77.5 | 114.6 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 49.4 | 31.7 | 65.6 | 61.7 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| - | - | - | - |

- **Abstract:**

```
Video captioning involves the generation of textual descriptions for videos. Many video captioning models are typically trained using video-text pairs and adopt maximum likelihood estimation as their learning method, which overlooks caption distinctiveness and results in high resemblance in model-generated captions. Although contrastive methods improve distinctiveness, their performance is influenced by the reference model, and they rely on a two-stage training process: first training a reference model, then training a target model to surpass the reference. In this work, we propose a simple yet effective Unified Hierarchical Contrastive Learning (UHCL) method, which harnesses triamese decoders and hierarchical contrastive learning to enhance overall performance and encourage distinctiveness within a unified framework. Specifically, UHCL employs triamese decoders to compute contrastive loss using additional matched and mismatched video-caption pairs. Furthermore, adaptive token fusion module is utilized to reduce the redundancy of visual tokens, guiding the model to apply hierarchical contrastive constraints at the noun and verb levels. It is worth noting that UHCL does not require prior information, thus avoiding the risk of introducing noise, and only a single decoder is required during inference, with no additional computational overhead. Extensive experiments conducted on MSR-VTT and MSVD, demonstrate the effectiveness of the proposed method and its superiority over the state-of-the-art methods. Particularly, UHCL outperforms the best results by improving 7.4 % and 0.6 % absolute CIDEr scores on the MSVD and MSR-VTT datasets, respectively.
```

- **Introduction:**

```
Video captioning automatically describes video content using
natural language through machine learning. It has potential applications in various fields, such as disability assistance, human-computer
interaction, and automated commentary [1–4].
Existing deep learning-based video captioning approaches are predominantly based on encoder-decoder architectures [5–10] which are
trained using the video-text pairs. It usually results in generated captions
that highly resemble the training set [11] and lack distinctiveness [12].
Two kinds of approaches have been proposed to address the aforementioned issue. The first one is prior information-based methods, as shown
in Fig. 1(a), which input the extra the prior information, such as concepts [13–15], retrievals [16–18], and topics [19], into the decoder to
enhance the detail, diversity of descriptions, and distinctiveness to some
extent. However, the inaccurate prior information possibly influences
the quality of the generated captions. The second one is contrastivebased methods, [12,20], as shown in Fig. 1(b). These methods employ a
pre-trained reference model as guidance, it can improve the distinctiveness of the target model. Nonetheless, the performance is influenced
by the reference model : they fail to account for relationships among positive samples, and in real application scenarios, the reference model
is updated [12] multiple times to maintain its effectiveness, which incurs a high cost.
In fact, the reference model is implicitly derived from the training
data [12]. In other words, the guidance provided by the reference model
is dissolved within the training set itself, which inherently contains both
similar and different samples. Contrastive learning [21], commonly used
in self-supervised learning [22–24], is particularly well-suited for modeling such relationships, as it pulls positive sample pairs closer while
pushing negative pairs apart. Therefore, it is feasible to train a unified
framework that reduces intra-class distances among positive samples
while increasing inter-class distances among negative samples through
contrastive learning [19]. This allows the model to simultaneously capture both the distinctiveness and similarity inherent in the data at a
relatively low computational cost.
In this paper, we propose a simple yet effective Unified Hierarchical
Contrast Learning (UHCL) method for video captioning, as illustrated
in Fig. 1(c). During training, a video and its corresponding caption (anchor), positive sample (caption from the same video but different from
the anchor) and negative sample (caption from different videos) are input into triamese decoders. Subsequently, the relationships among the generated video captions are constrained using contrastive learning. To
further constrain the solution space, visual tokens are processed twice
through the adaptive token fusion module. The fused visual tokens are
then subsequently employed to compute contrastive loss hierarchically
at both the noun and verb levels. Notably, while triamese decoders are
used during training, only a single decoder is used during inference
to generate sentence, thereby avoiding additional computational overhead.
The contributions of this paper are summarized below:
• A novel, simple yet effective contrastive framework for video captioning, named Unified Hierarchical Contrastive Learning (UHCL), is
proposed. Leveraging triamese decoders and adaptive token fusion
modules, this framework enforces hierarchical contrastive learning
constraints without requiring additional models or training.
• The relationships between the generated video captions and those of
the anchor, positive samples, and negative samples are constrained
using hierarchical contrastive learning. This method applies to both
noun and verb levels through the adaptive token fusion module, enabling multi-granularity constraints during model training.
• Experiments demonstrate that the proposed method outperforms
state-of-the-art methods by 7.4 % and 0.6 % in absolute CIDEr scores
on the widely used MSVD and MSR-VTT datasets, respectively. Additionally, it surpasses baseline methods by 9.7 % and 4.4 % in absolute
CIDEr scores, respectively.
```

- **Related work:**

```
2.1. Video captioning
Recent encoder-decoder-based video captioning methods have
demonstrated significant performance improvements and can be mainly
categorized into two types. The first type is trained in an end-to-end
manner [9,15,25] without relying on pre-trained feature extractors for
visual feature extraction. Typically, the visual encoder is pre-trained on
large datasets [26,27] and then the entire model fine-tuned on video
captioning datasets. While effective, this end-to-end approach generally
involves a large number of parameters. The second type employs frozen
pre-trained feature extractors to obtain video features with some methods further extracting motion features [28–32], and the model is subsequently trained on video captioning datasets. Improvements in this
type include enhancing modality alignment capabilities [32–34], optimizing network structures, and employing advanced feature extractors
[26] to boost feature extraction capabilities [10]. A notable advancement in this approach is the extraction and integration of additional
prior information, such as concepts [13–15] retrievals [16–18], optical
flow [9], sentiment [35], object detection [28–30], and topics [19]. Although these methods effectively extract prior information from videos,
they could potentially mislead the model [14]. Furthermore, the training of these methods typically employs maximum likelihood estimation
(MLE), without accounting for the relationships between captions from
different videos or varying captions within the same video, leading to
a lack of diversity and distinctiveness [12]. Our approach falls into the
second category but differs in key aspects. By leveraging triamese decoders and contrastive learning, we account for multi-level relationships
between captions from different videos and varying captions within the
same video. Moreover, our method achieves remarkable performance
without relying on prior information, thereby eliminating the introduction of noise and additional inference overhead.

2.2. Contrastive learning
The core idea of contrastive learning [21] is to bring positive pairs
closer together while pushing negative pairs further apart [22]. This approach is inherently well-suited for modeling relationships between anchors, positive and negative samples/features, and has been widely applied in self-supervised learning [22–24,36–38], image-to-image translation [39], image dehazing [40] and image-language pre-training [26].
To alleviate the limitations of MLE, which often results in high resemblance [11] and a lack of distinctiveness [12] in model-generated captions, contrastive learning [12] integrates negative samples into the loss
calculation, effectively mitigating issues of indistinctiveness in image
captions. Building on this foundation, contrastive learning [20] has been
extended to video captioning, enhancing caption distinctiveness and diversity by mining the hardest negative samples. Despite its effectiveness,
these methods typically rely on a two-stage training process: first training a reference model and then training a target model to surpass the reference. Unlike the aforementioned methods [12,20] that depend on pretrained reference models, our approach achieves a unified framework
through triamese decoders, which dynamically learn the relationships
between mismatched and matched video-text pairs. Additionally, finegrained relationships within captions, such as those involving nouns and
verbs, are explicitly constrained. Notably, inference can be performed
with single-stage training, ensuring the model both straightforward and
efficient.
```

# [23-25][v] Visual Commonsense-Aware Representation Network for Video Captioning

- **Link:** https://arxiv.org/pdf/2211.09469

- **Published in:** IEEE Transactions on Neural Networks and Learning Systems ( Volume: 36, Issue: 1, January 2025)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.1 | 37.4 | 74.6 | 100.8 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 41.5 | 28.1 | 61.2 | 50.2 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 32.4 | 22.4 | 48.9 | 49.9 |

- **Abstract:**

```
Generating consecutive descriptions for videos, i.e.,
Video Captioning, requires taking full advantage of visual representation along with the generation process. Existing video
captioning methods focus on making an exploration of spatialtemporal representations and their relationships to produce
inferences. However, such methods only exploit the superficial
association contained in the video itself without considering
the intrinsic visual commonsense knowledge that existed in a
video dataset, which may hinder their capabilities of knowledge
cognitive to reason accurate descriptions. To address this problem, we propose a simple yet effective method, called Visual
Commonsense-aware Representation Network (VCRN), for video
captioning. Specifically, we construct a Video Dictionary, a plugand-play component, obtained by clustering all video features
from the total dataset into multiple clustered centers without
additional annotation. Each center implicitly represents a visual
commonsense concept in the video domain, which is utilized in
our proposed Visual Concept Selection (VCS) to obtain a videorelated concept feature. Next, a Conceptual Integration Generation (CIG) is proposed to enhance the caption generation. Extensive experiments on three publicly video captioning benchmarks:
MSVD, MSR-VTT, and VATEX, demonstrate that our method
reaches state-of-the-art performance, indicating the effectiveness
of our method. In addition, our approach is integrated into the
existing method of video question answering and improves this
performance, further showing the generalization of our method.
```

- **Introduction:**

```
With the widespread use of mobile phones and computers,
millions of videos are uploaded daily by users to sharing sites
such as TikTok, YouTube, and Netflix. Thus, a powerful video
captioning method is significantly essential to automatically
generate the appropriate descriptions for user-uploaded videos,
which can improve the user experience. Besides, there are
other broad application scenarios for video captioning, including visually impaired assistance [1], [2], online video search
[3], [4], human-computer interaction [5], [6], etc. Compared
with its twin “image captioning” [7], [8] only dealing with
static spatial information, video captioning tends to be more
challenging since it involves both consecutive spatial and
temporal representations.

The mainstream approaches for video captioning follow
the paradigm of an encoder-decoder framework, where the
encoder employs CNNs to analyze and extract useful visual
context features from the source video, and the decoder utilizes
RNNs to generate the caption sequentially. One effective
solution is to learn representative visual features. Toward this
goal, existing methods propose a series of attention mechanism
by learning the temporal relation between video frames [9],
[10], the spatial relations between objects in every single frame
[11], [12], or spatial-temporal relation using appearance and
motion representations [13]–[15].
Although the above methods have achieved remarkable
progress, they focus on a source video to exploit spatialtemporal relationships to generate caption via recurrent decoder, which still rely on learning the superficial association
contained in the video itself. As an external information, commonsense knowledge is considered a necessary complement to
the cross-modal task [16]–[18], which remains under-explored.
For instance, [11] designs teacher-recommended learning to
take full advantage of the successful external language model
(ELM) to integrate rich language knowledge into the captioning model, which only exploits commonsense knowledge in
text domain. However, the commonsense knowledge in the
video domain [19] is neglected.
Generally, the generated words in the descriptions may
occur in multiple video scenes with similar but not identical
context information. For instance, the basic model in Fig. 1,
which is based on the encoder-decoder framework, cannot correspond the information in the source video to the words
“play instruments” and “a huge crowd” accurately because
of insufficient visual details. In reality, when comprehending
videos, human may also associate the source video with
other videos with similar visual concepts for an analogy to
generate more accurate descriptions. Thus, this indicates that
video captioning should have the cognitive power of visual
commonsense knowledge.
In this paper, we design a novel method for video captioning, called Visual Commonsense-aware Representation
Network (VCRN). Since directly modeling the relationship
between the source video and other all videos will inevitably
increase the computational and time cost, we design a video
dictionary to summarize the co-occurrence commonsense
knowledge of all videos, so as to explore the association
between the source video and visual commonsense knowledge.
Our network VCRN comprises the following three major components: 1) Video Dictionary construction (VDC), which
aims to build the commonsense knowledge from a video
dataset. Specifically, we employ a K-means algorithm on the
video frame representations derived from all videos to yield
a video dictionary consisting of a set of cluster centers. And
each center is regarded as a visual concept representing one
type implicit commonsense knowledge. 2) Visual Concept
Selection (VCS), which is to acquire visual commonsense
knowledge related to the source video from the video dictionary. In practice, we adopt a concept-aware multi-head
attention to obtain a video-related concept feature by selecting
key concept information from the video dictionary guided
by the source video. 3) Conceptual Integration Generation
(CIG), which is designed to enhance the caption generation
by exploring the relationship between the source video feature
and the video-related concept feature. Such a module can
provide dynamical control for the propagation of the above
two types of features by a gate mechanism. Fig. 1 shows
that our model can successfully generate fine-grained words
“play instruments” and “a huge crowd” because our method
can capture various relevant visual information corresponding
to the source video from the video dictionary. To evaluate
our proposed method, we conduct extensive experiments and
analyses it on the three publicly video captioning benchmarks :
MSVD, MSR-VTT, and VATEX. And comprehensive ablation
experiments are carried out to prove the effectiveness of our
each component. Besides, to further improve the generalization
of our method, our method is successfully applied to video
question answering task. Finally, we qualitatively show that
our method can contribute to improved captions through case
studies.
To summarize, the contributions of this work lie in threefold:
• We propose a simple yet effective method, namely
a Visual Commonsense-aware Representation Network
(VCRN), to explore the effect of visual commonsense
information for video captioning, which improves the
model’s capability of knowledge cognitive.
• We design a video dictionary, a plug-and-play component,
to model visual commonsense and exploit the association
between the source video and commonsense via our proposed visual concept selection and conceptual integration
generation to yield a more accurate caption.
• The extensive experimental results demonstrate the benefits of introducing visual commonsense for the video
captioning task. The proposed method VCRN achieves
state-of-the-art performance on MSVD and VATEX and
competitive performance on MSR-VTT. Besides, our
approach brings performance gains on video question
answering task, further demonstrating the generalization
of our method.
```

- **Related work:**

```
A. Video Captioning
Video captioning as one of the mainstay in the multimodal domain, this task has received extensive interest and
made rapid development. With the advent of the encoderdecoder framework, recent researches mainly focus on the
sequence-learning based methods for generation process [12],
[20]–[24]. Technically, these methods employ an encoder to
refine the video representation from a group of fixed video
frame features, and then a language-based decoder integrates
textual descriptions with the refined video features to learn a
modality-aligned representation for caption generation. As one
of the precedents that adopt such encoder-decoder structure,
[25] generates captions by LSTM with mean pooled video
representation overall frame features. And [9] proposes a
temporal attention to dynamically select video frames based
on the current decode step. To further align the semantic information between video and language modalities and improve
the performance, extensive approaches with elaborate structure
[13], [24], [26]–[28] have been proposed. For instance, [13]
encodes a video into semantic groups by aligning frames
around the phrases of partially decoded caption and describes
the video by exploiting the semantic groups as information
units. [29] utilizes optical flow to guide the spatial attention,
which can capture the pattern of apparent motion between
consecutive video frames. To improve caption quality, [24]
proposes an alternative paradigm to decompose the captioning
procedure into two stages. More recently, there are some
methods [11], [12], [30] have drawn attention to object-level
information. [30], [31] adopt a bidirectional temporal graph
to capture fine-grained dynamic flow for salient objects in the
video. [12] performs visual reasoning over both space and
time domains then locate region over the video by a spatialtemporal attention.
Unlike these methods, our approach does not introduce
extra visual features or pre-trained end-to-end architectures,
but mines the underlying semantic knowledge hidden in the
datasets, which aims to provide high-level visual concepts for
the model reasoning.
B. Knowledge-based Learning
To further move towards cognitive understanding of models,
many knowledge-based approaches have been proposed [32]–
[35]. In general, most of the existing methods can be categorized into two types. The first one focuses on the structured
knowledge base (e.g., DBpedia [36] and WordNet [37]) to perform knowledge inference and assist model reasoning.
For instance, [38] applies a large-scale knowledge base as
visual concepts, i.e., ConceptNet [39], for explainable visual
question answering (VQA). [40] leverages structured concept
graph to improve the performance of image captioning. [34]
proposes multi-level commonsense knowledge-based learning
for visual commonsense reasoning. The other one focuses on
the unstructured knowledge base, which explicitly represents
knowledge from the linguistic corpus or vision modality.
Compared with the structured one, it regularly be acquired
through elaborate design such as pre-trained language (LMs)
or retrieval model. For instance, [35] hypothesizes that a
system that relies exclusively on text will allow LMs to better
leverage their implicit knowledge and then utilize it on visual
question answering task. [41] proposes a pluggable retriever to
retrieve sentences as prior hints into video captioning model.
Different from previous approaches that exploit consensus
knowledge from the external source, our method aims to
explore latent association in video set and mine intrinsic
commonsense knowledge between videos from inside.
C. Video Question Answering
Video question answering is another fundamental multimodal task, which aims to predict an accurate answer according to a video and a corresponding question. The benefit
to the success of deep learning, various techniques, e.g.,
attention mechanism [42]–[44], memory network [45], [46],
and graph neural network [47], [48], have been proposed to
build the relationship between vision and language to answer
questions. For instance, [42] proposes a temporal attention to
focus on the key information through questions as guidance.
[45] applies a co-memory network to learn the important
cues from both motion and appearance and obtain the multilevel contextual facts to infer the answer. [43] introduces
a Hierarchical Conditional Relation Network to construct
more sophisticated relations across video and question, which
obtains diverse modalities and contextual information. [47] proposes a Motion-Appearance Synergistic Network to actionoriented cross-modal joint representations between motion
and appearance by graph neural network. In this paper, our
proposed method is applied to the task of video question
answering to verify its effectiveness.
```

# [25-26][v] Dual-hierarchical knowledge distillation for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0031320325008532

- **Published in:** Pattern Recognition, Volume 171, Part A, March 2026, 112192

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 61.6 | 42.4 | 79.1 | 127 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 45.3 | 30.2 | 63.8 | 57.9 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 31.6 | 23.9 | 49.6 | 55.3 |

- **Abstract:**

```
Video captioning remains a challenging task due to the tradeoff between accuracy and computational efficiency.
We propose CapDistill, a dual hierarchical distillation framework that transfers semantic knowledge from a
powerful teacher model to a lightweight student model. CapDistill captures object-level and action-level semantics from captions and transfers multilevel knowledge including object features, action features, and word-level
predictions through a hierarchical strategy. To reduce the impact of noisy annotations, we introduce a caption
quality grading mechanism that assigns quality-based weights to training captions. Experiments on MSR-VTT
and MSVD demonstrate that CapDistill achieves state-of-the-art accuracy while significantly reducing inference
cost.
```

- **Introduction:**

```
Video captioning, which involves generating coherent natural language descriptions from video content, plays a crucial role in applications such as assistive technologies for the visually impaired [1],
content-based video retrieval [2], and autonomous systems [3]. Unlike
image captioning that deals with static content, video captioning must
capture temporal dynamics, object interactions, and scene transitions.
This requires advanced visual representations, effective temporal modeling, and precise alignment between visual and linguistic modalities.
Early methods evolved from template-based pipelines to encoderdecoder architectures using Convolutional Neural Networks (CNNs) and
Recurrent Neural Networks (RNNs), such as Long Short-Term Memory
networks (LSTMs) and Gated Recurrent Units (GRUs) [4]. The adoption
of attention mechanisms allowed models to dynamically attend to informative temporal segments, improving alignment between visual input and generated text [5]. More recently, transformer-based models
have replaced RNNs, leveraging multi-head self-attention for superior
temporal and contextual modeling [6]. Progress has also been driven
by multimodal fusion techniques (e.g., subtitles, audio, and Automatic
Speech Recognition (ASR)) [7], graph-based reasoning for modeling
object-action relationships [8,9], and reinforcement learning for metricaware optimization [10].
Despite this progress, three major challenges persist:
• Efficient temporal modeling: Modeling long-range dependencies
across frames is crucial for understanding complex video content but is often computationally expensive, limiting deployment in real-time
or resource-constrained environments.
• Annotation variability and quality: Although datasets like MSVD
[11] and MSR-VTT [12] provide diverse human-written captions,
not all annotations are equally informative. Some captions include
vague descriptions, subjective phrasing, or misaligned semantics
that weaken the supervision signal during training. For example, as
shown in Fig. 1, “a man is standing with road” introduces off-topic
content, while “a girl is jumping” may misrepresent a water-skiing
scene. While these captions are linguistically valid, they hinder consistency in model learning and semantic alignment.
• Semantic feature learning: Generating high-quality captions requires structured understanding of objects, actions, and spatiotemporal relationships. Efficiently learning such features in a compact
representation remains an open problem.
To address these challenges, we propose a dual-hierarchical distillation framework that incorporates a caption evaluation mechanism to
prioritize semantically informative annotations without discarding valid
linguistic diversity. We opted for knowledge distillation from the outset because it offers an effective strategy to leverage the strong representational capacity of large teacher models while enabling efficient
compression into smaller student models suitable for real-world deployment. In video captioning, this approach allows the model to capture fine-grained semantic and temporal dependencies without requiring prohibitively large architectures. It also facilitates more stable and
sample-efficient training compared to training compact models from scratch, which is especially important when annotations are noisy or
limited.
Unlike existing methods that rely on end-to-end training with
uniform annotation treatment, our approach introduces a two-stage distillation process. First, high-quality captions are used to guide hierarchical learning in the teacher model by distilling object-level and actionlevel semantics. Second, the learned multi-level knowledge (including
object features, action features, and word-level predictions) is transferred from the teacher to a lightweight student model. To the best of
our knowledge, CapDistill is the first framework to realize this dualhierarchical distillation paradigm for video captioning. The integration
of caption grading further enhances training robustness by assigning
adaptive weights to noisy or inconsistent annotations.
The main contributions of this work are:
• Efficient temporal modeling through distillation: We adopt a
teacher-student framework in which a high-capacity teacher captures rich temporal and contextual video features, which are distilled
into a compact student model to enable efficient and scalable inference.
• Noise-robust training via caption grading: A caption quality grading mechanism is introduced to assess and weight ground-truth captions based on their semantic relevance and consistency, reducing
the influence of noisy or ambiguous annotations during supervision.
• Dual-hierarchical distillation of semantic knowledge: Our
framework jointly distills object-level and action-level semantics,
first from captions to the teacher network, then from the teacher to
the student model. This hierarchical supervision enhances semantic
alignment between visual inputs and linguistic outputs.
In summary, CapDistill addresses key challenges in video captioning
through efficient modeling, noise-resilient supervision, and dualhierarchical knowledge transfer. Extensive experiments on MSVD, MSRVTT and VATEX [13] demonstrate that our method outperforms stateof-the-art approaches while maintaining scalability and generalization
to real-world scenarios.
```

- **Related work:**

```
2.1. Video captioning
Early video captioning methods used template-based approaches
that mapped detected subjects, verbs, and objects (SVO) into predefined
sentence structures. These were later replaced by deep learning-based
models, which demonstrated improved capabilities to capture complex
spatio-temporal semantics and generate context-aware descriptions.
Encoder-decoder architectures. Encoder-decoder frameworks have
played a central role in advancing video captioning. In these models,
Convolutional Neural Networks (CNNs) were commonly employed for
spatial feature extraction, while Recurrent Neural Networks (RNNs)
were initially used to model temporal dependencies. Later works
introduced attention mechanisms and graph-based reasoning modules
to improve temporal alignment and capture inter-object relations.
For instance, Res-F2F [14] fused multi-stream CNN features, and
VASTA [5] applied adaptive spatio-temporal attention for informative
frame selection. HTG+HMG [8] and KG-VCN [9] utilized graph structures to model complex visual semantics, while IVRC [15] proposed
a token shift module and Refineformer for fine-grained temporal
modeling. MAN [16] introduced a memory-augmented encoder and
decoder with separate visual and textual memories to capture implicit
external knowledge within the dataset. GSEN [17] and Track4Cap [18]
emphasized global semantics and object interactions to enhance
caption coherence. Collectively, these methods improved contextual
understanding and semantic fidelity in video captioning.
Syntax-aware models. Syntax-aware methods aimed to enhance linguistic quality by incorporating grammatical and semantic structure into
caption generation. SAAT [19] focused on verb-centric alignment to
better capture actions, while TTA [20] aligned object tags with descriptive tokens to improve visual-language grounding. RSFD [21] employed frequency-aware attention to emphasize rare but informative
words. SNCL [22] introduced hierarchical reasoning guided by semantic contrastive learning. MesNet [4] adopted a multi-layer GRU decoder, where the lower layers modeled syntax and the higher layers
encoded semantics. EmVidCap [23] further extended this line by incorporating an emotion-aware framework and dataset, combining factual and emotional streams to generate more expressive and engaging
captions.
Retrieval and copy mechanisms. Retrieval-based methods enhanced
caption diversity and fluency by incorporating external corpora. OpenBook [2] retrieved contextually similar captions and integrated selected phrases using a copy mechanism. CARE [24] improved concept detection through multimodal video-to-text retrieval and applied
global-local semantic guidance to generate informative and coherent
descriptions. These methods benefited from external knowledge, but
were highly dependent on the relevance and quality of the retrieved
content.
Non-autoregressive and action-aware models. ALSO-Net [25] enhanced
non-autoregressive video captioning by integrating high-level action
features and optimizing motion-related losses, improving verb accuracy
and coherence with lower complexity.
End-to-end transformer models. Transformer-based models unified spatial, temporal, and semantic modeling through self-attention. Models such as SwinBERT [6] and CoCap [26] replaced RNNs with
transformer blocks, enabling joint optimization of vision and language
features. Although these approaches achieved strong performance, they
incurred significant computational costs and reduced flexibility for modular adaptation.
2.2. Training techniques
Advanced training paradigms have been explored to improve generalization, efficiency, and metric alignment.
Reinforcement learning. Methods like DRPN [10] optimized sequencelevel captioning metrics (e.g., CIDEr [27], BLEU [28]) using reward signals. Although effective, such methods often introduced instability and
required carefully crafted reward functions.
Adversarial training. Adversarial learning frameworks for video captioning [29] employed a generator-discriminator setup, where the discriminator guided the generator to produce more natural and fluent captions. Although this approach improved linguistic quality, it introduced
additional training complexity and required careful balancing between
adversarial and language modeling objectives.
Contrastive learning. Contrastive methods such as EMCL [30] aligned
visual and textual representations in shared latent spaces by maximizing
similarity for paired inputs. While efficient, they sometimes sacrificed
fine-grained detail due to representation compression.
Knowledge distillation. Knowledge distillation aims to transfer knowledge from a high-capacity teacher model to a compact student model.
Methods such as STG-KD [31] and ORG-TRL [32] distilled temporal
or relational representations but were typically limited to single-level
features, which constrained the semantic diversity transferred to the
student.
```

# [25-26][?][MK-VC] Scene adaptive dynamic multi-modal knowledge for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0957417425044355

- **Published in:** Expert Systems with Applications, Volume 305, 5 April 2026, 130820

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.3 | 40.6 | 77.8 | 115.1 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 48.1 |31.7 |65.5 | 62.0|

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning has achieved remarkable advancements in recent years. However, the long-tailed word distribution in real scenarios often limits the performance. To address this issue, we first construct the Dynamic
Multi-modal Knowledge (DMK). It consists of a Static Concept Knowledge (SCK) and a Dynamic Context Knowledge (DCK). The former encapsulates commonsense knowledge and dataset-specific information, which is invariant for a specific dataset. And the latter captures video-related multi-modal knowledge, including visual features, attributes, and object text, which vary across different videos. Then we propose an effective multi-modal
knowledge-based (MK-VC) video captioning framework. It incorporates Context-aware Fine-grained Adaptive
Fusion (CaFAF) module and Attribute-guided Hardest Negative Contrastive Alignment (AHNCA) module. The
CaFAF module dynamically adjusts the importance of the DMK with fine granularity by leveraging contextual
scene features. In this way, CaFAF emphasizes important knowledge, such as highly relevant or low-frequency
words, thereby alleviating the long-tailed word distribution problem to some extent. The AHNCA module aligns
different modalities by utilizing object attributes. And it further enhances the alignment using the hardest negative samples. The experimental results on two popular datasets, MSVD and MSR-VTT, demonstrate that the
proposed MK-VC achieves the SOTA performance across multiple metrics, including ROUGE-L, METEOR, and
CIDEr. In particular, the CIDEr score of our method is higher than the best methods by over 2.1 % on MSVD and
2.3 % on MSR-VTT, respectively.
```

- **Introduction:**

```
Video captioning (VC) utilizes machine learning techniques to generate natural language descriptions of video content. It has significant
potential for applications across various domains, such as assistive technology for individuals with disabilities (Guadarrama et al., 2013; Venugopalan et al., 2015), human-computer interaction (Guo et al., 2019;
Rohrbach et al., 2013), and automated commentary (Xi et al., 2025b,c).
Existing learning-based approaches predominantly rely on encoderdecoder architectures (Ballas et al., 2015; Pan et al., 2016; Wang et al.,
2018; Yao et al., 2015; Yu et al., 2016). Notably, the incorporation of
Transformer (Dosovitskiy, 2020) and CLIP (Radford et al., 2021) has
led to the development of highly efficient models (Korbar et al., 2020;
Luo et al., 2020; Shen et al., 2023; Zhu & Yang, 2020), yielding impressive results. However, these methods struggle to address the longtailed word distribution issue in video captioning datasets, as shown
in Fig. 1(a). A long-tailed distribution (Zhang et al., 2023) is typically
observed in training samples, where a small subset of items contains a
vast majority of the sample points, while the remaining items are associated with only a few samples. In such datasets, the items with low appearance frequency may influence the representation learning. Gu et al.
(2023) alleviates the long-tailed problem to some extent by introducing
structured knowledge within the text modality, but it possibly generates
inaccurate concepts because the text is independent of the visual modality. As a multi-modal task, video captioning should learn not only the
relationships within the text modality but also mappings between the visual and textual modalities. Therefore, for real-world videos, which are
inherently diverse and complex, the key to addressing this challenge
lies in enabling the model to accurately learn cross-modal mappings for
long-tailed items.
Multi-modal structured knowledge (Chen et al., 2024) serve as
bridges between various modalities, such as text and vision, by explicitly establishing relationships among them. This information can
be extracted from large-scale knowledge bases (Liu & Singh, 2004;
Speer et al., 2017) and datasets, and is not constrained by the limited samples in the training set. For example, when a rare items appears in a video, the relevant information from knowledge (such as
the object’s appearance, attributes, or action type) can help the model
fill in the gaps, enhancing its understanding and prediction capabilities for uncommon or unseen objects (Gu et al., 2023). This, in turn, mitigates the impact of long-tailed word distribution. Therefore,
by leveraging the prior knowledge and inter-modal relationships provided by multi-modal knowledge, captioning models can be equipped
with cross-modal mapping information. This helps address the longtailed word distribution problem and assist in generating more accurate
captions.
In this paper, we propose an effective multi-modal knowledge (MKVC) enhanced framework for video captioning. We first construct a Dynamic Multi-modal Knowledge (DMK) and use pre-trained CLIP to retrieve video-relevant knowledge. We then leverage adaptive fusion and
cross-modal alignment to fully exploit DMK and mitigate the impact of
long-tailed word distribution. Specifically, the MK-VC framework consists of two key modules: the Context-aware Fine-grained Adaptive Fusion (CaFAF) module and the Attribute-guided Hardest Negative Contrastive Alignment (AHNCA) module. As shown in Fig. 1(b) the DMK
exhibits a long-tailed distribution; consequently, the importance of different nodes, edges, and modalities varies in caption generation. Furthermore, although a pre-trained CLIP model retrieves scene-relevant
knowledge, noise may still be present. Therefore, we propose CaFAF
to mitigate interference from irrelevant information in the retrieved
knowledge and facilitate diverse multi-modal fusion of input knowledge
triples. Guided by contextual scene information, CaFAF adaptively adjusts the weights of nodes and relationships within the knowledge. To
achieve fine-grained cross-modal alignment, we propose AHNCA, which
additionally incorporates object attributes to align textual entities with
their visual representations using margin-based contrastive learning. In
addition, to improve alignment performance, AHNCA increases training
difficulty by selecting the most challenging negative sample pairs based
on similarity.
Our contributions can be summarized as three-fold:
• We introduce the MK-VC framework, which leverages multi-modal
knowledge to alleviate the long-tailed distribution in video captioning. To realize this, we develop a Dynamic Multi-modal Knowledge
(DMK) module and propose Attribute-guided Hardest Negative Contrastive Alignment (AHNCA) for fine-grained cross-modal alignment
within DMK.
• We propose the Context-aware Fine-grained Adaptive Fusion
(CaFAF) module, which generates adaptive weights to fine-tune the
importance of nodes and relationships within the input knowledge,
effectively reducing noise interference.
• Experiments show that the proposed method outperforms state-ofthe-art approaches by 2.1 % and 2.3 % in absolute CIDEr scores on
the widely used MSVD and MSR-VTT datasets, respectively.
```

- **Related work:**

```
2.1. Video captioning
Recent encoder-decoder-based VC methods have achieved notable
performance improvements, and can be broadly categorized into two
types. The first type is trained end-to-end (Lin et al., 2022; Shen et al.,
2023; Wu et al., 2023), without relying on pre-trained feature extractors for visual feature extraction. In this approach, the visual encoder is
typically pre-trained on large datasets (Liu et al., 2022; Radford et al.,
2021), and the entire model is then fine-tuned on VC datasets. While
effective, this end-to-end method tends to involve a large number of parameters. The second type uses frozen pre-trained feature extractors to
obtain video features, with some methods further extracting motion features (Chen et al., 2019; Li et al., 2022; Pan et al., 2020; Tu et al., 2021;
Zhang et al., 2020). Improvements in this category include enhancing
modality alignment (Shi et al., 2023; Tu et al., 2021; Zhang & Peng,
2018), optimizing network architectures, and employing advanced feature extractors (Radford et al., 2021) to strengthen feature extraction
(Tang et al., 2021). A significant advancement in this approach is the
integration of additional prior information, such as concept retrievals
(Wu et al., 2023; Yang et al., 2023, 2022), optical flow (Shen et al.,
2023), sentiment analysis (Song et al., 2024), object detection (Li et al.,
2022; Pan et al., 2020; Zhang et al., 2020), topics (Zeng et al., 2024),
and knowledge graphs (Gu et al., 2023; Sun et al., 2025).
Additionally, some methods (Shen et al., 2023; Song et al., 2024;
Sun et al., 2025; Tang et al., 2021; Yang et al., 2023) have achieved
excellent results by using the advanced feature extractor CLIP (Radford et al., 2021). For example, Cocap (Shen et al., 2023) trains the
model in an end-to-end manner to improve both speed and accuracy.
CARE (Yang et al., 2023) and VEIN (Song et al., 2024) detect concept
and emotion priors in videos, respectively, improving the accuracy of
key words detection. MGTR-MISS (Zhang et al., 2025) retrieves relevant
sentences from corpus, performs multi-modal interaction, and generates semantically rich captions through semantic supervision. DSSM-KG (Sun
et al., 2025) leverages Mamba (Gu et al., 2021; Lianghui et al., 2024)
and a commonsense knowledge graph to enhance the model’s spatiotemporal joint modeling and video understanding capabilities.
While these methods effectively extract internal prior information
or external unimodal textual commonsense knowledge, they struggle
to provide accurate and efficient cross-modal mappings or to establish
relationships between internal and external information. Consequently,
they may only partially address the long-tailed problem (Gu et al., 2023)
or mitigate the impact of noise (Yang et al., 2023) on performance. In
contrast, we construct a multi-modal knowledge to explicitly establish
cross-modal mappings among different modalities, thereby supplying
the model with rich prior knowledge and robust cross-modal relationships to mitigate the challenges posed by the long-tailed distribution.
2.2. Exploiting knowledge for captioning
The knowledge utilization in captioning models can be broadly classified into two categories: one involves mining internal knowledge from
the dataset, and the other leverages external knowledge.
For mining internal knowledge, the focus is currently on extracting
visual (Hou et al., 2020; Zhong et al., 2024) or textual knowledge (Wu
et al., 2022; Zhong et al., 2022). For instance, visual modality knowledge
is used in BTKG (Zhong et al., 2024) to predict relationships between
objects in videos via TransE (Bordes et al., 2013), with the encoded
relationships fed into the decoder. JointCR (Hou et al., 2020) generates a semantic graph using prior knowledge and object detection, with
the semantic graph containing contextual and commonsense knowledge.
In contrast, methods that leverage textual modality knowledge include
KAVC (Wu et al., 2022) and KGvideo (Zhong et al., 2022). Specifically,
KAVC (Wu et al., 2022) predicts actions between objects and transitive deep-level relationships between objects, and then employs a GCNbased approach to assist in video caption generation. KGvideo (Zhong
et al., 2022) utilizes a knowledge graph to predict relationships between
objects. These methods rely solely on visual or textual modality knowledge and do not incorporate external commonsense knowledge.
Early works in external knowledge-based captioning can be traced
back to the use of ConceptNet (Liu & Singh, 2004; Speer et al., 2017),
which provides commonsense knowledge to support model training
(Zhou et al., 2019). Similar efforts have employed more modern architectures to leverage the power of structure knowledge (Gu et al., 2023;
Santiesteban et al., 2024; Sun et al., 2025; Zhang et al., 2021a). Text-KG
(Gu et al., 2023) retrieves video-relevant commonsense knowledge and
feeds it into a dual-stream network. This helps alleviate the long-tailed
word distribution problem. DSSM-KG (Sun et al., 2025) addresses the
issues of spatio-temporal joint modeling and insufficient common sense
knowledge by improving model performance through a hybrid architecture and knowledge injection. Additionally, structure knowledge are
often utilized for entity-aware image captioning, as they provide rich
additional knowledge. For instance, commonsense knowledge can be
divided into relevant knowledge and explanatory knowledge (Xu et al.,
2024), which are then encoded to provide entity distribution and commonsense distribution for the final caption, thereby enhancing the accuracy and quality of entity-aware captioning. The challenge of identity
awareness often arises from the long-tailed distribution of identity information. Due to the difficulty of single-modality knowledge in providing cross-modal mapping relations, multi-modal knowledge enhanced
entity-aware image captioning (Zhao & Wu, 2023) offers a novel solution to address the long-tailed distribution problem.
However, methods that solely mine internal knowledge (Wu et al.,
2022; Zhong et al., 2024) or rely only on visual (Hou et al., 2020)
or textual knowledge (Gu et al., 2023; Sun et al., 2025; Zhong et al.,
2022) are suboptimal for video captioning. They cannot provide explicit
cross-modal mappings, and their coarse use of knowledge may introduce
irrelevant noise. Furthermore, the multi-modal knowledge employed
in entity-aware image captioning (Zhao & Wu, 2023) cannot adapt to
the complex and dynamic nature of video content. Therefore, it does
not directly apply to video captioning. In this paper, we construct a
multi-modal knowledge DMK, which contains rich internal knowledge
as well as external commonsense knowledge, and provides the model
with cross-modal mapping relationships. Additionally, we develop adaptive fusion to reduce the impact of knowledge noise, and use alignment
mechanisms to enhance the model’s learning of cross-modal mappings.
```

# [26-26][v][QPDC] Ask and focus more: Question-prompt uncertainty allocation for dual-controllable video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0031320326000683

- **Published in:** Pattern Recognition, Volume 175, July 2026, 113105

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.4 | 38.4 | 74.8 | 105.2 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 42.4 | 28.9 | 61.9 | 51.8 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 36.8 | 24.9 | 52.0 | 61.3 |

- **Abstract:**

```
Video captioning aims to generate natural language descriptions from video content via hierarchical architectures
that capture key visual elements. Although entities, predicates, and syntactic structures are critical for coherent
descriptions, existing controllable methods often struggle to dynamically emphasize salient information due to
reliance on predefined entity lists and static fusion strategies. Moreover, uncertainty in aligning multimodal features (e.g., visual actions and textual semantics) further weakens model robustness. To address these challenges,
we propose QPDC, a Dual-Controllable video captioning framework that introduces a Question-Prompt mechanism to adaptively guide caption focus. Unlike traditional entity-driven approaches, QPDC adopts a unified
question-prompt strategy to maintain semantic consistency while enabling flexible attention shifts. Specifically,
the Question-Prompt (Q-Prompt) module dynamically steers the model’s focus according to query intent, emphasizing salient content without predefined constraints. The Selective Fusion Module (SFM) hierarchically integrates Q-Prompt guidance with local action features and global sentence context to support multi-granularity
control. In addition, a Dirichlet-based strategy is employed to model uncertainty in multimodal weight allocation, improving adaptability under varying data conditions. Extensive experiments on MSVD and MSR-VTT show
that QPDC achieves competitive performance among controllable video captioning methods and produces more
diverse and user-preferred descriptions.
```

- **Introduction:**

```
Video captioning [1] aims to automatically generate natural language descriptions from videos and has been widely applied to assist
visually impaired individuals, enhance human-computer interaction,
and support video retrieval. However, without explicit contextual constraints, conventional video captioning models tend to produce relatively fixed descriptions driven primarily by dominant visual content.
Despite recent progress in large language models, video captioning remains challenging, as videos often contain complex and diverse information, of which only a small subset (e.g., two or three key elements) is
relevant for concise and informative descriptions.
Controllable video captioning (CVC) extends this paradigm by incorporating explicit control conditions (e.g., questions or instructions) to
generate customized and diverse descriptions that satisfy user-specified
intents. In recent years, CVC has attracted increasing attention as a
promising direction toward flexible and user-centered video understanding. A core challenge in CVC lies in detecting and integrating focused
actions and semantics. Existing approaches can be broadly categorized
into two classes. The first class relies on graph-based encoders to model spatial and temporal relations among objects. For example, Chen
et al. [2] design a spatiotemporal graph framework with recurrent region attention, cross-frame message passing, and a temporal graph decoder to capture high-order relations without external detectors. However, as observed in [3], such methods often generate generic descriptions and lack fine-grained controllability to emphasize specific points
of interest or narrative patterns. To mitigate this issue, predicate-based
feature selection and gated fusion mechanisms have been explored to
support diverse generation under different control signals.
The second class focuses on reducing the semantic gap between video
representations and linguistic captions prior to generation. GSEN [4],
for instance, aligns global visual content with caption semantics, while
IVRC [5] encodes object-, action-, and event-level features at multiple
granularities under caption supervision. Although these methods investigate both local word alignment and global sentence alignment, they
rarely address cross-granularity alignment between fine-grained actions
and coarse-grained contextual semantics. Consequently, representative
video embeddings may still overlook detailed actions or broader scenelevel cues. We argue that question prompts provide an effective means
to bridge this gap by explicitly guiding cross-granularity alignment. Notably, question-answer pairs lie at the core of visual question answering (VQA), which integrates visual understanding and language reasoning. While video captioning and VQA share common modeling components [6], we leverage questions not as queries to be answered, but as
prompts that guide the model to focus on relevant visual information.
Prior studies have shown that question-driven supervision can benefit text generation. However, existing video captioning methods often
lack sufficient textual guidance during visual feature learning, and the
relation between questions and visual entities remains underexplored.
For example, Hu et al. [7] introduce a question-prompt encoder to select relevant visual features from optical character recognition (OCR)
regions and design targeted questions to ground caption generation.
Nevertheless, the strong dependence on OCR features may induce shortcut learning. In contrast, our approach does not rely on pre-extracted
OCR cues and instead focuses on generating fairness-aware questions to
guide visual attention more generally. As illustrated in Fig. 1, captions
produced by prior methods may fail to identify salient attributes (e.g., a
“tall man”) or key actions (e.g., in a “pool”). From this perspective, video
captioning can be viewed as answering an implicit question: “What does
this video describe?” A high-quality caption should therefore respond to
a range of related queries about the video, even in the absence of explicit
visual input.
In this work, we propose QPDC, an encoder-decoder-based dualcontrollable video captioning framework that leverages BLIP-driven
questions, a Selective Fusion Module (SFM), and comprehensive focus
modeling. QPDC generates captions by responding to video-relevant
questions, thereby reducing omissions and semantic errors. Specifically, we generate visual questions from both videos and human-written
captions and integrate them into a dedicated question-prompt (QPrompt) captioning model. To improve robustness under alignment uncertainty, we introduce branch-adjustable supervision that allows different branches (e.g., entity and action branches) to dynamically adjust
their contributions based on input characteristics. The encoder jointly
processes video content and questions, followed by an alignment module for feature fusion, while the decoder generates captions conditioned
on the integrated representations. This design enables dynamic emphasis on specific actions or global semantics depending on the question
context. Moreover, the fusion module supports sample-adaptive multimodal integration without additional computational overhead and performs evidential-level fusion via a learnable Dirichlet distribution.
The main contributions of this paper are summarized threefold:
• We propose QPDC, a dual-controllable video captioning framework
that integrates question-compatible and branch-adjustable supervision through a unified sentence-level modeling paradigm.
• We design a Question Prompt (Q-Prompt) mechanism and a Selective
Fusion Module (SFM) to enhance controllability. Q-Prompt generates
BLIP-driven questions with global and local semantics, while SFM
adaptively selects appropriate branches to support customized and
diverse caption generation.
• We introduce a sample-adaptive multimodal integration strategy
based on evidential modeling, enabling explicit uncertainty characterization and improving robustness under noisy or degraded conditions.
```

- **Related work:**

```
2.1. Video captioning
Early video captioning methods were largely template-based, detecting visual concepts such as objects, relations, and attributes and composing captions using predefined grammatical structures (e.g., subjectverb-object). With the emergence of encoder-decoder frameworks,
MesNet [8] introduced a memory-sharing mechanism to strengthen recurrent connections, while MAN [9] enhanced encoders with learnable visual memory vectors and incorporated external linguistic cues
in the decoder. Subsequently, STAT [10] leveraged spatio-temporal attention to extract salient frames, and the 𝐼
2 Transformer [11] explored
intra- and cross-modal semantic correlations to construct unified multimodal representations.
The adoption of vision transformers [12] has further accelerated the
transition toward transformer-based architectures. DAST [13] employed
a depth-aware sparse transformer encoder to emphasize geometric relations, while VideoBLIP [14] adopted a transformer encoder-decoder
pipeline built upon CNN features. However, many such approaches rely
on frozen feature extractors or offline object detectors, which may introduce noise and limit end-to-end optimization. To address this issue,
SwinBERT [15] proposed a fully transformer-based architecture with
Video Swin Transformer encoders and a BERT-based [16] decoder.
With advances in multimodal learning [17], video captioning has
increasingly emphasized cross-modal alignment. Li et al. [18] proposed
a dual-branch MICM framework based on image-to-text and image-tovideo alignment, while Fu et al. [19] introduced CLIP-AGIQA to mitigate
domain gaps through multimodal prompt learning and vision-language
consistency. Zero-shot video captioning has also attracted attention: Ma
et al. [20] proposed a retrieval-enhanced test-time adaptation framework to improve robustness in open-domain settings. Other works incorporate textual supervision during visual encoding, jointly modeling
object-, action-, and context-level features [6,21] or aligning video representations with linguistic semantics across multiple granularities [22].
As effective fusion relies heavily on alignment quality [19,23,24], CLIPbased methods [19] serve as representative solutions in this direction.
Despite these advances, question-prompt video captioning [7],
which explicitly leverages questions to guide fine-grained and global
semantic interactions, remains relatively underexplored. In contrast to
alignment-only strategies, our work integrates question prompts with
multi-granularity visual cues to support more controllable caption generation.
2.2. Question answering for visual captioning
Question answering has been shown to be beneficial for text generation. Shen et al. [25] reformulated captioning as a question-answering
task to enhance semantic coverage. With the rapid development of large
language models, recent studies [26] explore collaborative question answering between ChatGPT-like models and vision encoders for visually grounded caption generation. PromptCap [27], for example, employs GPT-3 to generate targeted questions for images and rewrites
generic captions based on question-answer contexts using VQA data.
Unlike these approaches, which rely on predefined or customized
question-answer pairs, our method constructs fairness-aware questions
and adopts a unified question-prompt structure to guide caption generation consistently across videos.
2.3. Controllable visual captioning
Our work is closely related to controllable captioning methods that
incorporate explicit control signals [28]. Liu et al. [29] proposed O2NA,
which identifies key objects and their attributes to generate drafts that
are refined into fluent captions using contextual cues. Yang et al. [30]
introduced masked scene graphs and masked autoregressive decoding to
jointly model semantic and syntactic structures. Zhang et al. [31] incorporated part-of-speech tags as auxiliary inputs. Related efforts further
explore controllability through subject-oriented encoding [32], POSbased visual switches [33], example-driven syntactic control [34], and
two-stage style learning frameworks [35]. Other approaches leverage
mouse traces, scene graphs, or length-level embeddings to guide caption generation.
Compared with these methods, we develop a fully attentive framework that supports dual controllability through question prompts and
adaptive fusion, while maintaining explicit grounding between visual
content and textual semantics. Although related work [36] employs visual regions as control signals, it requires additional semantic annotations and introduces extra complexity, reflecting a different controllability setting from ours.
```

# [24-24][v][IVRC] Rethink video retrieval representation for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0031320324004953

- **Published in:** Pattern Recognition, Volume 156, December 2024, 110744

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 58.8 | 40.3 | 77.4 | 116.0 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 43.7 | 30.2 | 63.0 | 57.1 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 32.8 | 24.0 | 50.3 | 57.4 |

- **Abstract:**

```
Video captioning, a challenging task targeting the automatic generation of accurate and comprehensive
descriptions based on video content, has witnessed substantial success recently driven by bridging video
representations and textual semantics. Inspired by the nature of the video retrieval task, which learns visual
features strongly related to text queries, we propose to take advantage of visual representation learning from
the video retrieval framework to tackle video captioning tasks and construct adequate multi-grained crossmodal matching while extracting visual features. However, a simple direct application of recent video retrieval
models fails to capture sufficient temporal details and the rich visual features of local patch tokens of video
frames lack semantic information essential for captioning tasks. These deficiencies are primarily due to these
models lack fine-grained interactions between video frames and offer only weak textual supervision over frame
patch tokens. To increase the attention on temporal details, we propose a learnable token shift module, which
flexibly captures subtle movements in local regions across the temporal sequence. Furthermore, we devise a
Refineformer, which learns to integrate local video patch tokens strongly related to desired captions via a
cross-attention mechanism. Extensive experiments on MSVD, MSR-VTT and VATEX demonstrate the favorable
performance of our method.
```

- **Introduction:**

```
Video captioning is a focal point of research at the intersection of
computer vision and natural language processing. It aims to generate
precise descriptions reflecting video content. There have been numerous studies [1–3] in recent years on how to model the spatial–temporal
video representation. Despite progress, how to learn effective video representations facilitating the generation of accurate and information-rich
captions still remains a challenging task.
Existing works can be broadly categorized into two lines of research.
One line focuses on designing powerful video encoders to learn representative video features. In these works, pre-trained 2D/3D CNNs,
such as ResNet [4], Faster R-CNN [5] and I3D [6], are utilized to
extract initial visual features, which are further processed with Graph
Convolutional Networks (GCNs) [7] or transformer [8] architectures.
ORG-TRL [1] employs GCNs to construct relational graphs to encapsulate the spatial and temporal interactions among video objects. In the
wake of the transformer demonstrating superior performance in many
computer vision tasks, some studies have further encoded basic visual
features from CNN using transformer architecture. VideoBERT [9] uses
BERT [10] architecture to process the pre-computed CNN features. By
contrast, SwinBERT [3] directly learn visual features from raw video frames so as to perform end-to-end optimization and reduce the additional noise introduced from off-the-shelf models. Despite promising
results, most of these methods lack intermediate supervision for visual
features (i.e., they are supervised after words are predicted.)
The other line focuses on directly narrowing the semantic gap
between video representation and linguistic captions before generating
descriptions [11–14]. For example, GSEN [13] learns to align the global
visual content to semantics of a whole caption and achieves promising
results under different evaluation metrics. A textual-temporal attention
model is devised in TTA [14] to precisely align the key frames with
target words on a fine-grained level. HRNAT [11] and HMN [12]
encode objects, actions, and event features from local to global in a
video under the supervision of corresponding portions of the caption.
Though these approaches investigate the correspondence between local
visual features and words, as well as global visual feature and sentence,
cross-grained alignment (i.e., fine-grained features v.s. coarse-grained
feature) is rarely explored.
In summary, the existing video captioning methods do not introduce
sufficient textual supervision when extracting visual features. More
recently, several video retrieval models [15–17] demonstrate the strong
power of building a close relationship between text and video content via fine, coarse and cross-grained visual-textual matching. Considering
the excellent cross-modal alignment capability of video-text retrieval
models, we introduce this paradigm into the video captioning task to
produce well text-relevant visual features in the encoder part, thereby
facilitating the text decoder in generating captions. However, since
video encoders in these methods only extract and aggregate framelevel representation, a direct application of video encoders of retrieval
models to tackle video captioning tasks, such as [18], which uses a
mean pooling layer to aggregate frame-level features outputted from
the Vision Transformer (ViT) [19] adopted from [15], is void of finegrained interaction between adjacent frames and lacks the capacity
to model temporal details. Although TS2-Net [16] proposes to shift
several manually specified frame patches back and forth across adjacent
frames to capture subtle variations, such a handcrafted scheme cannot
adaptively reflect various needs of textual counterparts corresponding
to different videos. We show an example in Fig. 1, where the video
is about a man swinging an axe from above to chop a log on the
ground. Fig. 1(b) presents the result of patch token shift of TS2-Net,
where the human torso’s overall shape and body movement remain
virtually unchanged with the axe attached on the wood, failing to
introduce effective variations between adjacent frames. As a result,
the crucial action detail: the ‘‘swinging’’ motion from top to bottom
is not generated. Additionally, given that general retrieval frameworks
employ the ViT architecture as the visual encoder, only the [CLS]
token features, serving as concise representations of video frames, are
involved in alignment with text. However, frame patch tokens, which
contain specific and rich visual information, are only subject to weak
textual supervision. Consequently, local patch tokens of video frames
inputted into the caption decoder lack semantic information.
To effectively take advantage of video retrieval models, we propose an improved architecture based on video retrieval, named as
Improved Video Retrieval for Captioning (IVRC), which addresses the
above-mentioned issues. Our model consists of a text encoder, a video
encoder, a Refineformer (RF) for visual feature refinement, and a caption decoder for description generation. The two encoders are adopted
from TS2-Net except its token shift module. For retrieval-related training, we conduct multi-grained alignment, including fine, coarse, and
cross-grained, over two encoders to learn video features with rich
textual semantics. Moreover, we propose a novel Learnable Token
Shift (LTS) module inserted in the video encoder to capture subtle
movements in local temporal intervals. Instead of handcrafted shifting,
we utilize a neural network to learn the importance of each patch so as
to select patches that are worth moving with respect to target captions.
As shown in Fig. 1(c), our method is capable of introducing spatial information from the previous frame that significantly differs from the
current one, which results in effective fine-grained interaction. The ten
shifted patches can be categorized into two parts denoted by ⃝1 and
⃝2 . Area identified by ⃝1 reflects the earlier frame’s high-raised arm
posture, poised for axe descent, whereas region ⃝2 reveals the log’s
immediate vicinity as clear of objects in the preceding frame. With our
LTS, the core action of ‘‘swinging’’ from top to bottom is thus successfully generated by the model. Besides, we propose a Refineformer to
enhance the semantic information within the patch tokens fed into the
caption decoder. It utilizes the [CLS] tokens as query to learn which
local video patches are strongly related to the desired description to
provide additional well text-relevant spatial information for the caption
decoder.
To summarize, our contributions lie in four-fold:
• We propose a retrieval-based multi-task model for video captioning. In extracting visual features, we introduce textual supervision
via multi-grained alignment learning, which benefits subsequent
processes of generating video descriptions.
• We design a learnable token shift module to enhance fine-grained
inter-frame information interaction and better capture movements between adjacent frames.
• We design a simple yet effective Refineformer which integrates
local video patches strongly related to the desired description
to provide additional well text-related spatial information for
caption decoder.
• Our method achieves favorable performance on MSVD [20], MSRVTT [21] and VATEX [22] benchmarks without leveraging other
modalities such as speech transcripts or audio information. Thorough ablation studies demonstrate the merits of proposed modules.
```

- **Related work:**

```
In this section, we briefly review the most related work on video
captioning task. We also discuss works on video-text retrieval, among
which certain methods based on CLIP [23] inspire us in terms of
enhancing the semantic information contained within video features.
2.1. Video captioning
Many approaches have been proposed for video captioning. Early
studies primarily utilize template-based methods, and a conventional
procedure is to first utilize various classification tools to detect a set
of visual concepts, such as objects, relationships, and attributes. Then,
these concepts are formed as a caption based on the basic linguistic grammar (i.e., subjects-verbs-objects). Krishnamoorthy et al. [24]
propose to first generate words for objects and actions, and then fit predicted words into pre-defined sentence templates to generate descriptions. This kind of methods struggles to generate descriptions with flexible syntactic structures due to the limitation of pre-defined templates.
With the rise of CNN and RNN, numerous sequence learning methods
adopt the encoder–decoder architecture, which enables flexible generation of captions. Some works employ various CNN models to construct
powerful visual encoders and utilize RNN to serve as the text decoder.
The caption decoder in GL-RG [2] is based on LSTM [25], while the
global-local video encoder including ResNeXt [26] and Res3D [27]
exploits extensive visual representations from different video ranges to
improve the final linguistic expression. MesNet [28] utilizes a memory
sharing structure to strengthen the connections between RNN layers
and makes the deep network easier to train. MAN [29] and VCRN [30]
augment the video encoder using different approaches. MAN employs
learnable visual memory vectors, while VCRN utilizes a video dictionary containing commonsense knowledge within the dataset. Additionally, MAN’s decoder captures external language clues of descriptions.
Some methods [1,31] also employ GCNs to further process visual
features from the encoder. In recent years, with the development of
vision transformer, an increasing number of studies use transformerbased models as the encoder and decoder. DAST [32] introduces a
depth-aware sparse transformer in the video encoder to focus on the
geometrical relationships of instances in videos, achieving remarkable
results. VideoBERT [9] employs the transformer encoder to process
the initial CNN visual features and utilizes transformer decoder to
generate captions. Since most of these methods require the use of
frozen feature extractors and object detectors, they encounter issues
with interruptions in the gradient flow and noisy information from
offline features. To facilitate end-to-end optimization, SwinBERT [3]
employs a purely transformer-based architecture to directly encode
the input video frames. It adopts the Video Swin Transformer [33]
as the video encoder and utilizes a sparse attention mask to address
the redundancy among video tokens. The caption decoder is based on
BERT [10] architecture.
Video captioning, as a multi-modal task, greatly emphasizes the
interaction and integration of visual and textual information. Consequently, some existing works not only design robust visual encoders
like many methods mentioned above but also introduce textual information as supervision during visual feature extraction. For example,
Luo et al. [13] learn to align the global representation of a video to
the embedding of a whole sentence and achieve superior performance.
Tu et al. [14] incorporate a textual-temporal attention model into
the decoder to build the exact alignment between target words and
corresponding frames on a fine-grained level. Ye et al. [12] and Gao
et al. [11] encode objects, actions, and context feature from local to
global in a video under the supervision of corresponding portions of
the description. However, cross-grained alignment (i.e., fine-grained
features v.s. coarse-grained feature) is scarcely explored. Different from
the existing efforts, we introduce multi-grained video-text alignment to
derive strongly text-related video features to facilitate the generation
of captions.
2.2. Video-text retrieval
Traditional video-text retrieval methods tend to transfer knowledge
from ‘‘expert’’ models and capture intra-modal and cross-modal interaction based on offline extracted features. However, the performance
of these methods is limited since they cannot perform end-to-end optimization. Recently, some researchers utilize the end-to-end paradigm
for video-text retrieval. One typical strategy [34] involves initially conducting large-scale text-video pre-training, followed by transferring the
model to downstream retrieval tasks. With the emergence of pre-trained
Vision-Language Models (VLMs), another line is to directly expand
the pre-trained VLM to the video-text retrieval task. For instance,
some recent works [15–17,35,36] focus on transferring knowledge from
CLIP [23]. CLIP4Clip [15] firstly transferred the knowledge of largescale image-text pre-training to the task of video-text retrieval with
fine-tuning. TS2-Net [16] proposes handcrafted token shift to obain a
better video representation temporally, while we propose a learnable
token shift module to enable more flexible and accurate fine-grained
interaction between adjacent frames.
```

# [24-25][v] CroCaps: A CLIP-assisted cross-domain video captioner

- **Link:** https://www.sciencedirect.com/science/article/pii/S0957417424031634

- **Published in:** Expert Systems with Applications, Volume 268, 5 April 2025, 126296

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 58.2 | 39.8 | 77.4 | 112.3 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 45.0 | 29.3 | 62.9 | 53.8 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Nowadays, researchers of video captioning face challenges in collecting paired video-language datasets, which
are often limited in size and time-consuming to create. This limitation hinders the generalization ability of
models, especially for unseen domains. To address this, we propose CroCaps, a CLIP-assisted cross-domain
video captioner. CroCaps utilizes CLIP to boost visual encoders in learning text-correlated video features from
annotated in-domain data. It also leverages out-of-domain data and employs a self-supervised discriminative
communication objective to learn from unpaired data in the target domain. Additionally, we incorporate a
domain separation mechanism in both visual and linguistic domains, allowing for the extraction of domaininvariant features and the generation of domain-invariant sentences. Cross-domain evaluations on MSVD and
MSR-VTT, with one dataset serving as the source domain and the other as the target domain, consistently
demonstrate the commendable performance of CroCaps on both datasets, with significant improvements after
adaptation. Furthermore, within-domain evaluations reveal that CroCaps achieves comparable or even superior
performance compared to state-of-the-art methods, even with limited paired data available in the target
domain.
```

- **Introduction:**

```
With the proliferation of video content across various domains,
such as surveillance, entertainment, education, and healthcare, there
is a growing need for effective video understanding systems. Video
captioning (Chen et al., 2019; Gu et al., 2023; Lin et al., 2021; Yang
et al., 2021) provides textual descriptions that help users navigate and
comprehend video content efficiently. However, it is labor-intensive
and time-consuming to collect a sufficient number of paired videos and
sentences in each domain, which fails to generalize from controlled
videos to real application. When we use cross-dataset simulation for
cross-domain cases, Table 1 reveals that the current methods (Xu et al.,
2022) demonstrate robust performance within their native domains;
however, their efficacy significantly diminishes when applied across
different domains. Therefore, it may be beneficial to transfer the model
trained in an existing domain (i.e., source domain) with pairs of videos
and sentences to a new domain (i.e., target domain) with only unpaired
data or limited annotated samples.
Cross-domain video captioning allows for the utilization of knowledge learned from one domain to benefit another. However, several
limitations hinder the full potential of cross-domain video captioning
methods, which remains largely unexplored. (1) Domain shift. The performance of cross-domain video captioning models is greatly affected by domain shifts, which arise from differences in scene types,
video styles, topics, data collection methods, and caption annotations
across various domains. Table 1 clearly shows that when there is
a significant difference between the two datasets, the cross-domain
performance of existing work (Xu et al., 2022) is very poor. The result
between YouCook (Zhou et al., 2018) and MSR-VTT (Xu et al., 2016)
is much worse than results between MSVD (Chen & Dolan, 2011) and
MSR-VTT, since YouCook only contains videos in cooking scenario.
It is also important to note that these variations are not limited to
the visual domain, as shown in Fig. 1 that illustrates the disparity in
feature distributions and word clouds between the MSR-VTT and MSVD
datasets. Developing robust techniques to effectively address domain
shifts remains a key challenge in this field.
(2) Insufficient availability of annotated data. The limited availability or absence of domain-specific annotated datasets poses a challenge
in training and evaluating cross-domain video captioning models. In
actual practice, it is infeasible for a dataset to include the totality of
videos, and accordingly, it is implausible for a model to have been
exposed to every conceivable scene and genre of video. Collecting a
substantial number of paired videos and sentences for each domain is labor-intensive and time-consuming, but it is also required to maintain
the model’s fundamental performance, with only a limited number of
annotated samples in the new scenarios. This scarcity of labeled data
hampers the development of domain-adaptive models.
Inspired by the strong zero-shot capability of CLIP (Contrastive
Language-Image Pre-training) (Li et al., 2022; Luo et al., 2020), we
propose CroCaps, a CLIP-assisted cross-domain video captioner. The
benefit brought by CLIP is in two typical scenarios: (1) it boosts visual
encoders to learn strongly text-correlated video features trained with
in-domain annotated data; (2) it adjusts the model using out-of-domain
data and employs a self-supervised discriminative communication objective to learn from unpaired data in the target domain. In addition,
domain separation mechanism is also used to extract domain-invariant
features and also generate domain-invariant sentences to reduce the
gap not only in visual domain but also in lingual domain.
To the first challenge in video captioning—‘‘domain shift’’, traditional transfer methods that solely focus on learning low-level domaininvariant features are insufficient, as the gap exists not only in the
visual domain but also in the linguistic domain. To overcome this, we
propose to learn domain-invariant and text-correlated semantic features
in the visual domain, as well as constraining the caption generation
process in the linguistic domain to minimize the influence of domain
characteristics. In regards to the second challenge—‘‘the scarcity of
paired annotations in the target domain’’, traditional unsupervised
learning methods, such as using retrieved sentences or generating
sentences from the source domain model as pseudo-labels, are not
effective due to the substantial differences between description sentences in different domains. To address this, we utilize self-supervised
reinforcement learning with a CLIP-based reward function to ensure the
relevance of description sentences to visual content, aiming to generate
domain-invariant yet accurate captions and improve the efficiency of
using unpaired samples.
Contribution of this paper is concluded as following: In this paper,
we propose a novel CLIP-assisted method for the cross-domain video
captioning task which limited number of researches focus on. Firstly, in
order to effectively address the challenges of domain shift and improve
the cross-domain transferability, the proposed CroCaps learns domaininvariant and text-correlated semantic features in the visual domain,
as well as constraining the caption generation process in the linguistic
domain to minimize the influence of domain characteristics. Secondly,
in order to effectively address the challenges of insufficient annotation
and enhance the efficiency of data utilization, CroCaps provides a CLIPbased reward function for using unpaired out-of-domain data with
self-supervised reinforcement learning. Thirdly, it demonstrates the
superior transferability and generalization capability of CroCaps, as
well as its efficiency in utilizing both paired and unpaired samples,
which obtains better performance on video captioning benchmarks
under cross-domain settings and even with limited annotated data.
```

- **Related work:**

```
In this section, we review related works on visual captioning task,
including within-domain captioning, cross-domain captioning, unsupervised captioning, semi-supervised captioning with unlabeled corpus.
Within-domain video captioning. In the field of video captioning,
there have been significant advancements in developing models that
generate captions for videos within the same domain. These models
typically rely on supervised learning techniques (Chen & Li, 2024;
Nabati & Behrad, 2021; Parvin et al., 2023; Zeng et al., 2024), where
the paired video-sentence annotations are used for training. Various approaches, such as encoder–decoder architectures with attention mechanisms (FCVC-CF-IA (Fang et al., 2019), hLSTMat (Gao et al., 2019),
SHAN (Deng et al., 2021)), have been proposed to improve the quality
and fluency of generated captions. Techniques like semantic fusion
(SAM-SS (Chen et al., 2020), AS-Transformer (Zhang et al., 2022),
TextKG (Gu et al., 2023), SibNet (Liu et al., 2021), SGN (Ryu et al.,
2021), HMN (Ye et al., 2022)) and multi-modal fusion (𝑀3
(Wang
et al., 2018), V-ShaWei-GA (Hao et al., 2018)) have also been explored to capture the semantic knowledge and integrate both visual
and linguistic information in video captioning. Also some polishing
mechanism based methods (Xu et al., 2022) and external knowledge
augmented method (Zhang et al., 2021) are proposed for video captioning. TSTPN (Xu et al., 2022) introduced the polishing mechanism in
an attempt to mimic human polishing process and propose a generateand-polish framework for video captioning. To address the open-book
video captioning problem, a novel Retrieve-Copy-Generate network
is proposed in Zhang et al. (2021), where a pluggable video-to-text
retriever is constructed to retrieve sentences as hints from the training
corpus effectively, and a copy-mechanism generator is introduced to
extract expressions from multi-retrieved sentences dynamically.
Unsupervised image/video captioning. Unsupervised image/
video captioning approaches aim to generate captions without explicitly relying on paired image-sentence or video-sentence annotations.
Instead, they leverage unsupervised learning techniques, such as generative adversarial networks, dual learning and self-supervised training,
to learn from unannotated data. A R2M is proposed in Guo et al. (2020),
which maps semantic concepts to caption sentences using a memory translator that includes a recurrent memory and fusion memory.
In Laina et al. (2019), a novel method is presented to align images and
text in a shared hidden space constructed by structured visual concepts.
By exploiting visual similarities and temporal coherence, these methods
attempt to generate coherent and meaningful captions for images and videos. In Zhu et al. (2023), a prompt-based learning method is
proposed for unpaired image captioning, which attempts to infer the
cross-domain cue information about a given image from the large VLPTMs. In Feng et al. (2019), a novel approach is proposed to train an
image captioning model by an unsupervised strategy with all unpaired
image-sentence training samples, which consists of a sentence generator, a discriminator and an image encoder. However, unsupervised
methods often face challenges in terms of caption quality and diversity,
as they lack explicit supervision from ground-truth annotations.
Semi-supervised captioning with unlabeled corpus. Recently,
most captioning approaches are trained using all paired video-sentence
data. Given that producing such data is an expensive process, there
has been some interests in the community (Chen et al., 2016; Vo
et al., 2022; Wang et al., 2019) to train models with less supervision
that relaxes the reliance on strictly paired data. A reviewer-decoder
architecture with an attention mechanism is introduced in Chen et al.
(2016), which proposes a novel way of using unlabeled textual data
by artificially generating missing visual information. An end-to-end
NOC-REK (Vo et al., 2022) is proposed for novel object captioning,
which enables to generate novel object words beyond the training set,
and two tasks are simultaneously optimized: caption generation and
vocabulary retrieval. A novel topic-aware mixture of experts model
(TAMoE) (Wang et al., 2019) is introduced for zero-shot video captioning and transfers implicit knowledge between seen and unseen actions,
where several topic embeddings are learned to constitute different
experts.
Cross-domain video/image captioning. Cross-domain video and
image captioning refers to the task of generating captions for videos
or images in a domain that is different from the domain used for
model training. These methods aim to leverage knowledge from the
source domain to improve captioning performance in the target domain
by leveraging transfer learning techniques, such as pre-trained model,
domain-invariant representation learning (Bousmalis et al., 2016), dual
learning (Yang et al., 2018), adversarial training (Chen et al., 2017),
self-supervised training (Duan et al., 2023), cross-modal retrieval (Dessì
et al., 2023) and model adaptation (Zhao et al., 2020). Additionally,
several zero-shot methods are proposed for captioning task, which also
requires transfers implicit knowledge between seen and unseen actions
or words.
Overall, within-domain video captioning has made significant
progress, while unsupervised methods provide potential solutions for
generating captions without explicit annotations. However, crossdomain video captioning remains a challenging problem, especially
when transferring to a target domain with significant domain shifts but
no paired training data.
```

# [24-25][v][ASGNet] Adaptive semantic guidance network for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S1077314224003369

- **Published in:** Computer Vision and Image Understanding, Volume 251, February 2025, 104255

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 55.2 | 36.6 | 74.3 | 101.8 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 43.4 | 29.6 | 62.5 | 52.6 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning aims to describe video content using natural language, and effectively integrating information
of visual and textual is crucial for generating accurate captions. However, we find that the existing methods
over-rely on the language-prior information about the text acquired by training, resulting in the model
tending to output high-frequency fixed phrases. In order to solve the above problems, we extract highquality semantic information from multi-modal input and then build a semantic guidance mechanism to adapt
to the contribution of visual semantics and text semantics to generate captions. We propose an Adaptive
Semantic Guidance Network (ASGNet) for video captioning. The ASGNet consists of a Semantic Enhancement
Encoder (SEE) and an Adaptive Control Decoder (ACD). Specifically, the SEE helps the model obtain highquality semantic representations by exploring the rich semantic information from visual and textual. The
ACD dynamically adjusts the contribution weights of semantics about visual and textual for word generation,
guiding the model to adaptively focus on the correct semantic information. These two modules work together
to help the model overcome the problem of over-reliance on language priors, resulting in more accurate video
captions. Finally, we conducted extensive experiments on commonly used video captioning datasets. MSVD
and MSR-VTT reached the state-of-the-art, and YouCookII also achieved good performance. These experiments
fully verified the advantages of our method.
```

- **Introduction:**

```
Video captioning aims to describe video content using natural language, which is a cross-modal task between vision and text. The task
requires computers to be able to efficiently acquire semantic information in videos and output accurate textual descriptions through
semantic alignment between vision and text. Video captioning plays an
important role in many applications such as visual impairment assistance, human–computer interaction, and video retrieval (Chen et al.,
2018; Pan et al., 2016; Song et al., 2017; Wang et al., 2018). Unlike
single-modal visual classification tasks (Dosovitskiy et al., 2020; Liu
et al., 2021a; Touvron et al., 2021) or natural language processing
tasks (Devlin et al., 2018; Radford et al., 2018, 2019; Brown et al.,
2020), cross-modal tasks (Sun et al., 2019; Radford et al., 2021; Kim
et al., 2021) require computers to analyze and perceive data at a
finer granularity. This requires computers not only to understand the
high-level visual semantics in videos at an abstract level but also to
accurately establish feature alignment between vision and text.
Current mainstream video captioning methods usually use an
encoder–decoder structure. Where the encoder is used to extract semantic features from video and text and the decoder is used to generate
text descriptions. The rapid development of large-scale visual and
language models (Wang et al., 2023b; Li et al., 2022a) in recent
years has greatly improved the techniques for models to learn feature
representations from video and text. Most video captioning models
focus mostly on improving the performance of the encoder by utilizing
different levels of features on the encoder side. However, effectively
utilizing the semantic information contained in the features is crucial
to improving the quality of feature representation. He et al. (2023)
proposed a novel video-text pre-training method, VLAB, to transfer
CLIP representations to video pre-training tasks through feature adaptation and blending. Rao et al. (2024) introduced the Collaborative
Multimodal Graph Network, CMGNet, which explored the interactions
between multimodal features within video captions. On the decoder
side, existing approaches have begun to explore improving the semantic consistency between video and text (Huo et al., 2021). However, the
model tends to rely excessively on the co-occurrence relationships
between words learned during training when generating descriptive
statements, resulting in a tendency for the model to output highfrequency phrases. As shown in Fig. 1, the base model is correct in
generating the first few words of the captions. However, when failing
to focus on the correct visual information, the model tends to rely
on previously learned language-prior information to generate some
high-frequency phrases. This results in generating descriptions that are
inaccurate or irrelevant to the video content.
To solve the above problems, we propose an Adaptive Semantic
Guidance Network (ASGNet). ASGNet is a new approach to video
captioning. It not only extracts high-quality semantic information from
multimodal video inputs but also constructs a semantic guidance mechanism to adaptively adjust the contribution weights of visual and textual semantics toward caption generation. Our ASGNet is constructed
based on an encoder–decoder architecture, consisting of a semantic
embedding encoder (SEE) and an adaptive control decoder (ACD).
Specifically, in the SEE encoder, we first design a Semantic Embedding
Module (SEM), which aims to mine semantic information in video static
and temporal features. Then, we propose a Semantic Fusion Module
(SFM) to capture the interactions between video features and textual
features to enhance the semantic expressiveness of the encoder. The
ACD decoder we designed contains a text decision module (TDM) and
a dependency control module (DCM). The TDM guides the decoder to
adaptively focus on the corresponding visual information by utilizing
textual contextual information, while the DCM dynamically controls
the contribution of visual and textual information to the generated
words. These two modules work together to help the model overcome
the problem of over-reliance on language-prior information and thus
achieve cross-modal semantic alignment.
In summary, the main contributions of this paper are as follows:
• We design a semantic enhancement encoder. It contains a Semantic Embedding Module (SEM) and a Semantic Fusion Module
(SFM). The SEM aims at mining semantic information inside the
static and temporal features of the video. The SFM aims to capture
the interaction between video features and textual features to
enhance the semantic expressiveness of the encoder.
• We propose an adaptive control decoder. It contains a Text Decision Module (TDM) and a Dependent Control Module (DCM). The
TDM guides the decoder to adaptively focus on the corresponding
visual information by utilizing textual contextual information.
The DCM dynamically controls the contribution of visual and
textual information to generate words. This reduces the degree
of reliance on language-prior information and helps the model
generate descriptions that are more in accordance with the video
captions.
• We conducted extensive experiments on commonly used video
captioning datasets. MSVD and MSR-VTT reached the state-ofthe-art, and YouCookII also achieved good performance. These
experiments fully validate the advantages of our approach.
```

- **Related work:**

```
2.1. Video captioning
Video captioning aims to generate natural language descriptions
for input videos. In comparison to image captioning, which only
requires understanding static information from a single image, video
captioning is more challenging. It involves comprehending complex
temporal, action, scene, and attribute information across multiple
video frames. Drawing inspiration from machine translation and advancements in convolutional neural networks and large-scale language
models, the mainstream video captioning methods adopt an encoder–
decoder framework. (Chen and Jiang, 2021; Pan et al., 2020; Wang
et al., 2019; Zhang and Peng, 2019; Zhang et al., 2020; Aafaq et al.,
2019; Chen and Jiang, 2019; Venugopalan et al., 2015). The encoder
component employs convolutional neural networks (Szegedy et al.,
2015; Park et al., 2020; Tan et al., 2020) to extract video features
and word embedding models (Mikolov et al., 2013; Joulin et al.,
2016) to capture textual features. These features are then projected
into a shared feature space. Decoders, typically based on recurrent
neural networks (Song et al., 2017; Zhao et al., 2019; Ullah and
Mohanta, 2022) or Transformers (Ghaderi et al., 2022; Gu et al., 2022;
Lebron et al., 2022), align the visual and textual features and generate
video captions. This encoder–decoder approach is straightforward and
efficient, enabling the generation of description sentences with flexible
structure and high accuracy.
In recent years, the video captioning field has had remarkable
innovations and advancements through the utilization of large language models (LLMs). Include VideoGPT+ (Maaz et al., 2024), which
combines the complementary strengths of an image encoder (for detailed spatial understanding) and a video encoder (for modeling global
temporal context), LAVILA (Zhao et al., 2023), which leverages large
language models to learn video-language representations, and VideoLLaMA2 (Cheng et al., 2024), designed to enhance spatiotemporal
modeling and audio comprehension capabilities for video- and audiooriented tasks. These works have not only propelled the development of
multimodal representation learning but also introduced fresh perspectives for future video understanding tasks. Nevertheless, it is crucial to
note that the substantial demands of large language models on data processing scale and computational resources pose formidable challenges
that cannot be overlooked. Exist video captioning methods suffer from
an over-reliance on language priors information, leading to models
tending to output high-frequency fixed phrases. To address this issue,
this paper innovatively designs an adaptive semantic guidance network.
This not only actively addresses the current technical bottlenecks in
video captioning but also offers additional innovative points for the
application of large language models.
2.2. Encoder–decoder model
As early as 2015, Venugopalan et al. (2015) first proposed an
encoder–decoder based video captioning model (S2VT). That is, the
video is first sampled with equally spaced frames, and then using a
feature extraction network for each frame to extract visual features and
employ Long Short-Term Memory (LSTM) (Graves and Graves, 2012)
to generate captions. Chen et al. (2018) considered that sampling video
frames at equal intervals during the encoding stage introduces redundant visual information, and therefore proposed PickNet to extract key
frames of the video. Benefiting from the parallel computing power and
scalability of the Transformer (Vaswani et al., 2017), many excellent
encoder–decoder based methods have emerged in recent years.

In the encoding stage: in order to obtain rich visual semantic
information, Zhang and Peng (2019), Pan et al. (2020), Zhang et al.
(2020) constructs different graph network structures to capture more
detailed video features, respectively. Wu et al. (2022) proposes a Transfer Visual Relationship Detection (TVRD) model to explore depth-level
object relationships. Aafaq et al. (2019) applied the Fourier transform
to the CNN features of the whole video by hierarchically enriching
the temporal action visual features. In order to obtain more textual
semantic information, Pei et al. (2019), Wang et al. (2019), Zheng et al.
(2020), Zhao et al. (2021), Zhong et al. (2023) introduces syntactic
information in captions. In addition, Tu et al. (2021) uses visual labels
to connect vision and text. Liu et al. (2018) proposes the use of twobranch encoding, i.e., content branching and text branching, to encode
the semantic information of the video using visual-text embedding
respectively. Chen et al. (2022) mines the information shared between
samples in the semantic subspace.
In the decoding stage: Ryu et al. (2021) focuses on forming semantic groups using word information decoded at previous time steps
and grouping of video features to guide the prediction of words at
the next moment. Ji et al. (2022)propose a novel attention-based
double-learning approach (ADL) for video captioning. Li et al. (2022b)
proposed an adaptive spatial position module that dynamically predicts
the significant position of each video frame when generating subtitles.
While existing approaches have made good progress in exploring ways
to improve semantic consistency between video and text. However,
the models tend to rely excessively on the co-occurrence relationships
between words learned during training when generating subtitles, resulting in a tendency to output high-frequency phrases. This is still a
difficult problem for video captioning that needs to be solved urgently.
Therefore, we propose an adaptive semantic guidance network for
video captioning.
```

# [24-24][v] Center-enhanced video captioning model with multimodal semantic alignment

- **Link:** https://www.sciencedirect.com/science/article/pii/S0893608024006683

- **Published in:** Neural Networks, Volume 180, December 2024, 106744

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 60.9 | 40.5 | 77.9 | 117.9 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 43.6 | 31.1 | 62.1 | 53.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning aims at automatically generating descriptive sentences based on the given video, establishing an association between the visual contents and textual languages, has attracted great attention and plays a significant role in many practical applications. Previous researches focus more on the aspect of caption generation, ignoring the alignment of multimodal feature and just simply concatenating them. Besides, video feature extraction is usually done in an off-line manner, which leads to the fact that the extracted feature may not adapted to the subsequent caption generation task. To improve the applicability of extracted features for downstream caption generation and to address the issue of multimodal semantic alignment fusion, we propose an end-to-end center-enhanced video captioning model with multimodal semantic alignment, which integrates feature extraction and caption generation task into a unified framework. In order to enhance the completeness of semantic features, we design a center enhancement strategy where the visual–textual deep joint semantic feature can be captured via incremental clustering, then the cluster centers can serve as the guidance for better caption generation. Moreover, we propose to promote the visual–textual multimodal alignment fusion by learning the visual and textual representation in a shared latent semantic space, so as to alleviate the multimodal misalignment problem. Experimental results on two popular datasets MSVD and MSR-VTT demonstrate that the proposed model could outperform the state-of-the-art methods, obtaining higher-quality caption results.
```

- **Introduction:**

```
The task of generating descriptive sentence for a given video naturally establishes association between visual information and natural language, which has been rapidly advancing and has garnered significant attention from both academic and industry communities (Cheng and Gu, 2021, Wang et al., 2023). Video captioning aims at providing natural-language descriptions of the content to the given video, which is an important area of research for video understanding and multimodal representation (Li, Xie, Zhang and Shi, 2023, Wu et al., 2023), and can be applied to many practical applications, such as danger identification and warning, visually impaired people assistance, and human–machine collaboration (Gao et al., 2021, Gao et al., 2023, Gao et al., 2020, Yuan et al., 2019).
Early researches in video captioning primarily focus on template-based models. These approaches involve selecting a pre-defined template and attempting to align sentence components with the corresponding visual content to generate captions. Over time, the research direction of video captioning has shifted toward sequence learning-based methods. These approaches typically adopt a CNN–RNN structure, where the CNN network captures visual features, and the RNN-based decoder (LSTM, GRU) generates descriptive language. Recently, the Transformer model (Vaswani et al., 2017) has outperformed conventional recurrent methods in many sequential modeling tasks. Inspired by the success of Transformer-style models in natural language processing, researchers have explored their use for vision-language multimodal tasks, demonstrating their effectiveness in multimodal representation learning and downstream tasks. Among them, Zhou, Zhou, Corso, Socher, and Xiong (2018) are the first to make a breakthrough in extending the conventional Transformer model to video caption tasks. Li, Yatskar, Yin, Hsieh, and Chang (2019) and Su et al. (2020) employ Bert (Devlin, Chang, Lee, & Toutanova, 2018) to implicitly align visual and textual features to learn multimodal representations for downstream tasks.
However, a common situation in most previous works is the use of offline 2D and 3D extractors to obtain visual features, which raises question about how well these extracted features can be adapted to the captioning task. In the commonly adopted CNN–RNN or CNN-Transformer encoder–decoder structures, video and text are represented in their own specific embedding spaces using different encoding methods, making cross-modal information harder to fuse interactively. In addition, most existing works focus primarily on generating high-quality captions, with little attention given to the importance of cross-modal interactive fusion and its role in enhancing caption generation. Furthermore, there may also be situation where the visual representation and the textual representation do have high similarity in value to some extent, but are in different embedding spaces. Therefore, the multimodal information is actually not well aligned. Generating captions based on such misaligned multimodal information would lead to low-quality results.
To overcome the aforementioned problems, in this paper, we propose an end-to-end center-enhanced video captioning model with multimodal semantic alignment. Compared with previous works that adopt the offline mode to extract visual features, as shown in Fig. 1, the feature extraction and caption generation are integrated into a unified framework, which would make the extracted feature more adapted to the downstream caption generation task. The whole model consists of three main components: feature extraction, multimodal semantic alignment, and center enhancement for caption generation. During the training process, the visual and textual features are first extracted by video and text encoder respectively, and the aligned multimodal representations are learned in a shared latent semantic space. These multimodal aligned representations are then fed into the multimodal encoder–decoder for caption generation, which is further enhanced by the proposed center enhancement strategy.

In all, the main contributions of this work are summarized as
follows:
∙ We propose an end-to-end center-enhanced video captioning
model with multimodal semantic alignment, which integrates the
feature extraction and downstream caption generation task into a
unified framework.
∙ To enhance the completeness of semantic features, we design a
cluster center enhancement strategy to capture key information
within the multimodal features.
∙ To alleviate the misalignment problem of multimodal semantic
fusion, we design to learn the representation of visual and textual feature in the shared latent semantic space for multimodal
semantic alignment.
```

- **Related work:**

```
2.1. Video captioning
Video captioning, which aims to generate descriptive captions for given videos, has gained increasing attention in the field of multimodal representation learning. Existing methods can be broadly divided into three categories: (1) template-based methods, (2) RNNs-based methods, and (3) Transformer-based methods.
2.1.1. Template-based methods
Early research on video captioning adopts template-based methods, where captions are generated by filling in pre-defined sentence templates with relevant visual information extracted from the video. These templates consisted of sentence structures with placeholders for specific types of information such as objects and actions. Typical templated-based video captioning methods are Rohrbach et al. (2013) and Xu, Xiong, Chen, and Corso (2015), in which the models first identify visual contents with classifier. Then the identified concepts are used to form a caption using basic linguistic grammar principles such as subject, verb and object. Although this kind of method is intuitive and easy to understand, it is not flexible and suitable for more complex or diverse video content, as the templates is fixed and may not be able to capture all the details of video.
2.1.2. RNN (LSTM, GRU)-based methods
For the limitations of template-based methods, recent research direction on video captioning have shifted toward sequence-to-sequence learning methods, where the encoder–decoder structure is typically adopted (Chen et al., 2019). In this structure, the pre-trained 2D and 3D CNN extractors are first utilized to encoder the video frame contents into the visual feature, then the RNNs network would serve as language decoder to generate captions based on the sequence of encoded visual feature. Gao et al., 2022, Wang et al., 2022, Wang et al., 2018, Zhang and Peng, 2020, Zhang et al., 2020 and Zhang, Wang, Ma, and Liu (2019) are the representative works of this kind of methods. Specifically, Venugopalan et al. (2015) are the first to use such CNN–RNN structure for video captioning task, however, they adopt a mean pooled scheme when obtaining visual feature, which is too coarse for the language decoder to generate captions. Baraldi, Grana, and Cucchiara (2017) and Pan, Xu, Yang, Wu, and Zhuang (2016) utilize the hierarchical characteristic of video when capturing visual representation. Tang, Wang, and Li (2019) capture rich semantic visual and language features for video captioning with a wider and deeper LSTM network. To generate an imaginary and coherent story with narrative multi-sentences from a group of relevant images, Li, Wang, He, Chen and Wen (2023) propose a knowledge-enriched attention network with group-wise semantic for visual storytelling. Aafaq, Akhtar, Liu, Gilani, and Mian (2019) enriches the visual encoding of video frames with semantic attributes and spatio-temporal dynamics information to improve the performance of video captioning. Pei et al. (2019) design a memory module as an auxiliary to record the correspondence between a word and its corresponding visual contexts across input videos, enabling the model to obtain a more comprehensive understanding for each word and generate high-quality captions. Chen and Jiang (2021) utilizes both visual and temporal features by incorporating motion-guided region proposals and message passing to generate accurate captions for video content. Tu, Zhou, Guo, Gao, and Yu (2021) utilizes pre-detected visual tags which provide crucial visual information while belonging to the textual modality, and adopts a Textual–Temporal Attention Model to align target words with corresponding frames, bridging the gap between vision and language for caption generation. Zeng et al. (2023) construct a video dictionary from all videos to mine the cognitive power of the model’s visual commonsense knowledge, capturing video-related commonsense information and generate more accurate captions. In addition, Ryu, Kang, Kang, and Yoo (2021) and Yu, Ko, Choi, and Kim (2017) further leverage semantic information to improve the caption generation. Jing et al. (2024) and Niu et al. (2023) utilize memory information from designed memory module to facilitate caption generation. Although different, all these methods use a RNNs-based decoder for caption generation.
2.1.3. Transformer-based methods
Recently, inspired by the success of Transformers in the area of sequential modeling tasks, many researchers have explored extending Transformer to video captioning task (Gu et al., 2023, Vaidya et al., 2022, Zhang, Gao, and Yuan, 2024, Zhao et al., 2021, Zhong et al., 2023). Similar to the RNNs-based methods, the pre-trained 2D and 3D CNN extractors are utilized to encode visual feature. Specifically, Jin, Huang, Chen, Li, and Zhang (2020) use a sparse boundary-aware self-attention mechanism and a sparse boundary-aware cross-attention mechanism to selectively attend to important video frames and textual features, which could effectively capture long-term dependencies in video sequences and produces more accurate and diverse captions. Zhang et al. (2020) leverages object and relation information by constructing an object relational graph and employs teacher-recommended learning to guide the caption generation process. Zheng, Wang, and Tao (2020) targets actions in video frames by utilizing syntax-aware embeddings and a multi-modal transformer architecture, improving video captioning by focusing on the most relevant actions. Ye et al. (2022) utilizes a hierarchical modular network, which decomposes the task into sub-tasks of scene recognition, action recognition, and caption generation, and then combines the outputs of these modules to generate a final caption. Li et al. (2022) utilizes a Long Short-Term Relation Transformer with global gating to model long-term temporal dependencies in videos for video captioning. Zhang et al. (2023) propose Spatial Pyramid Transformer, building pyramid structure with shared parameters that consider the semantic connections among different grid resolutions to produce multi-scale captions. Zhong et al. (2022) utilizes skeleton-level tags to enhance the semantic dependencies among visual words, and employs a dual-scale visual-language alignment to reinforce the intra and inter relevance of the tags, ensuring that the generated captions are semantically coherent and aligned with the visual content. Wang, Tang, Li and Cheng (2022) build a dataset for video captioning with emotions named EmVidCap, and propose a framework which takes the consideration of both facts and emotions for video captioning with emotions. Despite significant advancements in previous research, the commonly used offline feature extracted features would be not adaptable to the downstream caption task. And most of previous methods usually focus on the process of text generation, while overlooking the importance of cross-modal alignment, which can aid in generating captions. For the similar image caption task, Zeng, Zhu, Song, and Gao (2022) adopt Swin Transformer as visual backbone, doing end-to-end image captioning based on the learned fine-grained visual semantic information from the proposed tree-structured prototype network. To alleviate the negative impact of offline feature extraction on video, SwinBERT (Lin et al., 2022) is proposed as an end-to-end transformer-based video captioning model, in which the adaptively sparse attention mask is learned for better video sequence modeling. In this paper, we utilize an end-to-end mode to integrate the feature extraction and downstream caption task into a unified framework, and we design a cluster center enhancement strategy to capture key information within the multimodal features, and propose to learn the representation of visual and textual feature in the shared latent semantic space to alleviate the misalignment problem of multimodal semantic fusion.
2.2. Visual encoder
Vision Transformer (ViT) is proposed by Dosovitskiy et al. (2020) as a novel way of applying the Transformer architecture to computer vision tasks. The ViT model breaks down an input image into small patches and flattens them into a sequence of tokens, which can be processed by the standard Transformer encoder. The model then learns to attend to different patches and their features to extract useful visual representations, which can be used for various downstream tasks. ViT has shown impressive results on a number of benchmark datasets, and its success has inspired further research into the use of Transformers for video tasks. For example, ViViT (Arnab et al., 2021) and TimeSformer (Bertasius, Wang, & Torresani, 2021) adopt the Transformer architecture to encode spatial–temporal feature, thus obtaining improved visual representation. Video Swin Transformer (VST) (Liu et al., 2022) adopts a hierarchical design with a Swin Transformer (Liu et al., 2021) as the backbone and a temporal attention module to capture both spatial and temporal information in videos. Deformable Video Transformer (DVT) (Wang & Torresani, 2022) proposes a new video transformer architecture that incorporates deformable self-attention, allowing the model to better capture the spatial–temporal dependencies between video frames. The above video transformer models primarily focus on visual feature extraction and video action detection tasks. In this paper, the VST model serves as the video encoder, and our work focuses on the video captioning task.
2.3. Multimodal representation
Joint vision language understanding is a field that combines computer vision and natural language processing, which has gained attention in recent years. Several studies have demonstrated the potential of multimodal representation learning for vision-language tasks (Gao et al., 2024, Li, Wang, Funakoshi, Okumura and Manabu, 2023, You et al., 2022). However, most language models require large-scale training data, leading to a loss of computation and memory. To address this issue, recent works have explored pre-trained language models for vision-language tasks. By freezing the weights of a pre-trained language model (Alayrac et al., 2022, Zhang, Li, and Okumura, 2024), promising results have been achieved. Besides, masked language models have also been successful in pre-training transformer-based structures to learn language representations (Devlin et al., 2018, Lewis et al., 2020), achieving competitive performance in downstream tasks after fine-tuning. This success has driven exploration of applying masked language models to multimodal representation models with paired visual–textual data (Li et al., 2019, Lin et al., 2022), resulting in competitive performance on vision-language tasks. In this paper, we focus on learning representations for multimodal information in a shared latent semantic space and further capture deep semantic features via incremental clustering based on aligned cross-modal features.
```

# [25-25][v] From visual features to key concepts: A Dynamic and Static Concept-driven approach for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0167865525001394

- **Published in:** Pattern Recognition Letters, Volume 193, July 2025, Pages 64-70

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 57.4 | 38.5 | 74.8 | 111.8 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 44.0 | 29.4 | 62.7 | 55.7 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
In video captioning, accurately identifying and summarizing key concepts while ignoring irrelevant details remains a significant challenge. Mainstream approaches often suffer from the inclusion of semantically irrelevant features, leading to inaccuracies and hallucinations in the generated captions. This study aims to develop a novel framework, Dynamic and Static Concept-driven video captioning model(DiSCo), to enhance the accuracy and coherence of video captions by effectively leveraging pre-trained models and addressing the issue of semantic irrelevance. DiSCo builds upon the conventional encoder–decoder architecture by incorporating a Semantic Feature Extractor (SFE) and a Static-Dynamic Concept Detector (S-DCD). The SFE filters out semantically irrelevant features extracted by the visual model, while the S-DCD identifies critical concepts to guide the large language model (LLM) in generating captions. Both the visual model and the LLM are pre-trained and their parameters are frozen; only the SFE and S-DCD are trained to optimize the feature extraction and concept detection processes. Comprehensive experiments conducted on the MSVD and MSR-VTT datasets show that DiSCo significantly outperforms existing methods, achieving notable improvements in the quality and relevance of the generated captions. The proposed DiSCo framework demonstrates a robust solution for enhancing the accuracy and coherence of video captions by effectively integrating semantic feature extraction and concept-driven guidance.
```

- **Introduction:**

```
Video captioning is the task of automatically generating brief yet informative textual descriptions for videos. This interdisciplinary subfield combines computer vision and natural language processing to process spatiotemporal information, align it with text, and generate captions. Recent advancements [1], [2], [3], [4], [5] in deep learning have propelled video captioning research towards a predominance of encoder–decoder frameworks, leveraging sophisticated models from computer vision and natural language processing (NLP). However, the direct application of these pre-trained models to video captioning encounters significant challenges: (1)Task Mismatch. Visual encoders [6], [7], originally trained for video retrieval or action recognition, struggle to provide the nuanced detail required for generative tasks like video captioning. Their proficiency in recognizing broad actions and retrieving visuals does not translate seamlessly to the detailed scene understanding needed for comprehensive caption generation. (2)Modal Discrepancy. Integrating CV and NLP for video captioning faces the inherent challenge of bridging two vastly different domains. Videos encapsulate rich, multifaceted information, including repetitive or less critical elements that can introduce noise, potentially resulting in misunderstandings by the language model and impacting its interpretative accuracy. It is imperative to refine the encoder’s output by filtering out irrelevant concepts and features to enhance the relevance and accuracy of generated captions.
In this paper, we present a Dynamic and Static Concept-driven video captioning model(DiSCo), which is an LLM-based end-to-end model for video captioning that effectively utilizes pre-trained models. Inspired by human behavior—wherein viewers focus on core, persistent concepts, and neglect ancillary details, DiSCo innovates by utilizing concepts to help training instead of traditional fine-tuning. This approach allows for the identification and emphasis of critical concepts during training, enabling the model to filter and refine semantically relevant features while discarding extraneous visual information. Furthermore, concepts as hints ensures that DiSCo accurately captures the essence of the content and effectively avoids generating off-topic descriptions. With concepts as the driving force, DiSCo efficiently bridges the pre-trained visual model and the LLM without necessitating fine-tuning of these pre-trained models, thus excelling at video captioning tasks while significantly reducing the training cost.
In our approach, the encoder adopts a two-stream architecture, including an RGB branch and optical flow branch, to extract static features and dynamic features respectively, to represent video features more comprehensively. To bridge the video-text disparity and utilize crucial concepts, DiSCo introduces a Semantic Feature Extraction (SFE) module and a Static-Dynamic Concept Detection (S-DCD) module. The SFE module refines raw visual features from encoded video frames. Meanwhile, the S-DCD module utilizes these refined features to anticipate relevant concepts, serving as informative cues for the decoder. Distinct from conventional approaches, DiSCo bypasses the creation of dataset-specific vocabularies for classification, opting instead to estimate word embeddings directly. Our decoder, powered by Large Language Models (LLMs), integrates both semantic and concept features through a carefully crafted prompt. Here, semantic features take precedence as the primary input, complemented by concept features acting as guidance. Leveraging the vast knowledge base and superior reasoning capabilities of LLMs, concept prompts play a pivotal role in steering caption generation. Notably, the synergistic effect of S-DCD in aiding SFE to distill crucial features and align them with textual representations obviates the need for encoder and decoder finetuning. This design dramatically curtails overall training duration, showcasing DiSCo’s efficiency in video captioning while maintaining high accuracy. In summary, our main contributions are as follows:
(1) We present a Dynamic and Static Concept-driven video captioning model(DiSCo), a concept-driven, LLM-based end-to-end framework for video captioning. It pioneers cross-modal synergy via concept prompting, adeptly leveraging pre-trained models’ inherent knowledge for video captioning without necessitating fine-tuning.
(2) Semantic feature extractor(SFE) is proposed to distill critical features from the visual representations of the pre-trained encoders, thus enabling adaptation to video captioning tasks and enhancing the semantic relevance of these features.
(3) Static-dynamic concept detector(S-DCD) is proposed to drive SFE to focus on extracting pertinent semantic features and predict key concepts to guide LLM to concentrate on core concepts to reduce hallucination.
```

- **Related work:**

```
2.1. Video captioning
Originating from seminal work [8], the encoder–decoder architecture has solidified its dominance in video captioning. This paradigm entails encoding individual video frames to extract features, which are then processed by a decoder. Building upon this foundational structure, research efforts have predominantly focused on two core dimensions. First, enhancing video representation: [9] extracted features across multiple levels. Second, bridging the gap between visual and textual modalities, where [10] leveraged high-level semantics for enhanced video-text interaction.
2.2. Video representation
Video is unique compared to pictures in that it contains not only static content (e.g., scenes, objects, and people), but also dynamic content (e.g., events), so spatio-temporal modeling is necessary. Researchers have focused on exploring different 2D/3D visual representations [6], [7], [11]. Since video captioning requires a deep understanding of open word semantics, existing video captioning [4], [10], [12], [13] used features extracted based on models pre-trained on large-scale action recognition datasets. However, for many video tasks (such as action recognition) and datasets, given a single frame of video, existing models trained on image datasets can achieve high performance, even on par with models using multiple frames. The strong performance of single video frames indicates that the video representation is biased towards static appearance information.
2.3. Concept-augmented visual captioning
Numerous research efforts have been dedicated to identifying and leveraging concepts in images or videos to enhance the performance of caption generation. The pioneering work [14] lays the groundwork for concept-augmented visual captioning by selecting a predetermined set of concepts from image captions within the training data, subsequently designing a tailored concept detection network that employs semantic attention to steer caption creation. Subsequent studies [15], [16] also frequently employ concept detection strategies but differ significantly in their application methods. In contrast to static images, video captioning presents a multi-modal challenge, thus necessitating distinct input configurations for concept detection. To tackle this complexity, [17] incorporated audio modalities to enrich the feature representation, yet their approach lacked a training objective for extracting key concepts. On the other hand, [18] augmented optical flow information to improve detection accuracy, but their method still required extracting a fixed set of concepts from the dataset, thus lacking generalizability.
```

# [24-25][v][KG-VCN] Fully exploring object relation interaction and hidden state attention for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0031320324008896

- **Published in:** Pattern Recognition, Volume 159, March 2025, 111138

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 64.9 | 39.7 | 77.2 | 107.1  |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 45.0 | 28.7 | 62.5 | 51.9 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 33.3 | 22.9 | 49.5 | 53.3 |

- **Abstract:**

```
Video Captioning (VC) is a challenging task of automatically generating natural language sentences for describing video contents. As a video often contains multiple objects, it is comprehensively crucial to identify multiple objects and model relationships between them. Previous models usually adopt Graph Convolutional Networks (GCN) to infer relational information via object nodes, but there exist uncertainty and over-smoothing issues of relational reasoning. To tackle these issues, we propose a Knowledge Graph based Video Captioning Network (KG-VCN) by fully exploring object relation interaction, hidden state and attention enhancement. In encoding stages, we present a Graph and Convolution Hybrid Encoder (GCHE), which uses an object detector to find visual objects with bounding boxes for Knowledge Graph (KG) and Convolutional Neural Network (CNN). To model intrinsic relations between detected objects, we propose a knowledge graph based Object Relation Graph Interaction (ORGI) module. In ORGI, we design triplets (head, relation, tail) to efficiently mine object relations, and create a global node to enable adequate information flow among all graph nodes for avoiding possibly missed relations. To produce accurate and rich captions, we propose a hidden State and Attention Enhanced Decoder (SAED) by integrating hidden states and dynamically updated attention features. Our SAED accepts both relational and visual features, adopts Long Short-Term Memory (LSTM) to produce hidden states, and dynamically update attention features. Unlike existing methods, we concatenate state and attention features to predict next word sequentially. To demonstrate the effectiveness of our model, we conduct experiments on three well-known datasets (MSVD, MSR-VTT, VaTeX), and our model achieves impressive results significantly outperforming existing state-of-the-art models.
```

- **Introduction:**

```
Video Captioning (VC) is an interdisciplinary task that combines the Computer Vision (CV) and Natural Language Processing (NLP) domains. Generally, given the video content, the VC algorithms will produce accurate and understandable sentences. To this end, the bulk of the VC models adopt an attentive encoder-decoder framework [[1], [2], [3]] as the baseline, enabled by the advancements in Deep Learning. In the encoder phase, the focus is on obtaining discriminative visual representations via processing original video content. In the decoder phase, the decoder LSTM [4] learns the latent semantic information while using the last hidden state to process encoder outputs for achieving discriminative visual features and outputting semantic-rich captions. By adopting this framework, VC models can effectively translate video content into natural language sentences.
In recent years, the advent of object detection and attention mechanisms has drawn much research attention to investigating object-level relationships. Zhang et al. [5] and Yan et al. [6] established a learnable object relation graph to thoroughly study the relationships between objects in temporal and spatial terms with the assistance of GCNs. To obtain the enhanced object proposals, Bai et al. [7] proposed a Conditional Graph to aggregate spatio-temporal information into the latent features. However, these existing methods relied solely on object features, because a feed-forward network followed by a softmax function is used to generate an adjacency matrix for exploring object relationships. As a result, the learned adjacency matrix might miss critical semantic relationships that could be inferred from prior knowledge, failing to capture complex and nuanced interactions between objects. This leads to less informative or overly simplistic object relationships. Additionally, these methods often use Graph Convolutional Networks (GCNs) to update the relationship graphs, which are prone to over-smoothing and relational uncertainty. In such cases, either predicted relationships between objects are too weak to produce semantic-rich captions, or excessive object relationships are created, resulting in over-smoothed terms to construct the objects-relation graph. In our model, to overcome this problem, the external knowledge graph model is introduced to precisely predict the relationships of objects, bridging inter-objects with substantial and explicit information. Meanwhile, a global vertex is introduced to the object-relation graph. In this way, communications between objects are enabled to anticipate relationships by KGs as well as aggregated features. Instead of GCNs, we choose a Graph Transformer to capture both local and global relationships within the graph.
On the other hand, in the decoder phase, several researchers prefer to utilize the last hidden state of the decoder LSTM to filter the encoded visual features, thereby providing the visual feats inputted to the current decoder LSTM cell with more advanced and discriminative information at each time step. For example, Hua et al. [8] suggest a bi-layer decoder including Attention and Language LSTMs to produce captions step by step. Owing to the interaction between Attention LSTM and Language LSTM, visual features input to the Language LSTM are continuously updated and improved. One issue with these methods is that they do not fully leverage the elaborative semantic visual feats by directly applying the current hidden state of the decoder LSTM to forecast the next word, and as a corollary, the predicted words lack semantic information. For instance, in Fig. 1, the baseline model with an attentive decoder is shown to have limited ability to capture fine-grained information. While it is able to identify the broad video content of "cooking", it fails to distinguish specific actions such as "frying". This limitation can be attributed to the model's simple reliance on trying to acquire semantic-rich and discriminative visual features to input into the decoder LSTM for learning. And then, the current hidden state of the decoder LSTM, i.e., the learning results, will be employed to predict the next word. Nevertheless, in order to earn coherent and appropriate captions, wouldn't it be preferable to combine the elaborative and discriminative features with the learning results? Guided by this perspective, we propose a hidden State and Attention Enhanced Decoder (SAED). In this plan, the current hidden state is processed by integrating it with the dynamically updated visual and graph feats to amplify its semantic information before generating the next word.

Combining the above insights, we propose a new framework for video captioning by exploring object relation interaction and hidden state attention. Our model incorporates three blocks, i.e., Knowledge Graph Prediction, Object Relation Graph Interaction, and Hidden State and Attention Enhanced Decoder. First, we utilize the pre-trained Knowledge Graph (KG) on the identified objects to predict the relationships between them. Notably, we pre-train an exclusive KG using captions from the VC Datasets. After that, we use the Object Relation Graph Interaction (ORGI) for extracting relation features, where a global node is created to interact with other graph nodes under the aid of the Graph Transformer [9]. Finally, hidden states and updated attention features are concatenated to predict words. In brief, our major contributions are summarized as follows:
• We design an Object Relation Graph Interaction module (ORGI) to propose a Knowledge Graph based Video Captioning Network (KG-VCN). A knowledge graph is pre-trained to predict relations between detected objects for efficiently assisting decoders to produce robust language sentences with accurate relation descriptions.
• To implement adequate information flow across all nodes, we specially construct a global node that connects all graph nodes. In this manner, we can alleviate the adverse effects due to erroneous or missed relations by the pre-trained knowledge graph.
• We propose a hidden State and Attention Enhanced Decoder (SAED) that concatenates hidden states and updated attentions for improving the prediction ability of next words. Unlike existing methods, we use both hidden states and dynamically updated attention features to enhance the accuracy of decoding.
```

- **Related work:**

```
2.1. Video captioning
In early years, template-based methods [[10], [11], [12]] have mainly been proposed to handle the issue of video captioning. Specifically, after performing object, scene, and motion detection, these methods utilize language templates to assemble these identified components to provide standardized and smoothed video captions. Afterwards, due to the boom of Convolutional Neural Networks (CNN) [13,14] and Recurrent Neural Networks (RNN) [15,16], worldwide researchers have drawn extensive attention to encoder-decoder neural network structures. In the sub-sections, we clarify and review the prosperous development of video captioning models with encoder-decoder structures from the standpoints of video feature extraction strategies.
2.1.1. Only Extracting Appearance Features
For the sake of simplicity and efficiency, many methods[[17], [18], [19], [20]] intend to generate video captions by using pre-trained 2DCNN models to extract only appearance features from a sequence of frames, including VGG [21], ResNet [22] and Inception-V4 [23]. For instance, Song et al. [19] employed ResNet [22] to extract representative frame-level features in the encoder. It is inevitably unstable or incomplete [20] when a single processing flow is used to encode visual information. With this in mind, Liu et al. [20] employed two branches to encode visual information for forming two data processing flows. One branch collects global visual scene information, another one captures local visual semantic information, and the outputs of two branches are then combined by using soft-attention mechanisms. For achieving the unsupervised objectives during VC training, Ji et al. [17] proposed an encoder-decoder framework to reconstruct the original video content via decoder outputs. Undoubtedly, these models have achieved excellent performance due to wonderful structures and advanced thoughts. Nevertheless, the generalization performance of these methods is still not satisfied on more datasets. Instead, researchers have sought to identify additional motion features to improve the logic and semantic complexity of generated captions.
2.1.2. Appearance and motion features
For instance, Gao et al. [24] extracted appearance and motion features that are fused to produce global semantic features for predicting the semantic concepts. To produce more precise visual features, Chen et al. [25] extracted motion features by optical flow graphs [26], and then utilized these motion features as a spatial attention weight matrix to perform attention operations on visual features. Zheng et al. [27] employed motion features collected by C3D [28] to guide the construction of "predicate". In order to validate the case of possibly generating incorrect action verbs, an action-guided captioning model is often designed to dynamically fuse the "predicate" with the previously predicted words. Pei et al. [29] built a memory structure to store the descriptive information for each word in the vocabulary for accurately detecting actions. Thanks to the various methods of processing motion features, video captioning has undoubtedly been experiencing its thriving and prosperous times.
2.2. Graph structures
As object detection technologies have achieved great advances [[30], [31], [32], [33]] in recent years, many researchers have focused on exploring relation interactions among detected objects by constructing graph structures. Tu et al. [34] introduced a novel approach attaching information from the hidden states of future and past LSTM decoders to object attention features, and employed GCNs [35] to collect intra-relations between attention results. Based on the work, Yan et al. [6] constructed an object relation graph (ORG) based on object similarity and spatial-temporal relations, and then used GCNs to encode object relationships. Zhang et al. [5] took a similar approach but built a learnable ORG to better investigate temporal and spatial connections between objects. In all of these studies, GCNs are used to reinforce object features during relational reasoning. Bai et al. [7] proposed a different approach in which they incorporated spatio-temporal information into latent object proposals to obtain enhanced object proposals. They then constructed a dynamic graph to derive higher-dimensional semantic knowledge from the enhanced object proposals. Finally, Han et al. [36] focused on learning discriminative object representations by modeling the relationships between indistinguishable and tricky proposals in different videos.
The aforementioned methods with graph structures have utilized GCNs to update relationships between objects. However, GCNs have the drawback of being inflexible. Besides, the ways that they constructed graph structures and adopted attention mechanisms among object features are prone to the problems of redundant connections and relationship ambiguity. To tackle this challenging problem, we introduce an externally pre-trained knowledge graph to predict the relationships between detected objects, leading to much precise and rich information on object interactions. It should be mentioned that Tu et al. [37] also proposed the idea of using KGs to predict the relationships among objects. However, instead of constructing a graph structure, they directly concatenated the vectors of predicted relationships and object features, and then used a Transformer model to further encode the concatenated feature vectors. Direct concatenation without a graph structure is possibly short of logical reasoning. Moreover, relying solely on KGs to predict the relationships among objects may be arbitrary and unpredictable, as most methods neglect the possibility of connections between objects whose relationships have not been predicted by KGs, resulting in an incomplete understanding of the underlying relation features. In consideration of possibly missed relations not predicted by KGs, we also introduce a global node to implement communications among detected objects whose relationships are not anticipated by KGs, and also to obtain an aggregate of object features. Finally, we chose the graph transformer to update the co-occurrence of objects.
2.3. Semantic enhancement
Attention mechanisms have widely been used in deep learning, but they often struggle to focus on the most relevant visual features that are necessary for generating reliable and productive word prediction scores. To address this issue, there are several methods developed for acquiring and enhancing discriminative visual semantic features, which can generate accurate description words.
In order to take into account both the spatial and temporal structures in a video, Yan et al. [38] proposed a novel spatial-temporal attention mechanism. It allows the decoder to automatically select significant regions in the most relevant temporal segments to aid in word prediction. To the same end, Zhao et al. [18] proposed a Co-Attention Model Based RNN (CAM-RNN) to identify the most relevant visual and textual features for generating captions. Ryu et al. [39] proposed a Semantic Grouping Network, which encodes videos into semantic groups according to relevant frames and the corresponding words of partially decoded captions, and adaptively produces the next word based on the semantic groups. In order to enhance the alignment between target words and corresponding frames, Tu et al. [40] presented a Textual-Temporal Attention model (TTA), which uses language contexts and key visual tags to determine key frames. For delving into the high-level semantic attributes of videos, Aafaq et al. [3] enriched the temporal and spatial dynamics of video representations by hierarchically applying Short Fourier Transforms and object detections. Liu et al. [20] used a dual-branch encoding scheme to encode visual information. One branch of the scheme captures visual appearance information, and the other one uses visual-semantic joint embeddings to extract semantic information about the video. The features by the two branches are then combined using soft-attention mechanisms, and fed to decoders for predicting captions. With the aim of consolidating object relations in a spatio-temporal context, Deng et al. [41] presented a novel Relation Distillation Network, where the features of each object proposal in the reference frame are augmented by aggregating their relation features over the proposals in the supportive pool.
Undoubtedly, the strategies of existing models can earn semantic enhancement to a certain extent. However, one major flaw of these models is that they fail to make full use of semantic features, as they directly take the hidden states of the LSTM decoder to make predictions for next words. These results in captions are predicted in a semantically vague way. To solve this problem, we propose a hidden State and Attention Enhanced Decoder (SAED), where the current hidden state is integrated with visual features to amplify semantic information before generating the next word.
```

# [25-25][v][Track4Cap] Frame-by-Frame Multi-Object Tracking-Guided Video Captioning

- **Link:** https://ieeexplore.ieee.org/abstract/document/10884880

- **Published in:** IEEE Transactions on Circuits and Systems for Video Technology ( Volume: 35, Issue: 7, July 2025)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 62.1 | 42.5 | 79.8 | 127.2|

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 44.6| 30.5| 63.6| 57.7 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning through deep learning presents a multifaceted challenge that encompasses the extraction of complex spatio-temporal visual features and the synthesis of meaningful natural language descriptions. Most of the existing deep learning models can be broadly grouped as either convolution-based or transformer-based encoder-decoder networks, with video captions generated from features encoded at the pixel level for the former, and from features encoded at grid, frame, or video levels depending on encoder complexity for the latter. This paper advocates frame-level features as a more balanced and compact representation for fast caption generation, and introduces the Tracking-guided Information Augmentation for Captioning (Track4Cap) model, which integrates tracking-guided information augmentation to enhance frame-level features without relying on complex architectures or additional data modalities. Specifically, Track4Cap employs the Frame-by-Frame Multi-object Tracking module (FMoT) to identify the most relevant objects in the input video and the Object Relation Encoder (ORE) to model inter-object relationships as supplementary high-level cues for caption generation. By avoiding time-consuming end-to-end training and leveraging compact representations, Track4Cap achieves computational efficiency while improving captioning performance. Extensive experiments on two commonly used benchmark datasets demonstrate that Track4Cap not only achieves faster inference times but also outperforms state-of-the-art convolution-based and transformer-based video captioning models.
```

- **Introduction:**

```
In the early years of information transmission, images and text were the main mediums. However, with the rapid advancement of technology and multimedia, videos have become a new and powerful way of transmitting information. In this context, video captioning [1] came into being which aims to generate meaningful and coherent descriptions for video content. It plays an important role in daily life, with various applications such as assisting the visually impaired [2], enhancing human computer interaction [3], and improving video retrieval [4], [5].

Although video captioning shares a common foundation with image captioning [6] as both strive to capture and describe the essence of visual content, they differ significantly in terms of complexity, as video captioning expands upon image captioning by moving from interpreting a single static image to a sequence of frames over time. This shift requires understanding not only objects and their relationships in the spatial domain, but also their dynamic variations and contextual interactions in the spatio-temporal domain. Consequently, video captioning faces the added challenge of identifying the most relevant objects and their interactions within dynamic and intricate multi-component scenes, since the most salient object in a single frame may not represent the primary object of the entire video sequence, and not all object relationships are relevant for caption generation.

Most existing video captioning methods follow an encoder-decoder architecture. The encoder processes the input video to extract features, which are then mapped to textual descriptions by the decoder. These encoder-decoder frameworks can be broadly categorized into convolution-based and transformer-based models. Convolution-based models, as illustrated in Fig. 1(a), predominantly utilize pre-trained 2D and 3D convolutional neural networks (CNNs) [7], [8], [9], [10] to extract spatial and temporal features at the pixel level, where information is derived directly from the intensity or color values of individual pixels and their local neighborhoods. While providing detailed local appearance and motion information, they often suffer from high computational costs due to the large data volume and fine spatial granularity. Some convolution-based methods incorporate object detection modules [11], [12], [13] to extract salient object-level features, improving object-centric descriptions. However, these methods further increase computational overhead and are prone to errors in identifying primary objects in cluttered scenes.

Transformer-based models, as shown in Fig. 1(b), encode information at various levels of granularity depending on the encoder’s complexity. Grid-level features represent coarse-grained information derived from non-overlapping patches of pixels, aggregating local details into compact representations that balance spatial resolution and computational efficiency [14], [15], [16]. Frame-level features capture the overall spatial context of individual frames, summarizing their appearance while disregarding temporal dependencies [17]. In contrast, video-level features provide a holistic global representation by integrating information across multiple frames, emphasizing long-term dependencies and overall video context [18], [19], [20].

While transformer-based models offer flexibility in processing different levels of granularity, they face significant challenges. Optimizing both encoder and decoder components during end-to-end training can be computationally demanding, and large-scale transformer-based models often require extensive training data and resources to effectively learn interdependencies between video content and captions. These challenges make it crucial to strike a balance between computational efficiency and representation quality.

To achieve a balance between computational efficiency and representation quality, this paper leverages frame-level features, which provide a compact, low-dimensional representation that significantly reduces computational overhead compared to pixel-, grid-, and video-level representations. Frame-level features effectively capture high-level contextual information but inherently lack the fine-grained spatio-temporal details necessary to model dynamic object interactions and generate accurate video captions. To address these limitations, we propose the Tracking-guided Information Augmentation for Captioning (Track4Cap) framework, as illustrated in Fig. 1(c).

Track4Cap integrates an intermediate information augmentation stage designed to enhance frame-level features with finer-grained spatio-temporal cues. This stage comprises two key modules: the Frame-by-Frame Multi-object Tracking (FMoT) module, which identifies and tracks salient objects across frames, and the Object Relation Encoder (ORE) module, which models inter-object relationships to provide high-level semantic cues. By augmenting frame-level features with dynamic and relational information, Track4Cap enables more accurate and contextually aware caption generation while maintaining computational efficiency.

The motivations for introducing the information augmentation stage are threefold. First, it enriches frame-level features with object-centric and relational information, effectively bridging the gap between simple frame-level encoding and the detailed modeling found in grid- or video-level representations. Second, it reduces computational costs by focusing on tracked objects rather than processing all pixel-level information in video frames. Third, it enhances contextual understanding by explicitly modeling interactions between tracked objects, which are critical for generating accurate and meaningful captions.

The main contributions of this paper are summarized as follows:

- We propose Track4Cap, a novel encoder-decoder framework that leverages frame-level features enriched with tracking-guided information for efficient and accurate video captioning.

- We introduce two core modules for information augmentation: (i) the Frame-by-frame Multi-Object Tracking (FMoT) module, which identifies and tracks salient objects, and (ii) the Object Relation Encoder (ORE), which models inter-object relationships to enhance contextual understanding.

- We demonstrate that Track4Cap achieves state-of-the-art performance on benchmark datasets, with significant reductions in inference time and computational complexity. Extensive ablation studies validate the contributions of individual components and their combinations.
```

- **Related work:**

```
In this section, a review of the latest video captioning models is provided, categorized into convolution-based models and transformer-based models based on their encoder architectures.

A. Convolution-Based Models
The development of video captioning in deep learning began with stage-wise encoder-decoder frameworks that combined convolutional neural networks (CNNs) and recurrent neural networks (RNNs). Typically, pre-trained 2D/3D CNNs were employed in the encoder to extract video features at the pixel level, with or without auxiliary object detection using models such as Region-based Convolutional Neural Networks (R-CNN) [11]. These pixel-level features were then used by RNN-based decoders, such as Long Short-Term Memory networks (LSTMs) [21], to generate captions.

Despite the effectiveness of 2D/3D CNNs in capturing local appearance and motion features, the lack of semantic representation in these features has been a significant limitation. Consequently, efforts have been directed toward enhancing spatial and temporal semantic representations within the encoder. For example, the Semantic Grouping Network (SGN) [22] improves spatial semantics by aligning video frames with partially decoded subtitles, reducing visual redundancy and aiding the decoder in more accurate word prediction. Similarly, the Motion Guided Region Message Passing (MGRMP) model [23] enhances temporal semantics by employing 3D CNNs to extract regional temporal features and model inter-regional relationships across frames.

On the decoder side, research has focused on refining low-level semantic information using high-level cues through knowledge distillation. For instance, the Decoder Refined Semantic enhancement towards Frequency Diffusion (RSFD) [24] employs frequency-aware diffusion to capture critical low-frequency semantic details, thereby improving captioning performance. The Two-Step Transformer-based Polishing Network (TSTPN) [25] combines 2D/3D CNNs with a generation transformer and a polishing transformer to enhance cross-modal sequence mapping via cross-modal attention and knowledge distillation.

Integrated improvements targeting both encoder and decoder architectures have also gained traction. The Hierarchical Representation Network with Auxiliary Tasks (HRNAT) [26] reconstructs visual content through auxiliary tasks such as cross-modality matching and syntax-guided learning, resulting in linguistically and grammatically accurate captions. Stay-in-Grid Video Captioning (SGCAP) [16] incorporates a bilinear sequential attention encoder for spatial-temporal modeling, complemented by a cross-modal sequential attention decoder for dynamic region representation.

Object detection has been another effective strategy to improve captioning by incorporating high-level cues about object relationships and interactions. For instance, the Object Relational Graph with Teacher-Recommended Learning (ORG-TRL) [27] leverages graph learning to represent object interactions, integrating these cues with an external language model through a teacher-recommended learning mechanism. The Spatio-Temporal Graph with Knowledge Distillation (STG-KD) [28] models object interactions across space and time, providing interpretable links while enhancing stability with a knowledge distillation framework. Transitive Visual Relationship Detection (TVRD) [29] introduces a novel action detection mechanism to extract deep semantic relationships between objects through object-action and object-object graphs. Additionally, the Long Short-Term Relation Transformer (LSRT) [30] captures both short-term spatial relations and long-term dependencies among objects, employing a global gating unit to regulate information flow.

While graph-based approaches effectively capture interaction information, they often introduce redundant connections that can reduce computational efficiency. To address this, some methods incorporate language semantics to refine object interactions. For example, Syntax-Aware Action Targeting (SAAT) [31] associates sentence structure with the actions of detected objects, providing precise action representations. Similarly, the Textual-Temporal Attention Model (TTA) [32] aligns visual tags of detected objects with descriptive words, improving the alignment of visual and textual information.

Recently, hierarchical and multi-branch networks have further advanced video captioning by integrating object-level semantics with scene-level information. For example, Reinforcement Learning with Hierarchical Modular Network (RLHMN) [33] learns multi-level visual representations—spanning entities, predicates, and sentences—thereby reducing ambiguities caused by polysemous verbs. Element-aware Video Captioning (EvCap) [34] integrates object, action, and scene features via a multi-branch encoder-decoder, enhancing the perception of specific elements through a post-fusion mechanism.

Despite these advancements, convolution-based approaches face significant computational challenges in capturing global spatio-temporal semantic features. However, their strength lies in their ability to extract fine-grained spatial and temporal details, which remain a crucial foundation for video captioning tasks.

B. Transformer-Based Models
Transformers, known for their exceptional sequence processing capabilities, have transitioned successfully from natural language processing to image and video processing, establishing themselves as a dominant framework for video captioning. By extending the Swin Transformer originally designed for images, the Video Swin Transformer (VidSwin) incorporates the temporal dimension, enabling grid-level feature encoding [35]. VidSwin serves as the encoder backbone for several transformer-based video captioning models, including SwinBERT [1], Diverse Video Captioning by Adaptive Spatio-temporal Attention (VASTA) [36], and the Concept-aware and Task-specific model (CAT) [14], all of which employ Bidirectional Encoder Representations from Transformers (BERT) [37] as the decoder.

SwinBERT employs an end-to-end training strategy that integrates VidSwin and leverages sparse attention mechanisms to process video data efficiently, reducing redundancy during feature extraction. VASTA further optimizes encoding by dynamically selecting keyframes to minimize redundant information entering VidSwin. CAT enhances semantic representation by introducing a concept parser to extract high-level cues and a multi-modal graph to model relationships among visual, semantic, and textual features, enabling richer and deeper semantic representations in generated captions.

To capture more holistic spatio-temporal features, the Unified Video and Language Pre-training model (UniVL) [38] leverages large-scale pre-training to establish connections between videos and text, focusing on video-level representations. However, transformer-based video-level models face significant computational challenges, particularly during end-to-end training, where optimizing both encoder and decoder on large datasets often leads to slow convergence and inefficient loss utilization. To address these issues, the MEta Loss TRansformer (MELTR) [19] was introduced as a plug-in module that dynamically integrates multiple loss functions to improve the optimization balance between the encoder and decoder. While MELTR enhances performance, its additional computational overhead increases overall complexity. To alleviate these computational demands, CoCap [20] directly utilizes compressed video data, extracting video-level information from I-frames, motion vectors, and residuals. This approach significantly reduces the data volume and inference time, offering a more efficient alternative for video-level processing.

Recent advancements in cross-modal learning have spurred the development of transformer backbones for encoding features at various levels of granularity. The Contrastive Language-Image Pre-training model (CLIP) [39], built on the Vision Transformer (ViT) [40], specializes in frame-level feature generation through extensive pre-training to associate images and text. CLIP-based video captioning models, such as CLIP4Clip [41] and Expectation-Maximization Contrastive Learning (EMCL) [42], demonstrate the utility of CLIP as an encoder. CLIP4Clip identifies the most relevant frames by comparing frame-level features with captions, while EMCL iteratively optimizes CLIP’s feature space, yielding compact representations. Both methods have shown significant improvements across multiple evaluation metrics.

The Concept-awARE video captioning framework (CARE) [17] addresses the limitations of frame-level features by incorporating additional modalities, such as audio and text, and employing multimodal-driven concept detection to uncover latent video themes, thereby enhancing decoding performance. Hierarchical Semantic Representation and Aggregation (HSRA) [43] complements frame-level features by integrating finer-grained pixel-level visual information. HSRA reconfigures visual semantics into a hierarchical “object-action-event” structure, effectively capturing key object details and dynamic global context.

The proposed Track4Cap model distinguishes itself from existing transformer-based frame-level approaches through its simplicity and computational efficiency. Unlike CLIP4Clip, EMCL, and HSRA, Track4Cap does not rely on additional ground-truth captions for feature enhancement learning. It also avoids using additional modalities, such as audio or text, integral to CARE, and does not incorporate finer-grained pixel-level visual information, as employed by HSRA. Instead, Track4Cap exclusively utilizes frame-level features and adopts a stage-wise framework that leverages a pre-trained CLIP model as the decoder without requiring additional fine-tuning. By integrating a high-level information augmentation module, Track4Cap enhances frame-level features with salient object-relation cues, effectively addressing the limitations of frame-level representations. This design achieves an optimal trade-off between computational efficiency and captioning performance, eliminating the need for complex encoders to model temporal dependencies or additional data modalities for caption quality improvement.
```

# [24-25][v][HSRA] Action-Driven Semantic Representation and Aggregation for Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10759582

- **Published in:** IEEE Transactions on Circuits and Systems for Video Technology ( Volume: 35, Issue: 4, April 2025)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 62.2 | 39.2 | 78.4 | 110.1 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 46.9 | 30.9 | 64.8 | 55.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning, a challenging task that entails generating natural language descriptions of visual content, often fails to effectively grasp the essence of action semantics. To harness the power of action detection to facilitate a deeper understanding of the video content, we propose an action-driven method, named Hierarchical Semantic Representation and Aggregation (HSRA) network. This method explicitly exploits action clues with a hierarchical semantic representation module, which models visual semantics in a three-level structure: “object-action-event”. By employing learnable action queries, our approach injects extensive action semantics into the model, thereby enabling more accurate and context-rich captions. To further enhance semantic alignment and understanding, we introduce a semantic aggregation composed of a semantic interaction module and a semantic refinement module. This component facilitates the alignment of semantics across different levels and emphasizes key information, ultimately leading to significant improvements in semantic consistency between the video and generated captions. We performed extensive evaluations on two well-established public datasets, MSVD and MSR-VTT, and the findings consistently demonstrate that our proposed HSRA network outperforms contemporary state-of-the-art methods.
```

- **Introduction:**

```
Video captioning, a research field dedicated to automatically describing the visual content of videos using natural language, has attracted considerable attention owing to its potential applications across diverse domains, including video understanding [1], [2], video summarization [3], [4], video quality assessment [5] and video question answering [6], [7]. Achieving proficiency in this task necessitates a comprehensive and fine-grained understanding of visual semantics, thereby rendering it challenging.

Efforts to extract and model visual semantics have led to the development of complex semantic encoders. For instance, Yan et al. [10] propose Global-Local Representation Granularity for Video Captioning(GL-RG), employing a global-local encoder to capture diverse visual representations and generate comprehensive lexical features tailored based on varying video content. Zhang et al. [11] propose Object Relational Graph with Teacher-Recommended Learning for Video Captioning (ORG-TRL), leveraging an object relation graph in the encoder stage to capture object interaction characteristics and enhance the visual representation of video captions. To enhance the alignment between visual content and generated captions at a finer granularity, Shen et al. [12] propose Weakly Supervised Dense Video Captioning (WSDVC), which establishes a weak mapping between nouns and grid areas to diversify video captions at a fine-grained level. Furthermore, Ye et al. [8] introduce Hierarchical Modular Network for Video Captioning (HMN), a hierarchical modular network providing rich semantic information at the entity, predicate, and sentence layers, thereby ensuring more coherent and meaningful generated captions.

Despite the promising results demonstrated by the aforementioned models, they still exhibit several limitations: 1) Although fine-grained semantic encoding has been investigated, the extraction of action semantics remains insufficient. For example, as illustrated in Fig. 1, a marked disparity is evident between the distributions of ground truth captions (Fig. 1 (a)) and those generated by the HMN (Fig. 1 (b)) and RSFD (Fig. 1 (c)). Specifically, more than 30% of the human-annotated ground truth captions contain equal to or more than two verbs, whereas only equal or less than 9% of the predicted captions include two verbs. Additionally, Fig. 1 (d) shows that humans may describe a video with a variety of actions, such as “playing a guitar,” “singing,” and “sitting.” In contrast, the HMN model captures only one action, specifically “playing a guitar”. This suggests that the absence of adequate action semantics in the visual representations restricts the model’s capability to comprehensively describe activities involving multiple objects in the video. 2) Existing methods that utilize hierarchical or multi-granularity architectures often overlook semantic alignment between different layers or granularities. These semantic gaps hinder the comprehensive understanding of the video and may result in semantic inconsistencies and ambiguities in the captions, leading to a mismatch with the video content. 3) Most methods treat the multi-granularity semantic outputs of the encoder equally, potentially resulting in poor-quality captions that describe irrelevant or unimportant events, thereby diminishing the performance of video captioning. For instance, a video depicting children playing in a park should emphasize the actions of the children rather than the trees or the buildings in the background. If the model treats all semantic information equally, it may include excessive background details in the description, thereby diluting the focus on the children’s movements during play.

To address the aforementioned issues, we propose a Hierarchical Semantic Representation and Aggregation (HSRA) network. This network is designed to capture fine-grained semantics of visual content through a hierarchical semantic representation module, structuring the visual semantics into an “object-action-event” hierarchy. The “event” denotes specific occurrences involving one or more “objects” engaging in “actions” within a given context, which is supposed to be semantic-accurately described by the generated caption sentence. “Objects” refer to the entities partaking in the event, aligning with the nouns in the captions, while “actions” denote the behaviors or activities undertaken by these objects, correlating with the verbs in the captions. Notably, different from existing methods, our approach diverges from existing methodologies by explicitly detecting actions in the video with an encoder-decoder architecture motivated by End-to-End Object Detection with Transformers (DETR) [13]. This strategy enhances the depiction of interactions among objects in the generated captions, offering a more accurate and detailed description of the visual content.

To bridge the semantic gap across different layers and highlight visually prominent semantics, we further incorporate a semantic aggregation module into the proposed HSRA. The semantic aggregation module comprises two components: a semantic interaction module and a semantic refinement module. The former deepens the aggregation of video content and aligns semantics across varying levels of granularity, enabling the generation of semantically consistent captions. The latter diminishes the impact of non-essential semantics and emphasizes visually significant content, thereby producing captions that are both more precise and visually grounded. Subsequently, captions are generated using a Bi-LSTM based decoder. Our proposed method has been rigorously evaluated using four metrics across two benchmark datasets (MSVD [14] and MSRVTT [15]). It not only attains state-of-the-art results on both datasets but also validates the effectiveness of each module through comprehensive experimentation.

To summarize, the contributions of this paper are three-fold:
- We present the Hierarchical Semantic Representation and Aggregation (HSRA) network, designed to capture the hierarchical visual semantics of “object-action-event” using a hierarchical semantic representation module. This module, featuring a DETR-inspired action detection layer, significantly improves the precision and variety of verbs in the generated captions.

- We propose a semantic aggregation that incorporates both a semantic interaction module and a semantic refinement module to align semantics across different hierarchical levels and augment principal visual semantics, benefiting from which the HSRA could produce video captions that are both semantically consistent and visually grounded.

- We conduct extensive evaluations of the proposed method on two benchmark datasets (MSVD [14] and MSRVTT [15]), and the results show that our HSRA effectively promotes the quality of generated captions.
```

- **Related work:**

```
Our work is closely related to several popular research topics. We first review the Encoder-decoder based Methods in the field of video captioning. Then, we discuss the methods related to Semantic Extraction and Modeling in the domain of video captioning.

A. Encoder-Decoder Based Methods
The purpose of video captioning is to “understand” and “describe” the semantic content of video data. Currently, the majority of video captioning models adopt the encoder-decoder structure. For instance, S2VT [16] introduced a two-layer encoder-decoder structure with shared LSTM, encoding video data features in the first layer and decoding them in the second. Yao et al. [17] introduced an attention mechanism focusing on temporal rather than spatial features for video captioning. For a long time, researchers have focused on how to improve the encoder-decoder structure modification. For example, Peris et al. [18] developed a Bi-LSTM-based encoder that processes both preceding and forthcoming information. Song et al. [19] designed a hierarchical encoder utilizing an attention mechanism to selectively focus on specific frames for predicting relevant words based on temporal attention.

Zhang et al. [11] incorporated a GCN (Graph Convolutional Neural Network) in the encoder and an external language model (ELM) in the decoder to infuse extensive language knowledge into the video captioning model. Tang et al. [20] utilized a Transformer-based decoder network in the pre-training phase to adeptly learn distant visual and linguistic dependencies, coupled with a clip-based video encoder to enhance performance. Wu et al. have proposed a unified framework that captures multi-level cues in an end-to-end manner, integrating low-level features and high-level cues to generate captions for a given video sequence. Most recently, Lin et al. [21] implemented a video transformer, SwinBERT, to encode spatiotemporal representations adaptable to various video lengths and frame rates without specific design modifications, benefiting from denser video frame sampling to achieve significant performance improvements. In contrast, our approach focuses on capturing action semantics through learnable action queries within a module, resulting in more accurate captions characterized by precise and diverse verbs.

B. Semantic Extraction and Modeling
Extensive research has been conducted on semantic information extraction and modeling in video captioning. Zhang et al. [11] utilized object relational graphs to capture object interactions and employed an external ELM language model to enrich the visual-semantic representation of video captions. Ryu et al. [22] introduced an algorithm to identify the most distinctive phrases in partially decoded captions, establishing a correlation between phrases and corresponding video frames to cluster semantically similar frames, thus enhancing visual-text alignment. Yan et al. [23] presented the GLR model, incorporating a global-local encoder to process diverse visual representations and generate detailed lexical features tailored to varying video contents. Ye et al. [8] developed hierarchical modular networks that deliver comprehensive semantic information across the entity, predicate, and sentence levels, ensuring the generation of more coherent and meaningful captions. Wang et al. [24] propose a feature extraction module that skillfully learns distinct tendencies of visual feature representations for sentence components, such as objects, actions, and states, under the guidance of Part-Of-Speech (POS) tags. Nadeem et al. [25] unveiled a novel global-local fusion network that leverages global-local fusion blocks (GLFB) for encoding and integrating features along with visual-spatial aspects derived from different parts of speech (POS) components. Although existing works [23], [24] consider the representation of video content from the perspectives of part-of-speech and global-local contexts, they do not emphasize the importance of action-level semantics in video captioning. Our model, however, explicitly captures and leverages action cues in videos by introducing learnable action queries, thus enabling the generation of more accurate captions. Besides, we introduce a semantic aggregation aimed at achieving a deeper understanding of visual semantics, thereby enhancing the semantic coherence between the caption and the visual content.
```

# [24-24][v] OmniViD: A Generative Framework for Universal Video Understanding

- **Link:** https://ieeexplore.ieee.org/document/10654930, https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_OmniViD_A_Generative_Framework_for_Universal_Video_Understanding_CVPR_2024_paper.pdf

- **Published in:** 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.7 | 42.2 | 78.1 | 122.5 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 44.3 | 29.9 | 62.7 | 56.6 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
The core of video understanding tasks, such as recognition, captioning, and tracking, is to automatically detect objects or actions in a video and analyze their temporal evolution. Despite sharing a common goal, different
tasks often rely on distinct model architectures and annotation formats. In contrast, natural language processing
benefits from a unified output space, i.e., text sequences,
which simplifies the training of powerful foundational language models, such as GPT-3, with extensive training corpora. Inspired by this, we seek to unify the output space
of video understanding tasks by using languages as labels
and additionally introducing time and box tokens. In this
way, a variety of video tasks could be formulated as videogrounded token generation. This enables us to address various types of video tasks, including classification (such as
action recognition), captioning (covering clip captioning,
video question answering, and dense video captioning), and
localization tasks (such as visual object tracking) within a
fully shared encoder-decoder architecture, following a generative framework. Through comprehensive experiments,
we demonstrate such a simple and straightforward idea is
quite effective and can achieve state-of-the-art or competitive results on seven video benchmarks, providing a novel
perspective for more universal video understanding.
```

- **Introduction:**

```
In recent years, the proliferation of video content across various applications, such as online education and live streaming, has profoundly impacted our daily lives. Videos have
evolved into a captivating and immersive medium for information delivery, emphasizing the pressing demand for
the development of automated algorithms capable of understanding the actions [46], events [49], and moving objects [81] within video sequences. As a result, the field
of video understanding has undergone significant expansion and encompassed a diverse range of tasks, including
action recognition [3, 33, 60, 64, 80, 103], video captioning [16, 36, 61], and object tracking [4, 20, 95, 116].
For a long period, research in video understanding has
often adopted a task-specific paradigm, i.e., designing specialized architectures and loss functions to cater to the
unique requirements of different tasks and benchmarks [10,
46, 69, 107]. Despite the promising results with highcapacity deep neural networks, these methods [30, 82, 104,
113] are tailored for a particular objective and less adaptable to deployment in scenarios of diverse needs. To mitigate this issue, video foundation models [92, 93, 101, 105],
have gained emerging attention for their impressive performance across a broad spectrum of video tasks and potential in realizing the vision of Artificial General Intelligence
(AGI). However, while generic spatial-temporal representations can be learned with these models, adapting them to
different downstream tasks oftentimes requires carefully designing and fine-tuning task-specific heads.
In this paper, we posit such limitation originates from
the diversified annotations for different video tasks, e.g., a
set of action categories for action recognition [12, 80, 103],
sentences for captioning [36, 61], and continuous segments
(coordinates) for events (object) localization [20, 23, 71].
This naturally necessitates task-specific designs for better
optimization. In contrast, different tasks in natural language
processing (NLP) enjoy a sharable output space, i.e., text
sequences, which promotes the development of large language models, such as GPT [77, 78] and Llama [45, 83, 84].
Drawing inspiration from this, we leverage word tokens in
natural languages to represent semantic information that is
important for coarse-grained tasks like action recognition,
video captioning, and video question answering, and additionally introduce special time tokens and box tokens that
provide localization capabilities in both spatial and temporal dimensions, particularly useful for fine-grained tasks
like dense video captioning and visual object tracking. With
such an enriched vocabulary that consists of word, time, and
box tokens, the output format, as well as training objectives
of different tasks, can be well unified. Please refer to Figure 1 for a better illustration.
With this in mind, we present OmniViD, a generative
framework that approaches various video tasks as a language modeling task conditioned on video inputs. OmniViD adopts an encoder-decoder architecture, where a dedicated video encoder and a language encoder are employed
to extract the multimodal features from diverse inputs. Considering the remarkable redundancy in video data, we propose a lightweight MQ-former to enhance the efficiency of
video representations for subsequent modeling. The MQformer utilizes three types of learnable queries, i.e., content,
sentence, and box queries, to aggregate the frame features
from the video encoder through cross-attention. Finally, a
token decoder is applied to generate a token sequence from
the above vocabulary.
We validate the effectiveness of OmniViD on five representative video tasks, including action recognition, clip
captioning, video question answering, dense video captioning, and visual object tracking. The results demonstrate that
OmniViD achieves new state-of-the-art or at least competitive results on the prevalent video benchmarks. For example, using VideoSwin-Base [64] as the video encoder, we
achieve state-of-the-art performance on action recognition
(83.6% top1 accuracy on Kinetics-400 [46]), clip captioning (56.6 on MSRVTT [107] in terms of CIDEr ), video
question answering (42.3% accuracy on MSRVTT [107]),
dense video captioning (5.6 on ActivityNet [10] in terms
of SODA c), and visual object tracking (88.9 on TrackingNet [69] in terms of normalized precision). For the first
time, video tasks of different modalities and granularity can
be supported by a single framework.
```

- **Related work:**

```
2.1. Task-specific Methods for Video Understanding
Task-specific video understanding models could be roughly
divided into classification, captioning, and localization approaches. Video action recognition is the most representative classification task in the video domain, which aims to
recognize human actions in a video. Existing methods, including both CNN-based [32, 33, 46, 66] and Transformerbased models [3, 30, 64], widely encode the action labels
as one-hot vectors and employ cross-entropy loss for supervised training. Captioning tasks, on the other hand, typically generate a textual description for a video clip [61, 125,
126] or an untrimmed long video [44, 99, 113] with a text
decoder like BERT [47]. It is worth noting that captioning
long videos involves the additional challenge of temporal
event localization within the video, making it a more complex task. We categorize the open-ended video question answering [50, 51, 59] as a specific type of captioning task due
to the consistent output format between them. Localization
tasks, represented by visual object tracking [20, 25, 96], estimate the trajectory of a target object in a video sequence
given its position in the first frame. Following the practice
in object detection [11, 38, 40], a box head is oftentimes
adopted to regress the coordinates of the tracking object.
In summary, divergent prediction heads have been developed in various video tasks to adapt to the specific format
of annotations, which poses a challenge to derive a unified
solution. In this paper, we rethink the design of a universal video understanding framework from a novel perspective, i.e., redefining an output space that could be shared
by different video tasks. Within this unified space, the development of general architectures and training objectives
become distinctly feasible.
2.2. Unified Video Models
Recently, researchers have undertaken prominent efforts to
unify video tasks within specific domains. OmniVL [92]
and InterVideo [101] represent significant strides in the
realm of video-language pretraining, which are pre-trained
on large-scale video-text data and achieve superior results on multimodal video tasks like text-to-video retrieval
and video captioning. Beyond these advancements, UNLoc [111] and UniVTG [76] have sought to tackle a diverse array of temporal localization tasks within a single
framework. They accomplish this by simultaneously predicting saliency scores and boundary offsets for each frame
(clip). Compared to video-language and temporal localization, spatial localization in the video domain, i.e., tracking,
is more fragmented in terms of task definition, model architecture, and benchmarks. Unicorn [109] marks a significant
step forward by employing a fully shared CNN-based encoder and box head for various tracking tasks, utilizing a
target before distinguishing between them. Subsequently,
with the prominent success of vision transformer [11],
OmniTracker [94] and UNINEXT [110] push the boundaries of unification in tracking models by incorporating
Transformer-based detectors. Despite the achievements of
these approaches, they are still constrained by task-specific
heads, leaving considerable space for greater unification of
video understanding. To address this limitation, we unify
diverse tasks with a sharable output space and address them
with a fully shared generative framework.
2.3. Autoregress Modeling in Computer Vision
AutoRegressive modeling [114] is a statistical modeling
technique that predicts the current state of a sequence
based on historical observations, which has achieved remarkable success in the field of natural language processing (NLP) [24] and time series analyasis [34, 68]. Inspired by this, researchers in the vision community have
also attempted to explore its potential for visual understanding. Pix2SeqV1&V2 [18, 19] expand the textual vocabulary
with quantized image coordinates. With this, they address
several fundamental image tasks, e.g., object detection, and
image captioning, in a unified autoregressive manner. Following this idea, ARTrack [102] and SeqTrack [21] further
support the visual object tracking task. VisionLLM [100],
on the other hand, directly builds vision-centric frameworks
upon pre-trained LLMs, with the hope of transferring their
knowledge to visual understanding with minimal resource
overhead. In this work, we leverage autoregressive modeling to the design of a universal video understanding framework. In addition to the expansion to temporal localization
tasks with unique time tokens, our method also explores the
advantages of autoregressive modeling for a universal video
understanding framework for the first time.
```

# [23-23][v] Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10203714, https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_CVPR_2023_paper.pdf

- **Published in:** 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| – | 41.4 | – | 120.3 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| – | 30.0 | – | 57.2 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
In this work, we introduce Vid2Seq, a multi-modal
single-stage dense event captioning model pretrained on
narrated videos which are readily-available at scale. The
Vid2Seq architecture augments a language model with special time tokens, allowing it to seamlessly predict event
boundaries and textual descriptions in the same output sequence. Such a unified model requires large-scale training
data, which is not available in current annotated datasets.
We show that it is possible to leverage unlabeled narrated
videos for dense video captioning, by reformulating sentence boundaries of transcribed speech as pseudo event
boundaries, and using the transcribed speech sentences as
pseudo event captions. The resulting Vid2Seq model pretrained on the YT-Temporal-1B dataset improves the state
of the art on a variety of dense video captioning benchmarks including YouCook2, ViTT and ActivityNet Captions.
Vid2Seq also generalizes well to the tasks of video paragraph captioning and video clip captioning, and to few-shot
settings. 
```

- **Introduction:**

```
Dense video captioning requires the temporal localization and captioning of all events in an untrimmed video [45,
98, 127]. This differs from standard video captioning [62,
69, 79], where the goal is to produce a single caption for
a given short video clip. Dense captioning is significantly
more difficult, as it raises the additional complexity of localizing the events in minutes-long videos. However, it also benefits from long-range video information. This task is
potentially highly useful in applications such as large-scale
video search and indexing, where the video content is not
segmented into clips.
Existing methods mostly resort to two-stage approaches [36, 45, 96], where events are first localized and
then captioned. To further enhance the inter-task interaction between event localization and captioning, some approaches have introduced models that jointly solve the two
tasks [19,98,127]. However, often these approaches still require task-specific components such as event counters [98].
Furthermore, they exclusively train on manually annotated
datasets of limited size [34, 45, 126], which makes it difficult to effectively solve the task. To address these issues,
we take inspiration from recent sequence-to-sequence models pretrained on Web data which have been successful on a
wide range of vision and language tasks [3,10,12,101,113].
First, we propose a video language model, called
Vid2Seq. We start from a language model trained on Web
text [77] and augment it with special time tokens that represent timestamps in the video. Given video frames and transcribed speech inputs, the resulting model jointly predicts
all event captions and their corresponding temporal boundaries by generating a single sequence of discrete tokens, as
illustrated in Figure 1 (right). Such a model therefore has
the potential to learn multi-modal dependencies between
the different events in the video via attention [90]. However
this requires large-scale training data, which is not available in current dense video captioning datasets [34,45,126].
Moreover, collecting manual annotations of dense captions
for videos is expensive and prohibitive at scale.
Hence we propose to pretrain Vid2Seq by leveraging unlabeled narrated videos which are readily-available at scale.
To do this, we reformulate sentence boundaries of transcribed speech as pseudo event boundaries, and use the transcribed speech sentences as pseudo event captions. We then
pretrain Vid2Seq with a generative objective, that requires
predicting the transcribed speech given visual inputs, and
a denoising objective, which masks spans of transcribed
speech. Note that transcribed speech may not describe the
video content faithfully, and is often temporally misaligned
with the visual stream [31, 42, 70]. For instance, from the
example in Figure 1 (left), one can understand that the grey
skier has descended a slope from the last speech sentence
which is said after he actually descended the slope. Intuitively, Vid2Seq is particularly suited for learning from such
noisy supervision as it jointly models all narrations and the
corresponding timestamps in the video.
We demonstrate the effectiveness of our pretrained
model through extensive experiments. We show the importance of pretraining on untrimmed narrated videos, the
ability of Vid2Seq to use both the visual and speech modalities, the importance of the pretraining objectives, the benefit
of joint caption generation and localization, as well as the
importance of the language model size and the scale of the
pretraining dataset. The pretrained Vid2Seq model achieves
state-of-the-art performance on various dense video captioning benchmarks [34, 45, 126]. Our model also excels
at generating paragraphs of text describing the video: without using ground-truth event proposals at inference time,
our model outperforms all prior approaches including those
that rely on such proposals [49,75,124]. Moreover, Vid2Seq
generalizes well to the standard task of video clip captioning [8, 105]. Finally, we introduce a new few-shot
dense video captioning setting in which we finetune our pretrained model on a small fraction of the downstream training dataset and show benefits of Vid2Seq in this setting.
In summary, we make the following contributions:
(i) We introduce Vid2Seq for dense video captioning.
Given multi-modal inputs (transcribed speech and video),
Vid2Seq predicts a single sequence of discrete tokens that
includes caption tokens interleaved with special time tokens that represent event timestamps. (ii) We show that
transcribed speech and corresponding timestamps in unlabeled narrated videos can be effectively used as a source
of weak supervision for dense video captioning. (iii) Finally, our pretrained Vid2Seq model improves the state of
the art on three dense video captioning datasets (YouCook2,
ViTT, ActivityNet Captions), two video paragraph captioning benchmarks (YouCook2, ActivityNet Captions) and two
video clip captioning datasets (MSR-VTT, MSVD), and
also generalizes well to few-shot settings.
```

- **Related work:**

```
```

# [23-24][v] IcoCap: Improving Video Captioning by Compounding Images

- **Link:** https://ieeexplore.ieee.org/document/10272675

- **Published in:** IEEE Trans. Multimed., 26 (2024), pp. 4389-4400

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.1 | 39.5 | 76.5 | 110.3 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 47.0 | 31.1 | 64.9 | 60.2 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 37.4 | 25.7 | 53.1 | 67.8 |

- **Abstract:**

```
Video captioning is a more challenging task compared to image captioning, primarily due to differences in content density. Video data contains redundant visual content, making it difficult for captioners to generalize diverse content and avoid being misled by irrelevant elements. Moreover, redundant content is not well-trimmed to match the corresponding visual semantics in the ground truth, further increasing the difficulty of video captioning. Current research in video captioning predominantly focuses on captioner design, neglecting the impact of content density on captioner performance. Considering the differences between videos and images, there exists an another line to improve video captioning by leveraging concise and easily-learned image samples to further diversify video samples. This modification to content density compels the captioner to learn more effectively against redundancy and ambiguity. In this article, we propose a novel approach called Image-Compounded learning for video Captioners (IcoCap) to facilitate better learning of complex video semantics. IcoCap comprises two components: the Image-Video Compounding Strategy (ICS) and Visual-Semantic Guided Captioning (VGC). ICS compounds easily-learned image semantics into video semantics, further diversifying video content and prompting the network to generalize contents in a more diverse sample. Besides, learning with the sample compounded with image contents, the captioner is compelled to better extract valuable video cues in the presence of straightforward image semantics. This helps the captioner further focus on relevant information while filtering out extraneous content. Then, VGC guides the network in flexibly learning ground truth captions based on the compounded samples, helping to mitigate the mismatch between the ground truth and ambiguous semantics in video samples. Our experimental results demonstrate the effectiveness of IcoCap in improving the learning of video captioners. Applied to the widely-used MSVD, MSR-VTT, and VATEX datasets, our approach achieves competitive or superior results compared to state-of-the-art methods, illustrating its capacity to handle the redundant and ambiguous video data
```

- **Introduction:**

```
Video captioning is a challenging task that requires the model to learn semantics and express through natural language. The main challenge in this task is understanding the diverse visual contents in the videos. Recently, many solutions have been proposed to solve this problem, e.g., leveraging better video representations [1], [2], complex network designs [3], [4], and end-to-end learning [5], [6]. These works facilitate a better understanding of video semantics and generate coherent descriptions of the visual contents.

Despite significant improvements, understanding video semantics remains a challenging task. A major obstacle to achieving this understanding is the semantic ambiguity in videos, caused by their visual redundancy. The contents of videos are diverse and difficult to precisely trim with specific descriptions. As illustrated in Fig. 1(a), some contents, such as irrelevant and minor events, are not described by the ground truth, and without particular descriptions, they are implicit for the network to understand. In addition, contents such as transitions are related to the events described by the ground truth but do not contain valuable semantics for network learning. These contents present a challenge for neural networks. The video captioner is always hard to generalize redundant contents or misguided by ambiguous semantics. Meanwhile, the mismatch between descriptions and visual contents further increase the difficulties in learning video semantics. All this defeats induce the captioner may be misguided by trivial and irrelevant semantics.

Comparably, image contents are more concise, and the semantics are salient, as shown in Fig. 1(b). There are no irrelevant events or transitions apart from the valuable contents in the images. Meanwhile, the image descriptions are precise. The ground truth description can summarize most contents. This makes the image samples easier to be captioned. Empirically, results from image captioning datasets [7] are often better than video captioning in various metrics.

The primary distinction between images and videos for captioners lies in content density. The redundancy and ambiguity in video data cause the network to struggle in generalizing complex video semantics. While previous works have focused on proposing better captioner designs, improved architectures to increase network capacity, which aids in learning semantics and handling redundancy. However, an another line for improving video captioning has been overlooked: modifying content density to enhance the learning process of video captioners. In this work, we propose a novel learning method called Image-Compounded learning for video Captioners (IcoCap). IcoCap compounds concise and easily-learned image semantics into video samples, diversifying the visual contents and compelling the network to learn against redundant contents. Besides, the compounded image semantics are more easily learned compared to video semantics, which are similar to a strong competitor [8], [9] for learning video semantics. To address video captioning, the captioner must investigate valuable video cues in contrast to easily-learned image contents. This further enhances the captioner's ability to learn video semantics. Additionally, IcoCap alleviates the challenges of learning from mismatched descriptions by encouraging the network to flexibly learn descriptions based on visual semantics, rather than relying on rigidly pre-assigned captions.

Specifically, IcoCap comprises two modules: the Image-Video Compounding Strategy (ICS) and Visual-Semantic Guided Captioning (VGC). In detail, ICS is designed to compound image content into video content. This approach further diversifies the video samples, guiding the video captioner to learn against redundancy. Simultaneously, the introduction of easily-learned image content compels the network to extract valuable video cues while filtering out irrelevant elements. Additionally, IcoCap addresses the issue of ambiguous video semantics by VGC, which facilitates flexible learning of semantics based on visual content. In VGC, the ground truth is selected from relevant descriptions rather than strictly corresponding to the original video ground truth. A visual-semantic consistency factor is introduced to adjust the captioning process, promoting the network to focus on the salient visual content rather than concentrating on minor and detailed contents.

The main contributions can be summarized below:

We propose an Image-Compounded learning for video Captioners (IcoCap). IcoCap introduces image samples and compounds the images into video contents. IcoCap impels the network to mine valuable video cues against the semantic ambiguity in videos.

We propose an Image-video Compounding Strategy (ICS) and Visual-semantic Guided Captioning (VGC). ICS provides a series of operations to compound images and videos, which promotes the network's ability to learn video semantics against ambiguity. VGC helps the network to flexibly learn complex video contents from ICS, rather than rigidly following the ground truth.

Without complicated designs or networks, our method performs favorably or outperforms the state-of-the-art methods on various datasets, including MSR-VTT, MSVD, and VATEX.
```

- **Related work:**

```
Video Representation: Representation of video [5], [10], [11], [12], [13], [14] is a long-standing problem in the representation learning [15], [16], [17], [18]. Numerous works have emerged, proposing diverse architectures and approaches that focus on exploiting the unique characteristics of video data to achieve effective and robust representations. In representation, the intuitive idea behind video representation is to extend the principles of image-based CNNs, which have demonstrated remarkable success in tasks such as object recognition and image classification.

One notable approach to incorporate temporal information into the original CNN framework is by introducing 3D kernels [19], [20]. These kernels extend the receptive field in the time dimension, thereby enabling the network to capture the relationships between sequential frames. This extension results in 3D Convolutional Neural Networks (3D CNNs) [20], which are specifically designed to process video data by jointly learning spatial and temporal features and have demonstrated considerable improvements in video representation tasks compared to their 2D counterparts. However, one drawback of 3D CNNs is the increased computational complexity and memory requirements, which can pose challenges in terms of scalability and efficiency. I3D [21] inflated the filters and pooling layers of 2D CNNs into 3D, enabling the network to learn richer spatio-temporal features. The I3D model achieved significant improvements in action recognition tasks and demonstrated the potential of incorporating pre-trained 2D CNN knowledge into video representation learning. More variations [22] of 3D CNNs further provide many video-based designs to boost the performances of representations in various tasks.

Moreover, recent works [10], [23], [24], [25] pay more attention to the large-scale pre-training. Motivated by the success of Bert [26] in NLP, many works [24], [27] propose to leverage the similar pre-training strategies to videos. Significant improvements occur in video tasks after applying the large-scale pre-training [11], [28] and various transformer-based networks [5], [25], [29]. Besides, tasks like mask-modeling [26], contrastive learning [30], [31], etc., further empower the representation ability of networks. CLIP [23], as a typical pre-training model, has also been proven that possesses remarkable ability in correlating language semantics and has already been widely used in various domains [1], [32]. These video-based designs have contributed to the evolution of video representation learning, enabling more effective and discriminative representations for various tasks. Despite the progress made thus far, video representation remains an active area of research, with ongoing efforts to develop more efficient and accurate models capable of handling the ever-increasing complexity and scale of video data. In our work, we focus on the learning method of video captioning and directly applying representation method according [1], [23], [32].

Video Captioning: Video captioning [33], [34], [35] is a challenging and complex task that aims to generate a natural language sentence to describe a given video sequence. Unlike image captioning, where the objective is to generate descriptions for static images, video captioning methods need to handle intricate video data that encapsulates diverse and dynamic semantics. The temporal dimension of video data adds a level of complexity that requires sophisticated approaches to capture and summarize the underlying content effectively.

In detail, the common approach in video captioning is the encoder-decoder framework, which employs a CNN to encode visual information and an RNN or LSTM to generate captions sequentially. Donahue et al. [36] proposed the Sequence-to-Sequence Video-to-Text model, which combined a 2D CNN with an LSTM to generate captions. Chen et al. [37] introduced the TDConvED network—a convolutional sequence-to-sequence learning framework, specifically tailored to enhance video captioning. Most recent works [4], [5], [38], [39] also follow this framework and present various solutions to further boost the performances. Moreover, Chen et al. [40] propose to select frames in video for video captioning. Pan et al. [41] introduce a visual semantic embedding model to specifically consider the relationship between the semantics of the entire sentence and video content unexploited.

Moreover, another line of evolution is the video representation method. Works in video captioning apply features from some pre-trained models to represent videos. Models like bottom-up [42] in image representations, 3D CNNs [21], [22] in video representation, or generic large-scale pre-training models [5], [6], [23] are applied in video captioning to represent video data. Then, various methods [2], [43], [44], [45] are designed to investigate the semantic cues from well-trained representations and solve video captioning. Yang et al. [46] conducted a comparative analysis between CLIP features and ImageNet pre-trained features for video captioning. Additionally, they introduced an auxiliary task designed to discern the correspondence between video content and associated concepts. Some recent works [4], [47], [48] introduce complicated structures to mine detailed information from video features and achieve significant improvements. Besides, some works [4], [5] further propose end-to-end frameworks for representing videos from scratch and exploring the detailed instances and events in the video frames.

In this article, we propose a novel method to improve video captioning by introducing image samples to aid in video learning. The highly diverse video contents can induce ambiguity in video semantics, which can be challenging for network learning. In contrast, image samples are typically concise and explicit, making them easily learnable for the network and possessing clear semantics. Our approach compounds the image and video samples to impel the network to learn semantics from the combined samples. This leads the networks can better learn video semantics against the redundancy and ambiguity. Our experimental results demonstrate that the proposed learning approach outperforms existing methods across various datasets and metrics.
```

# [25-25][v] DSSM-KG: Dual-Stream State-Space Modeling with Adaptive Knowledge Injection for Video Captioning

- **Link:** https://dl.acm.org/doi/10.1145/3731715.3733474

- **Published in:** ICMR '25: Proceedings of the 2025 International Conference on Multimedia Retrieval

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 57.9 | 40.0 | 77.0 | 111.7 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 47.6 | 31.2 | 65.1 | 59.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning aims to generate natural language descriptionsof video content. Recent methods extract temporal and spatialinformation separately and use dataset-specific prior knowledgeto enhance caption quality. However, they may be inadequate injoint spatiotemporal modeling and lack the utilization of common-sense knowledge, making it difficult to fully understand the video.To address these issues, this paper proposes a dual-stream state-space model (DSSM-KG) based on cross-modal knowledge injec-tion. Specifically, by integrating the heterogeneous Mamba with theTransformer in both parallel and sequential manners, we constructthe spatially enhanced dual-stream state-space module (S-DSSM)and the temporally enhanced dual-stream state-space module (T-DSSM) to strengthen joint spatiotemporal modeling. Additionally,a knowledge graph that integrates both commonsense and dataset-specific information is constructed and adaptively injected into thedecoder to furnish the model with extensive video-related knowl-edge. Experimental results indicate that the structural designs ofDSSM-KG, together with the knowledge injection mechanism ,demonstrate significant efficacy, yielding competitive performanceon mainstream video captioning datasets such as MSVD and MSR-VTT.
```

- **Introduction:**

```
Learning-based video captioning is primarily based on the encoder-decoder architecture [17, 16], with significant advancements madethrough the integration of the Transformer, particularly in twokey directions. The first direction involves extensively mining ad-ditional sources of information, including conceptual priors [19],retrieval data and object detection. These methods primarily relyon priors derived from the training data and have not incorporatedadditional commonsense knowledge to further enhance the model’sunderstanding of video semantics. The second direction focuses onthe extraction of temporal information, such as motion informationand motion vectors [11]. Typically, such information is extractedusing pre-trained models, fused with spatial visual features, andthen fed into the decoder. However, the lack of interaction and inte-gration between spatial and temporal information may negativelyimpact model performance. In contrast, reference [14] employsCLIP to extract features and combines a Transformer encoder witha decoder to directly generate text descriptions; although the struc-ture is simple and avoids reliance on prior information, its superiorperformance is primarily attributable to the robust capabilities ofthe pre-trained CLIP model. Despite the rich knowledge embeddedin CLIP through training on image-text pairs, it still falls short incapturing dynamic temporal information in videos; a standaloneTransformer decoder is insufficient for fully modeling complexspatiotemporal relationships, thereby directly limiting the model’scomprehension of video semantics and the overall quality of captiongeneration.In this paper, we propose a dual-stream state-space model withcross-modal knowledge injection (DSSM-KG) to enhance the model’sspatiotemporal modeling capability while leveraging a knowledgegraph (KG) to improve video comprehension. Specifically, to en-hance the representation of spatial features in videos, we developeda Spatial Enhanced Dual-stream State-Space Module (S-DSSM) thatemploys heterogeneous Mamba and Transformer components toprocess features independently. To improve temporal modeling, we introduced a Temporal Enhanced Dual-stream State-Space Module(T-DSSM) that rearranges spatial features; this module leveragesthe time scanning capabilities of the SSM to reinforce temporal fea-ture modeling and further employs a Transformer to boost overallrepresentational capacity. Subsequently, we implemented weightsharing between S-DSSM and T-DSSM, which not only reduces thenumber of model parameters but also facilitates joint spatiotempo-ral modeling of visual features. Finally, we inject knowledge intothe decoder in an adaptive manner, thereby assisting the model inachieving a more comprehensive understanding of video content.The main contributions of this paper are as follows:
• We propose a dual-stream state-space model with cross-modal knowledge injection . By integrating S-DSSM andT-DSSM, the model effectively enhances the extraction ofspatial and temporal information, thereby improving jointspatiotemporal modeling capabilities;
• We propose an adaptive knowledge injection strategy that in-tegrates a constructed KG into the model, thereby enrichingits knowledge base and enhancing its video understandingcapabilities;
• Quantitative experiments on the MSVD and MSR-VTT datasetsdemonstrate that the proposed model generates video cap-tions that outperform those of most competing methods,with absolute CIDEr scores exceeding the state-of-the-art by4.8% and 0.1% on the MSVD and MSR-VTT datasets, respectively.
```

- **Related work:**

```
```

# [25-25][v] MGTR-MISS: More Ground Truth Retrieving based Multimodal Interaction and Semantic Supervision for video description

- **Link:** https://www.sciencedirect.com/science/article/pii/S0893608025006975

- **Published in:** Neural Networks, Volume 192, December 2025, 107817

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.8 | 39.7 | 77.0 | 111.1 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 43.5 | 29.2 | 63.0 | 55.0 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 33.2 | 22.7 | 49.3 | 49.9 |

- **Abstract:**

```
Describing a video with accurate words and appropriate sentence patterns is interesting and challenging. Recently, excellent models have been proposed to generate smooth and semantically rich video descriptions. However, the language generally does not participate in encoding training, and different modalities including vision and language cannot be effectively interacted and accurately aligned. In this work, a novel model named MGTR-MISS which consists of multimodal interaction and semantic supervision, is proposed to generate more accurate and semantically rich video descriptions with the help of more ground truth. In detail, more external language knowledge is firstly retrieved from the ground truth corpus in the training set to capture richer linguistic semantics for the video. Then the visual features and retrieved linguistic features are fed into a multimodal interaction module to achieve effective interaction and accurate alignment between modalities. The output multimodal representation is then fed to a caption generator for language decoding with visual-textual attention and semantic supervision mechanisms. Experimental results on the popular MSVD, MSR-VTT and VATEX datasets show that our proposed MGTR-MISS outperforms not only the baseline model but also the recent state-of-the-art methods. Particularly, the CIDEr performances reach to 111.1 and 55.0 on MSVD and MSR-VTT respectively.
```

- **Introduction:**

```
Video description is a cross-modal task that aims at generating decent natural language descriptions for a given video. The task possesses promising potential application prospects in fields such as intelligent interaction systems and embodied intelligence. The task is interesting but challenging since the spatio-temporal semantics of the video have to be effectively modeled across hundreds of consecutive frames, and the alignment and correlation between vision and language must be explored and established. Nowadays, lots of effective models (Li et al., 2023, Tang et al., 2025, Zeng et al., 2025, Zhang et al., 2023, Zheng et al., 2020) are proposed for the task with the help of deep learning techniques, especially Transformer based sequence modeling mechanisms.
However, in the prior popular works (Chen and Jiang, 2019, Yu et al., 2016, Zheng et al., 2020) where “encoder–decoder” architecture is employed and 2D/3D CNN networks (Hara et al., 2018, Szegedy et al., 2017) are used to model the spatiotemporal motion representation, the temporal dynamic features of continuous motion in the video are usually difficult to be captured. The textual language corresponding to the visual information cannot be fully comprehended by the models, resulting in insufficient and confusing semantics of generated sentences. Additionally, the models in which visual guiding information such as object is employed to enhance the representative ability of the feature and further improve the semantics of generated descriptions (Zhang and Peng, 2019, Zhang et al., 2020a), mainly focus on visual information refining. This optimized visual information is often separately fed into a decoder, and the generated sentences are frequently shorter, and difficult to describe more details of a video. Then, the generated sentences are plagued by long-tail problems, especially a few rare but specific words related to visual content that are usually overlooked by the model during training. Furthermore, because the auto-regression method is usually employed for language decoding in current works, the semantic errors of vision and language are inevitably accumulated continuously, further limiting the generalization ability and accuracy improvement of generated sentences.
In human habits, rich external experience and knowledge are usually introduced into current descriptive task to capture more details of the video and align the language with visual content closely. We learn from the observation and propose a novel framework called MGTR-MISS for video description, where more linguistic information is retrieved for the video to be described from the training set in addition to the ground truths. As shown in Fig. 1, different from the previous works, the retrieval module is employed to get more external ground truth for video, and multimodal interaction and semantic supervision modules are proposed to accurately align different modalities and guide the model for full optimization.

The overview of our model is presented in Fig. 2. At the encoding stage, the vision and linguistic features are firstly extracted
from a given video and all descriptions in the training set with a
pre-trained model (e.g. CLIP), respectively. The visual features and
retrieved top 𝑘(𝑘 ≥ 1) features of references are integrated into multimodal representation. Then they are fed into an interactive encoder
to catch and align more comprehensive spatio-temporal information
of vision and language. For decoding, a dual decoding architecture
is constructed, which aims to establish stronger connections between
vision and language, enabling a seamless conversion from visual to
linguistic representations according to jointly training the two modal
branches. Furthermore, the retrieved language feature is utilized as
prior knowledge to perform semantic correction at each decoding
time step 𝑋𝑡
, which alleviates the problem of error accumulation in
autoregressive decoding. In contrast to the previous works (Luo et al.,
2023; Shi et al., 2023; Zhang et al., 2021) where external corpora
are employed as input sources, our proposed approach focuses more
on systematic interaction between different modalities and integrating
semantic knowledge throughout the entire training process rather than
isolated stages. The main contributions of this work are summarized as
follows:
• A multimodal interaction module is proposed to improve the
accuracy of semantic alignment between vision and language
based on external linguistic knowledge retrieving from the whole
training set. Specifically, the retrieved language features and
visual features participate in encoding and interaction together
during model training to achieve accurate multimodal alignment.
• A semantic supervisor module is developed to suppress error
accumulation during decoding and enhance the generalization
ability of the model. The retrieved best language features are
integrated to the hidden state of sequential decoder to guide the
model to generate sentences that are closer to the video content.
• Experiments are conducted on popular datasets like MSVD and
MSR-VTT, and the state-of-the-art performances are achieved
on multiple metrics compared to the similar popular methods,
demonstrating the effectiveness and superiority of our proposed
model.
```

- **Related work:**

```
2.1. Video description
As a pivotal subtask in the field of visual understanding, video description possesses a long research history and broad application prospects. In the early days, sentence templates or visual retrieval techniques are usually employed to complete the task. On one hand, in template-based model (Das et al., 2013, Guadarrama et al., 2013, Kojima et al., 2002, Rohrbach et al., 2013), visual concepts extracted from the video are mechanistically plugged into pre-defined text templates using classifiers. However, the fixed templates are frequently rigid and tend to compromise flexibility. On another hand, the retrieval-based methods (Farhadi et al., 2010, Gupta et al., 2012, Hodosh et al., 2013) directly employ statements that matched the given video from the retrieval space as descriptions. Although the method improves the fluency of generated sentences, the accuracy is often insufficient and limited by constraints in retrieval sample capacity.
Inspired by the encoder–decoder pipeline, recent researches (Chen and Jin, 2020, Chen et al., 2018, Pei et al., 2019, Venugopalan, Rohrbach, et al., 2015, Wang, Ma, et al., 2018, Yao et al., 2015) for video description encode video with deep models like convolutional neural networks (CNN) and decode the extracted visual features by sequential model such as recurrent neural networks (RNN) and long-short term memory (LSTM), to understand the video naturally and comprehensively. Venugopalan, Xu, et al. (2015) introduce CNN-RNN architecture to model video sequence for the context of video captioning, significantly improving the quality of generated sentences. Yan et al. (2019) propose a spatial–temporal attention mechanism within an encoder–decoder neural network for video description. Lei et al. (2020) propose a Transformer based variant where a particular memory mechanism is designed for highly generalized memory state from video clips and sentence history. However, the nuanced distinctions between various visual cues in video cannot be effectively captured in these works, which makes it difficult for generated sentences to accurately reflect the uniqueness of video content, leading to the long-tail problem. In this work, a novel approach in which the retrieval techniques with sequence learning is employed to provide richer external knowledge and semantic supervision for model learning, and generate distinctive and representative descriptions for video content.
2.2. Multimodal alignment and interaction
There is a huge semantic gap between modalities (e.g. vision vs language), but there is a corresponding relationship between different modal semantics. During translation from video to language description, the semantic alignment between different modalities is an issue that must be addressed. In recent years, a few researchers have diligently worked to integrate multimodal information into the model for video description, achieving noteworthy accomplishments. By incorporating modalities beyond raw video frame features, more nuanced information in video is explored and used for accurate word prediction. In the ORG-TRL model (Zhang et al., 2020a), object detectors are used to extract object modalities for visual representation enhancement. While in MA-LSTM (Xu, Yao, Zhang, & Mei, 2017), audio stream features of videos are introduced into the model for sequence learning and performance improvement. Besides, Shi et al. (2023) propose a video-text alignment module to learn multimodal aligned representations for video captioning.
However, the current generator only uses a method based on the current visual representation and the words it has generated to determine the source of newly predicted words during sentence generating, neglecting the interaction between vision and language and resulting in the inaccurate semantic alignment of different modalities. Also, the external corpus is often far from the current task, and hence it is generally only employed to fine-tune the model at the final stage. Facing the challenge, a multimodal interactive module is designed, where the visual features and retrieved external ground truth are integrated for temporal encoding and decoding during training, to achieve precise modal interaction and semantic alignment.
2.3. Retrieval knowledge augmentation
Retrieval technology has a long research history and a wide range of application fields. Specific to retrieval knowledge augmentation in the video description, it is to discover more relevant linguistic information for the video and facilitate mutual understanding and matching between the two modalities, and suppress the problem of visual illusion and improve the accuracy of generated sentences. In the previous video-text retrieval based models (Chen, Zhao, et al., 2020, Dzabraev et al., 2021, Zhang et al., 2018, Zhu and Yang, 2020), fusion architectures are usually designed to enable cross-modal learning. The following works that pre-trained models are employed for video-text retrieving, achieve remarkable performance improvement on video description. Particularly, the CLIP (Radford et al., 2021), which is trained on large-scale image-text pair dataset, shows significant superiority and robust capability on various vision-language learning tasks.
Subsequently, numerous models leverage the retrieval knowledge to augment language corpus and help performance breakthroughs on extensive downstream related tasks. The recent RCG (Zhang et al., 2021) employs a paradigm named open book video captioning, which incorporates relevant textual sentences from an external corpus to assist with language generation, breaking the limitation of relying solely on video content in the previous works. In the work of Shi et al. (2023), a retrieval unit is also developed to achieve video-text alignment, where the related sentences are retrieved as additional input and the semantic anchor between vision and language. Gu et al. (2023) employ knowledge graph to provide more semantic guidance for the model, making the generated sentences more precise.
However, most of these methods merely treat the acquired external textual knowledge as additional input, combining visual features into a unified representation through well-designed intermediate modules before sending it to the decoder for text generation and optimization. The decoder is forced to rely only on the lossy fused unified features, failing to fully leverage the enhancing effect of textual knowledge on the generation process. In this work, we propose a multimodal interaction module and a semantic supervisor module at the encoding and decoding stages, respectively, to facilitate modal interaction and dynamic adjustment. Additionally, a dual-decoder architecture is employed to allow various information sources (i.e. visual features, textual features, and unified features) to complement each other during model learning mutually, improving the model’s understanding ability of the video description task. By integrating text semantic knowledge in the whole training process, close modal cooperation is established, and the maximum fusion of different modal information is realized, thus improving the performance of the model.
Although there may be an extent of semantic bias in this approach, especially when the retrieved sentences may not be consistent with the actual visual content, the retrieved sentences have a strong correlation with the video, which can still provide multimodal interaction and serve as a semantic supervision role during sentence generation. At the same time, it also offers opportunities for improving the generalization ability of the model. We also learn from the practices and construct a CLIP based retriever as an integral component in our model to get richer external linguistic semantics for video to be described.
```

# [23-24][v] Memory-Based Augmentation Network for Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10183355

- **Published in:** IEEE Transactions on Multimedia ( Volume: 26)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.7 | 37.3 | 74.3 | 101.5 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 41.3 | 28.0 | 61.4 | 49.8|

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 32.7 | 22.4 | 49.1 | 48.9 |

- **Abstract:**

```
Video captioning focuses on generating natural language descriptions according to the video content. Existing works mainly explore this multimodal learning with the paired source video and corresponding sentence, which have achieved competitive performances. Nonetheless, learning from video-description pair cannot capture implicit external knowledge, i.e., multiple visual context information and linguistic clues existing in the video-language dataset, which may limit the cognitive capability of the model to generate diverse descriptions. To this end, we propose a Memory-based Augmentation Network (MAN), in which a memory structure is designed to augment the current encoder-decoder framework by incorporating implicit external knowledge with a neural memory. Specifically, we first propose a visual memory for the encoder to store multiple visual contexts across videos in the dataset, which is utilized to obtain memory-augmented contextual features for the source video. In addition, a textual memory is introduced for the decoder to capture the external language clues across sentences in the dataset. It is adapted to capture memory-augmented language features in each time step. The proposed approach is able to capture comprehensive contextual understanding compared to the basic encoder-decoder framework, which is more compatible with the human cognitive process. Extensive experiments on three video captioning datasets including MSVD, MSR-VTT, and VATEX demonstrate the effectiveness of the proposed method.
```

- **Introduction:**

```
Video captioning aims to generate a sequence of words to describe the visual content of the video. As a rapidly developing technique, it has received increasing attention [1], [2], [3] due to the explosion of uploaded video data in social media. There have been extensive applications for video captioning including visually impaired assistance [4], [5], [6], human-robot interaction [7], [8], [9], video indexing [10], [11], [12], and event detection [13]. Compared with its twin “image captioning” [14], [15] only dealing with static spatial information, video captioning tends to be more challenging since it involves both consecutive spatial and temporal representations.

Most existing methods for video ing follow the encoder-decoder framework [16], [17], [18], [19], which employs CNNs as the encoder to analyze visual semantics and extract useful visual context features from the source video, and utilizes RNNs as the decoder to generate the sequentially. To improve the performance of video ing, the attention mechanism is introduced in recent studies [16], [17], [20], [21], which is able to focus on the relevant visual content selectively. For example, the work [16] attempts to focus on learning the temporal relation between video frames by an LSTM-based decoder. And some methods [22], [23] learn spatial relations by exploiting semantic correlations and constraints between objects. There are several works [1], [24], [25] propose to extract the spatial-temporal representation using various attention mechanisms. Most recently, transformer-based architecture attracts extensive attention due to its powerful multimodal alignment ability [26], [27], [28].

Although these methods have achieved remarkable development, they only focus on one source video and its corresponding sentence for multimodal learning. This limits the existing methods in the superficial association only from the single input video- pair without effectively capturing implicit external knowledge, i.e., multiple visual context information and language clues existing in the video-language dataset. Typically, video as an image sequence contains dynamic scenarios and rich visual contexts such as movement, object, scene, and their interactions. It is difficult to use a single visual or language representation to capture all this information over a long-term sequence. For instance, as shown in Fig. 1, the basic encoder-decoder model cannot fully understand the visual information only from the source video and generate diverse words ‘band’ and ‘performing on stage’ due to the insufficient visual contextual details and language clues.

Recently, a memory-based neural model has been proposed and proven its effectiveness on the question answering task [29], [30], [31]. It shows a great advantage to modeling long-term semantic dependency in sequential problems. Moreover, some works [32], [33], [34] also explain that memory bank plays an important role in visual search which provide contextual visual information for plausible scenario understanding. Thus, implicitly modeling visual and textual memories for video ing can not only assist the language model (e.g., LSTM) in solving the long-term dependency with additional language clues but also provide rich information for visual context understanding.

In this article, we propose a novel framework for video ing, namely Memory-based Augmentation Network (MAN), where a memory structure is built to store implicit external knowledge from the video-language dataset and enhance the current encoder-decoder architecture. Specifically, we first introduce a visual memory for the encoder to provide visual context information. In practice, we introduce a set of learnable vectors to collect multiple visual contexts across videos in the dataset. For each source video, contextual-aware cross attention is adopted to obtain video-related contextual features by selecting key learnable vectors from visual memory guided by the source video. In addition, we then propose a symmetric textual memory for the decoder part, in which several learnable vectors are designed to capture language clues across sentences in the dataset. Particularly, for the current token, we propose linguistic-aware cross attention to select the most relevant language clues from textual memory guided by the hidden state of LSTM in the current time step. To this end, the proposed visual and textual memories can provide more multimodal context information, which is helpful to augment the basic encoder-decoder architecture. Fig. 1 shows that our model can successfully understand the object ‘band’, the corresponding action ‘performing’, and the background ‘on stage’ in the source video because of the various relevant information stored in the visual and text memories. To evaluate the effectiveness of our proposed method, we conduct extensive experiments and perform deep analysis on three video ing benchmark datasets: MSVD, MSR-VTT, and VATEX. The experimental results demonstrate that our method can achieve better performance with a simple yet effective memory structure.

To summarize, the contributions of this article are as follows:

We present a novel framework, designated as Memory-based Augmentation Network (MAN) for video ing task. Unlike previous memory-based methods which adopt memory learning with massive external knowledge, our approach extracts contextual semantics within datasets and stores latent contextual information into trainable memory structure as implicit knowledge for sentence augmentation.

We disentangle the visual and linguistic contextual learning with two separate memories (i.e., visual memory and textual memory). Without pre-storing a large volume of prior data in the memory module, our parameterized memories instead utilize a few learnable vectors to capture contextual information. This allows for optimization along with the model, freeing it from the complex memory updating procedures and maintaining less computational overhead.

Extensive experiments on three widely used datasets, i.e., MSVD, MSR-VTT, and VATEX demonstrate the effectiveness of the proposed method. Moreover, we conduct transferring experiments on image ing tasks, which shows the great generalization capability of our proposed MAN framework.
```

- **Related work:**

```
A. Video Captioning
Video ing is a key task in the multi-modal domain, which has received extensive attention and made rapid development in recent years. Traditional video ing approaches are mainly template-based language models [19], [35], [36], [37], [38], [39], [40]. These approaches predefine a set of language templates for generation, which limits the diversities of generated sentences due to the strong dependence on the predefined template. Recently, with the development of deep learning and neural machine translation, the encoder-decoder framework was introduced [41] and many sequence learning-based models [17], [28], [42], [43] were proposed to deal with the problems on video ing. These methods constructed the encoder-decoder structures to directly generate sentences from the video content. The work [41] generated video descriptions by LSTM with mean pooled video representation overall frame features, ignoring the temporal information. To address this issue, [16] proposed a temporal attention mechanism to dynamically select video frames. Another method [44] learned the different characteristics of video frames by capturing the regions of interest in each frame to get better video ing. In [43], it proposed a hierarchical modular network to reduce the representative gap between visual and linguistic views from three levels. As for [2], it presented the idea of guiding spatial attention through optical flow, which described the dynamics of behavior by calculating the pixel-by-pixel displacement between two consecutive video frames. To improve the quality, there were some breakthrough researches [24], [45], [46], [47], [48]. The work [45] constructed the temporal trajectory of an object by adopting a bidirectional temporal map on the object from video frames. For [24], it proposed to extract the spatial-temporal representation at the trajectory level by using the attention mechanism. And the approach [46] designed an evaluator to pick the best from multiple candidate s. The most recent research [49], [50], [51], [52] aimed to explore external knowledge to enhance quality. For instance, [49] proposed a mixed experts model to compose different experts based on different topic embeddings for knowledge transfer from seen activities to unseen ones. Further, the work [50] introduced a video-text retriever to select video-related sentences from the offline corpus containing the whole sentences of the training set. And [52] devised a retrieval mechanism that enhances the video-sentence pairs with retrieved explicit reference.

Most of these methods cannot effectively explore the relevant information appearing in similar videos and sentences in the datasets. Moreover, compared to the methods which introduce external knowledge from the large-scale corpus (e.g., Wikipedia [53] and WordNet [54]), our proposed approach is able to capture implicit knowledge including multiple visual contexts and external language clues existed in the dataset with a flexible memory structure, which is more in line with the human cognitive process.

B. Memory Network
To enhance the memory capacity of traditional neural networks, [55] proposed a Neural Turing Machine (NTM) with an external memory used by the attention mechanism. NTM had shown a superior potential for both storage and accessing long-term information. Besides the memory matrix in NTM, memory was also modeled as a differentiable list, a queue, or a deque [56], [57]. Memory networks were first proposed by modeling static memory to store long-term information with the purpose of rectifying the drawback of limited memory [58], [59]. The internal information stored in the static memory was not modified by external controllers, which were especially used for reading comprehension. These memory networks had been successfully applied to various long-term dependency reasoning tasks, such as machine translation [60], textual question answering [30], and visual question answering [29]. In addition, [61] introduced the memory networks into ing tasks by proposing an external multimodal memory to simultaneously interact with video and sentence. The approach in [62] proposed to augment image ing with an offline memory bank that consisted of billions of image-text pairs. To eliminate the tremendous pre-defined data, the work [63] used an external memory module to model the previous history of video segments and sentences in each transformer layer. In this article, we disentangle the vision and text's contextual learning into two separate memories (i.e., visual memory and textual memory), and apply them to video ing. Compared with the previous memory-based methods which adopt memory learning with substantial prior knowledge (e.g., pre-defined billions of visual-language features) or complex updating structures, the proposed memory mechanism contains trivial learnable vectors for implicit knowledge capture and can be well-trained with gradient descent as model training.

C. Image Captioning
Image ing is another fundamental multimodal task, which aims to generate a natural language description for an image, only dealing with static spatial information. In the early stage, retrieval-based [64], [65] and template-based [66], [67] methods were mainly applied to image ing. With the great progress in deep learning, RNN-based encoder-decoder significantly improved machine translation [68]. Subsequently, there were some attempts [69], [70] to directly employ the basic RNN-based encoder-decoder scheme for image ing, where CNN was used as the encoder to extract visual features from an image and RNN as the decoder to generate the sentence. After that, a series of innovations for image captioning had been proposed by exploring more interactions between visual and lingual via the attention mechanism. The work [14] presented soft and hard attention mechanisms into an LSTM-based decoder, which was able to select the most relevant image regions for word prediction. And there was a work [15] proposed an adaptive attention mechanism to dynamically decide whether to attend image regions when generating each word. Furthermore, [71] employed bottom-up and top-down attention mechanism that enabled attention measurement at the object level. Encouraged by the breakthrough in NLP field via Transformer [72], [73], the Transformer-based encoder-decoder scheme was studied in the image captioning field, which strengthened the visual encoding and vision-language interaction with self-attention or cross-attention mechanism. For example, [74] applied the primary Transformer structure in NLP in image captioning. Work such as [75] proposed a spatial and scale-aware Transformer to prevent the loss of spatial and fine-grained semantic information. In this article, our proposed method is applied to the task of image captioning to verify its effectiveness.
```

# [23-23][?] Concept-Aware Video Captioning: Describing Videos With Effective Prior Information

- **Link:** https://ieeexplore.ieee.org/document/10233200

- **Published in:** IEEE Transactions on Image Processing ( Volume: 32)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 56.3 | 39.1 | 75.6 | 106.9 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 48.7 | 31.5 | 65.2 | 59.7|

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 37.5 | 25.1 | 52.4 | 63.1|

- **Abstract:**

```
Concepts, a collective term for meaningful words that correspond to objects, actions, and attributes, can act as an intermediary for video captioning. While many efforts have been made to augment video captioning with concepts, most methods suffer from limited precision of concept detection and insufficient utilization of concepts, which could provide caption generation with inaccurate and inadequate prior information. Considering these issues, we propose a Concept-awARE video captioning framework (CARE) to facilitate plausible caption generation. Based on the encoder-decoder structure, CARE detects concepts precisely via multimodal-driven concept detection (MCD) and offers sufficient prior information to caption generation by global-local semantic guidance (G-LSG). Specifically, we implement MCD by leveraging video-to-text retrieval and the multimedia nature of videos. To achieve G-LSG, given the concept probabilities predicted by MCD, we weight and aggregate concepts to mine the video’s latent topic to affect decoding globally and devise a simple yet efficient hybrid attention module to exploit concepts and video content to impact decoding locally. Finally, to develop CARE, we emphasize on the knowledge transfer of a contrastive vision-language pre-trained model (i.e., CLIP) in terms of visual understanding and video-to-text retrieval. With the multi-role CLIP, CARE can outperform CLIP-based strong video captioning baselines with affordable extra parameter and inference latency costs. Extensive experiments on MSVD, MSR-VTT, and VATEX datasets demonstrate the versatility of our approach for different encoder-decoder networks and the superiority of CARE against state-of-the-art methods. Our code is available at https://github.com/yangbang18/CARE.
```

- **Introduction:**

```
Video captioning aims to depict video content with natural language automatically. As a test-bed to bridge computer vision and natural language processing, video captioning has potential applications in many real-world scenarios, such as blind navigation, video subtitling, and video retrieval.

In the past few years, neural network-based methods have become dominant in video captioning [1], [2], [3], [4], [5] and widely adopted the encoder-decoder framework, where the encoder is to represent video content with compact features while the decoder is for caption generation. One of the subsequent improvements to this nascent framework is to provide caption generation with prior information, such as sentence-relevant syntax [6], [7], [8] and video-content-related concepts [9], [10], [11], [12]. This paper pursues the latter research, i.e., concept-augmented video captioning. Generally speaking, concepts are a collective term for meaningful words that correspond to objects, actions, and attributes in video content. Concepts not only manifest as high-level semantics of video content but also serve as a bridge between video and text modalities.

There are two main concerns to augmenting video captioning with concepts, i.e., how to detect concepts precisely and how to capitalize on them. For concept detection, prevailing methods derive supervision signals from the original video-text annotations of a target dataset and design dedicated detection networks [11], [12], [13], [14], [15], [16], [17]. However, most practices rely on visual-driven concept detection, which limits the detection precision. Taking Fig. 1 as an example, we can not confirm the exact action of the little girl solely from the visual cues because she may be talking or singing. For concept utilization, mainstream studies exploit concepts to either affect the global preference of caption decoders or adaptively influence word prediction at each time step but rarely explore their joint impacts. These limitations in concept-augmented video captioning approaches may result in inaccurate and inadequate prior information to caption generation and thus lead to a sub-optimal solution.

In this paper we propose a Concept-awARE video captioning framework (CARE) to facilitate plausible caption generation. Based on the encoder-decoder structure, CARE detects concepts precisely via multimodal-driven concept detection (MCD) and offers sufficient prior information to caption generation by global-local semantic guidance (G-LSG). Specifically, we implement MCD by leveraging video-to-text retrieval and the multimedia nature of videos. As shown in Fig. 1, taking textual and visual-audio cues of a video as input, MCD has two merits. (1) The retrieved text may contain phrases indirectly associated with the video content (e.g., “shocks the judges”), which is conducive to enhancing the associative power of detection networks. (2) Using visual-audio information for concept detection is helpful for disambiguation (e.g., identifying the girl’s action). To achieve G-LSG, given the concept probabilities predicted by MCD, we weight and aggregate concepts to mine the video’s latent topic to affect decoding globally and devise a simple yet efficient hybrid attention module to exploit concepts and video content to impact decoding locally. The core idea of G-LSG comes from the observations that (1) one would not use sports-related words to describe a music video, and (2) concepts can be used as references when predicting words at each time step. Notably, our CARE is agnostic to the encoder-decoder architecture, making it versatile for different video captioning networks.

Besides framework design, the “pre-training and fine-tuning” paradigm also matters for the challenging video captioning task. While label-supervised pre-trained models (PTMs) have become ubiquitous in visual understanding, contrastive vision-language PTMs like CLIP [18] draw more and more attention in vision-language tasks [19], [20], [21], [22], [23]. In this paper, we shed new light on transferring the knowledge of CLIP-like PTMs to video captioning by fueling CARE with CLIP. Specifically, we use CLIP’s image encoder as a part of video encoding and utilize the full CLIP model to achieve the video-to-text retrieval requested by MCD. With the multi-role CLIP, our CARE can outperform CLIP-based strong video captioning baselines with affordable extra parameter and inference latency costs.

To summarize, we make the following contributions. (1) We propose a novel Concept-awARE video captioning framework (CARE) that detects concepts precisely via multimodal-driven concept detection and offers adequate prior information to caption generation by global-local semantic guidance. (2) We devise a simple yet efficient hybrid attention module that enables the flexible fusion of concepts and video content to guide word prediction at each time step. (3) We shed new light on transferring the knowledge of CLIP-like pre-trained models to video captioning by fully leveraging the potential of CLIP in visual understanding and video-to-text retrieval to develop CARE. (4) Extensive experiments on MSVD [24], MSR-VTT [25], and VATEX [26] show that CARE adapts well to various encoder-decoder networks and surpasses strong video captioning baselines and state-of-the-art methods.
```

- **Related work:**

```
We briefly review neural network-based video captioning and then introduce concept-augmented and retrieval-enhanced visual captioning and contrastive vision-language pre-training.

A. Video Captioning
The encoder-decoder framework has risen to prominence in video captioning since the early work [1]. Later improvements to this nascent framework mainly focus on four aspects. The first aspect is to acquire a holistic view of video content, which is usually achieved by exploiting multi-modalities [3], [27], extracting hierarchical representations [28], [29], and learning fine-grained local or semantic features [9], [30], [31]. The second aspect is to model video-text interactions and bridge video-text gap via attention mechanisms [32], [33], [34] or high-level semantics [6], [7], [8], [9], [10], [11], [12]. The third aspect is to generate captions in a controllable or flexible way [35], [36], whereas the fourth is to explore the knowledge transfer of large-scale pre-trained models [4], [37]. This paper is closely related to the second and the fourth aspects.

B. Concept-Augmented Visual Captioning
Identifying concepts in an image or a video has attracted enormous research interest. For concept-augmented visual captioning, pioneering work is [38], where You et al. select a fixed set of concepts from the image captions in training data, design a dedicated network for concept detection, and use detected concepts to guide caption generation via semantic attention. While concept selection in [38] has become a common practice in later works, the way to exploit detected concepts varies considerably, e.g., input enhancement [39], [40], [41], hidden state initialization [12], [40], [41], cross-modal attention [2], [15], [17], and semantic composition [11], [13], [27]. Compared with static images, videos are multimedia data. So differences also exist in concept detection inputs for video captioning. Unlike prevailing visual-driven concept detection, Xu et al. [42] propose to detect video concepts with audio-augmented features; Chen et al. [27] predict topic information of videos based on their visual-acoustic content; Sun et al. [13] additionally use optical flow to improve detection precision. Similar to [13] and [42], this paper also emphasizes multimodal-driven concept detection. But the core difference is that we consider textual cues, which are crucial supplements and can be obtained at a negligible latency cost (c.f., time-consuming extraction of optical flow). Besides, we focus on multi-faceted concept utilization that is rarely explored and propose a more efficient hybrid attention module compared with the previous ones [2], [15], [17].

C. Retrieval-Enhanced Visual Captioning
Different from regarding visual captioning as a retrieval task [43], retrieval-enhanced visual captioning aims to generate accurate captions with the aid of retrieval. With such spirit, there are many related methods in image captioning, such as the unification of retrieval- and generation-based captioning under adversarial training [44], [45], [46] and the use of cross-modal retrieval to assist domain adaption [47] or compensate missing information in object features [19]. There have been few related works for retrieval-enhanced video captioning [48], [49] due to the challenges of dynamic scene changes. Zhang et al. [48] introduce an open-book paradigm, where video-content-related sentences are retrieved first and then used as references when producing captions with a copy-mechanism generator. Chen et al. [49] condense the contextual knowledge of retrieved video-caption pairs into a key-value memory, from which the decoder can read to improve word prediction. Instead, this paper adopts video-to-text retrieval to facilitate accurate concept detection to extract more comprehensive high-level semantics.

D. Contrastive Vision-Language Pre-training (CVLP)
Inspired by the success of contrastive learning in learning general visual representations [50], [51], [52] and the superiority of Transformer-based architectures in large-scale pre-training [53], [54], [55], CVLP has rapidly developed in recent years to learn transferable visual representations from natural language supervisions [18], [56], [57], [58]. CVLP aims to pull multimodal views of the matched data closer while repelling views of the mismatched data apart. Fueled by millions of noisy image-text pairs crawled from the Internet, representative works of CVLP such as CLIP [18] and ALIGN [56] demonstrate an impressive zero-shot transfer ability in various vision tasks. Most recently, some works propose to apply CLIP to different vision-language tasks [19], [20], [21], [22], [59]. For example, the works [20], [22], [59] employ CLIP for image encoding to endow caption models with more power. Luo et al. [21] show that CLIP can benefit video-text retrieval via knowledge transferring. Different from previous works, we focus the potential of CLIP in both visual understanding and cross-modal retrieval, which enables a maximum use of CLIP’s internal knowledge.
```

# [24-24][v] Exploring the Role of Audio in Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10678632, https://arxiv.org/pdf/2306.12559

- **Published in:** 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 46.4 | 30.2 | 64.1 | 57.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 39.1 | 26.3 | 53.4 | 73.7 |

- **Abstract:**

```
Recent focus in video captioning has been on designing
architectures that can consume both video and text modalities, and using large-scale video datasets with text transcripts for pre-training, such as HowTo100M. Though these
approaches have achieved significant improvement, the audio modality is often ignored in video captioning. In this
work, we present an audio-visual framework, which aims
to fully exploit the potential of the audio modality for captioning. Instead of relying on text transcripts extracted via
automatic speech recognition (ASR), we argue that learning
with raw audio signals can be more beneficial, as audio has
additional information including acoustic events, speaker
identity, etc. Our contributions are twofold. First, we observed that the model overspecializes to the audio modality
when pre-training with both video and audio modality, since
the ground truth (i.e., text transcripts) can be solely predicted using audio. We proposed a Modality Balanced Pretraining (MBP) loss to mitigate this issue and significantly
improve the performance on downstream tasks. Second, we
slice and dice different design choices of the cross-modal
module, which may become an information bottleneck and
generate inferior results. We proposed new local-global fusion mechanisms to improve information exchange across
audio and video. We demonstrate significant improvements
by leveraging the audio modality on four datasets, and even
outperform the state of the art on some metrics without relying on the text modality as the input.
```

- **Introduction:**

```
Large-scale pre-training [3, 20, 33, 43, 50, 59, 66] plays
a key role in boosting the performance of modern deep
learning models. It is even more so for vision and language tasks, such as video captioning[1, 12, 36, 41, 58, 65],
where leveraging large video datasets with text supervision
for pre-training is essential to achieve competitive results.

However, manually annotating captions for video datasets
is costly and not scalable. Thus existing video captioning
datasets [49, 61, 63, 69] are often limited in size.
To address this challenge, recent work collected datasets
from instructional videos, where ASR generated transcripts
can be used as text supervision, e.g., How2 [46], CrossTask
[72], HowTo100M [34], HD-VILA-100M [64], etc. This
has established a new trend of pre-training on large-scale
video datasets with text transcripts for video captioning [16,
32, 47]. We argue that text transcripts from ASR only
includes partial information from audio, and hypothesize
end-to-end learning using the audio modality can potentially lead to better performance, since audio can provide
additional information (shown in Fig. 1) including acoustic
events, speaker identity, etc.
More specifically, our paper seeks to better understand
the following questions:
• To what extent, can the audio modality improve video
captioning?
• How can the potential of the audio modality be fully
realized in an audio-visual framework for captioning?
To this end, we start with a simple multi-modal pre-training
framework for video captioning with ASR transcripts as supervision (shown in Fig. 2), and look into different components that may hinder the performance of the pre-trained
audio-visual model on the downstream datasets.
First, we observed that simply jointly training of the audio and video modalities may result in degenerated models
that overspecialize to the audio modality and underfit on
the video modality. As text transcripts are used as video
captions during pre-training, the model essentially learns to
cheat and solve the ASR problem instead of extracting information from both visual and audio signals. To mitigate
this issue, we proposed the Modality Balanced Pre-training
(MBP) loss, which takes into account both the unimodal
losses and cross-modal loss. A weighting mechanism is introduced to automatically balance different modalities during training. Fig. 4 shows that our MBP loss enforces the
model to pay more attention to the underfitted video modality and drives the final loss much smaller.

Second, we did a thorough study on the design of the
cross-modal fusion module, which is responsible for the information exchange between the audio and video modality. An improperly designed cross-modal fusion module
may become an information bottleneck and result in inferior performance for video captioning. We proposed new
local-global fusion mechanisms to encourage the information flow cross different modalities. We analyzed the relevance of the annotated captions to the audio modality on
downstream datasets, and observed that the local fusion
mechanisms are more beneficial to the flow of fine-grained
information like single words in speech, while the global
fusion mechanisms are more effective on holistic information like acoustic events or scenes. The local-global design
enables the fusion module to capture information at different granularities, and mingle audio and video information at
different levels. Compared with existing designs, our localglobal fusion has shown empirically better results.
By combining the two contributions, we demonstrate
that audio is crucial to video captioning and provides both
speech and non-speech information. Fig. 1 shows a few examples on how our model effectively integrates the information from both the audio and video modality, and generates
better captions than video-only and video-text variants.
We summarize our contributions as follows:
• Proposed to pre-train video captioning models based
on video and audio modalities, explored the role of audio in video captioning, and demonstrated the benefits
of audio on four benchmarks.
• Proposed the MBP loss to balance different modalities automatically during training, and ease the issue
of overspecialization to the audio modality.
• Did an extensive evaluation on the effects of different cross-modal fusion modules on audio-visual video
captioning, and proposed a novel local-global fusion
module to effectively integrate audio and video information for video captioning.
```

- **Related work:**

```
Video Captioning. Most works in video captioning [1, 12,
36, 41, 58, 65] focus on designing a better model to generate text descriptions given precomputed video features via
an encoder-decoder framework. SwinBert [29] attempted
to train the encoder-decoder framework directly from raw
video pixels instead of refining precomputed features. In
addition to visual modality, some works studied video captioning from visual data and ASR texts [15, 32, 47, 48, 49,
53]. A few prior works also studied audio-visual video captioning [7, 17, 45, 54], but they are often limited to smallscale video captioning datasets and precomputed input features. To the best of our knowledge, we propose the first
end-to-end audio-visual video captioning framework.
Multi-Modal Pre-training. A growing number of works
are investigating multi-modal pre-training in videos, e.g.,
video-text pre-training [24, 32, 33, 34, 40, 50, 67] and
video-text-audio pre-training [3, 4, 66], which mostly adopt
contrastive learning and/or masked language modeling to
learn better representations for downstream tasks. As only
encoders are trained for multiple modalities, a separate decoder needs to be trained on top of the encoders for generative tasks such as video captioning. A recent work MVGPT [47] shows the benefits of pre-training an end-to-end
encoder-decoder framework to video captioning. Unlike
MV-GPT that relies on ASR text as input, our framework
directly uses video and audio. A Textless Vision-Language
Transformer (TVLT) [52] was recently proposed to take visual and audio inputs for multi-modal representation learning without ASR inputs. However, the pre-trained TVLT
is a discriminative model that cannot be directly applied
to generative tasks. While a multi-modal network receives
more information and is expected to boost performance, recent works [18, 37, 42, 60] have identified a key challenge in training a multi-modal network that one modality may converge faster than other modalities and undermine the representation learning of other modalities. We propose a Modality Balanced Pre-training objective to mitigate this issue and
facilitate a powerful audio-visual video captioning model.
Cross-Modal Fusion. Given the representations of multiple modalities, a cross-modal fusion module [19, 33, 34, 43]
will fuse these representations into a shared embedding
space to generate cross-modal representations. In order to
fuse a sequence of representations generated by Transformers [56], there are two major types of cross-modal fusion
modules: merged fusion, and cross fusion. In merged fusion, the two modalities are concatenated and then fed into a
Transformer block [25, 27, 32, 59]. In cross fusion, the two
modalities are fed into different Transformer blocks separately, where cross attention Transformers are used to allow
cross-modal interaction [31, 47, 48, 51, 55]. Besides, some
recent works propose variants of cross-modal fusion modules that use bottleneck tokens [35] or prune single-modal
units [62] to control the flow of cross-modal interaction.
```

# [25-25][idea] Pretrained Image-Text Models are Secretly Video Captioners

- **Link:** https://aclanthology.org/2025.naacl-short.26/

- **Published in:** Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Developing video captioning models is computationally expensive. The dynamic nature
of video also complicates the design of multimodal models that can effectively caption these
sequences. However, we find that by using
minimal computational resources and without
complex modifications to address video dynamics, an image-based model can be repurposed
to outperform several specialised video captioning systems. Our adapted model demonstrates top-tier performance on major benchmarks, ranking 2nd on MSR-VTT and MSVD,
and 3rd on VATEX. We transform it into a
competitive video captioner by post-training
a typical image captioning model BLIP-2 with
only 6,000 video-text pairs and simply concatenating frames—significantly fewer data than
other methods, which use 2.5 to 144 million
pairs. From a resource optimization perspective, this video captioning study focuses on
three fundamental factors: optimizing model
scale, maximizing data efficiency, and incorporating reinforcement learning. This extensive
study demonstrates that a lightweight, imagebased adaptation strategy can rival state-of-theart video captioning systems, offering a practical solution for low-resource scenarios.
```

- **Introduction:**

```
Vision-language pretraining significantly advances
multimodal tasks such as captioning, question answering, retrieval and broader video understanding. Among these, video captioning stands out as it narrates visual concepts and
their temporal interactions, reflecting the intricate multimodal processes as humans to perceive and
articulate dynamic visual experiences.
Current video-text methods often incorporate
intricate designs tailored to video inputs. For instance, some models extend existing frameworks
by integrating frame samplers to capture temporal
dynamics (Alayrac et al., 2022; Yang et al., 2021;
Xu et al., 2021). Other approaches, such as ALPRO (Li et al., 2022a) and VIOLET (Fu et al.,
2023), propose end-to-end models that are meticulously trained on large-scale video-text datasets
sourced from the Web (Zellers et al., 2021; Bain
et al., 2021). Despite their success, video captioning models remain highly resource-intensive, often
hitting performance bottlenecks when (i) computational resources are constrained, or (ii) the task
requires specialized priors without clear guidance
for model design and training. This raises a critical
question: for simplicity and efficiency, how can
we repurpose existing image captioning models
for video captioning, without relying on complex, hand-crafted video-specific designs?
To address this, we revisit fundamental factors
in training—model scale, data efficiency, and supervision—that critically influence video captioning while being agnostic to the variants of videospecific designs: First, we find that moderate-sized
language models (LMs) when fine-tuned for specific tasks, can meet the demands of video captioning efficiently. This challenges the common belief
that larger models are always superior, demonstrating that targeted optimization can outperform sheer
model size. Second, using extensive pretraining on
image-text pairs, as demonstrated with BLIP-2, is
transferable to video tasks. This allows the model
to achieve high performance with minimal video usage, offering an efficient alternative to training from
scratch. Third, instead of relying on traditional
cross-entropy loss, we optimize directly for nondifferentiable CIDEr with reinforcement learning,
ensuring that the generated captions better align with human-standard video descriptions.
By bypassing complex, specialized video input
designs, our experiments demonstrate that BLIP-2
straightforwardly derived from image captioning,
can be effectively optimized to deliver competitive video captioning performance. This study
underscores the potential of simplicity and efficiency in advancing multimodal video captioning, providing a streamlined yet stable solution.

**Recycling BLIP-2 for Video Captioning**
As shown in Fig. 1, we adapt BLIP-2, a typical
image-text model (details in App. B), for video
captioning without any additional parameters. Each
video frame is encoded by ViT, which generates
visual tokens that are concatenated to form a unified
representation (e.g., an 8-frame video produces a
token sequence of size 8×256). This unified token
sequence is then processed by the Q-former and
passed to the LM to generate captions.
```

- **Related work:**

```
Image-Text Models Large-scale pretraining has
revolutionized the field of image-text models, enabling significant advances. Models such as
CoCa (Yu et al., 2022) and SimVLM (Wang et al.,
2022b), which are trained from scratch on billions
of image-text pairs, have set new benchmarks in
generative tasks such as open-ended visual question
answering (VQA) and visual captioning. BLIP-2
addresses the computational demands of pretraining from scratch by reusing existing pre-trained parameters from Vision Transformer (ViT) and LLMs
and integrating them with a frozen pre-trained state.

A key innovation in BLIP-2 is the introduction of
the Q-former connector, carefully designed to enhance the interaction between visual and language
modalities (Li et al., 2023b). This methodology has
inspired subsequent innovations in visual-lingual
tuning, with newer models often incorporating
the pre-trained Q-former alongside the eva-vit-g
model from BLIP-2, demonstrating the lasting impact of this methodology (Dai et al., 2023b; Zhu
et al., 2024; Yang et al., 2024; Li et al., 2023c).
Video-Text Models Video-text models typically
extend the capabilities of image-text models by
integrating temporal feature aggregation to capture dynamic content, as exemplified by VideoCoCa (Yan et al., 2022). In addition, specialized models such as Video-LLaMA enhance the
processing of temporal dynamics by embedding
multiple temporal Q-former layers, facilitating nuanced interactions across modalities. Such advances refine the synergy between video Q-formers
and LLMs within the model architecture, building on the foundation of BLIP-2 (Zhang et al.,
2023). Building on these developments, recent
studies, including VideoChat, PandaGPT, Valley,
and Video-ChatGPT, investigate the embedding of
frozen LLMs into video LMs, pushing the boundaries of the field (Li et al., 2023c; Su et al., 2023;
Luo et al., 2023; Muhammad Maaz and Khan,
2023). In our study, we use BLIP-2 as a basic
model for captioning, first pre-trained on images
and then adapted to video by incorporating a video
frame merging mechanism that effectively captures
temporal nuances. This simplicity allows us to focus on evaluating the effects of model size, data
volume, and training strategies on video captioning
performance as we scale.
Difference between Image and Video Captioning The fundamental difference between image
and video annotation stems from their source inputs: image annotation processes a single static image, while video annotation requires an understanding of the temporal dynamics over a sequence of
frames. When adapted to video, pre-trained image
models such as GIT (Wang et al., 2022a), VideoCoCa (Yan et al., 2022), and IcoCap (Liang et al.,
2023) show remarkable adaptability to video with
only moderate modifications, demonstrating their
transferability. Conversely, video-specific models,
including Video-LLaMA (Zhang et al., 2023) and
VideoChat (Li et al., 2023c), use different sampling techniques to effectively capture temporal
dynamics. Furthermore, models such as ALPRO
(Li et al., 2022a) and VIOLET (Fu et al., 2023)
utilize extensive web-crawled datasets to achieve
end-to-end training, enriching their learning process. In our study, instead of emulating the complex
adaptations typical of specialized video models, we
adopt a streamlined approach that uses averaging or
concatenation to merge temporal information from
sampled video frames. This method allows us to
focus on evaluating the effects of model size, data
volume, and training strategies on video captioning
performance as we scale.
```

# [23-24][v] Learning Hierarchical Modular Networks for Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10296527

- **Published in:** IEEE Transactions on Pattern Analysis and Machine Intelligence ( Volume: 46, Issue: 2, February 2024)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.9 | 36.2 | 74.2 | 104.7|

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 45.1 | 28.8 | 63.6 | 54.2 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 35.3 | 23.1 | 50.9 | 58.8 |

- **Abstract:**

```
Video captioning aims to generate natural language descriptions for a given video clip. Existing methods mainly focus on end-to-end representation learning via word-by-word comparison between predicted captions and ground-truth texts. Although significant progress has been made, such supervised approaches neglect semantic alignment between visual and linguistic entities, which may negatively affect the generated captions. In this work, we propose a hierarchical modular network to bridge video representations and linguistic semantics at four granularities before generating captions: entity, verb, predicate, and sentence. Each level is implemented by one module to embed corresponding semantics into video representations. Additionally, we present a reinforcement learning module based on the scene graph of captions to better measure sentence similarity. Extensive experimental results show that the proposed method performs favorably against the state-of-the-art models on three widely-used benchmark datasets, including microsoft research video description corpus (MSVD), MSR-video to text (MSR-VTT), and video-and-TEXt (VATEX).
```

- **Introduction:**

```
Automatically describing videos in natural language, also known as video captioning, has attracted considerable attention in recent years. Video captioning plays a crucial role in numerous applications, such as disability assistance, video retrieval, and human-computer interaction [1], [2], [3], [4], [5], [6], [7]. Although significant progress has been made, it remains a challenging problem due to the rich and large diversity in video contents, such as dozens of objects appearing in a video but only two or three of them being mentioned in captions.

Existing methods can be broadly grouped into two categories. The first category focuses on developing effective encoders in order to learn better video representations [8], [9], [10], [11], [12], [13], [14]. For example, spatial and temporal relations between objects in a video have been exploited via graphs to learn rich representations [9], [10]. In addition, temporal dynamics and gate fusion networks have been exploited for learning richer expressions for video captioning [11], [12]. While these methods perform well on video captioning, the optimization objectives are formulated in the word-by-word fashion as a caption is generated, which neglects the relations between video representations and the linguistic counterparts. The second category focuses on narrowing the semantic gap between video representation and linguistic counterparts before generating captions [15], [16], [17]. Pan et al. [15] propose to learn the alignment between the global embedding of a video and the representation of a whole caption. In [16], [17], [18], [19], nouns and verbs are associated with visual representations to explore the video-language correspondence on a fine-grained level. Empirical demonstrations show that this line of approaches is able to generate more accurate captions. However, these methods either focus on global sentence or local word correspondence, which fail to take advantage of alignments at different granularities.

To address the above-mentioned issues, we propose a hierarchical modular network (HMN) [20] to directly align visual representations with textual counterparts at three granularities: entity, predicate, and sentence (see Fig. 1). The entities highlight visual objects that are most likely to be mentioned in a caption and are supervised by entities1 in the caption. The entities usually serve as the cornerstone of a video caption, which can be the subjects or objects of an action as well as modifiers. In language, a predicate refers to the part of a sentence or clause containing a verb and stating something about the subject. In the context of video captioning, the predicates describe the action representations conditioned on highlighted objects according to captions. These predicates help reduce the correspondence errors from a multi-meaning verb to a specific video action embedding, such as the play in playing soccer and playing the piano. The sentence alignment captures the global view of the scene according to the entire caption, which enables the generated caption to have a reasonable meaning. With the help of these three visual-textual alignments, each of which is implemented as a module, our model improves the caption quality significantly.

While HMN has been shown to perform well against state-of-the-art approaches, there remain several issues to be addressed. First, the fusion within visual action representations and the fusion across visual action and object representations are not well modeled by a simple BiLSTM in the predicate module of HMN. Second, the predicate module may learn a strong entity-action coupling (e.g., always predicting “ride” when the entity is a “bike”, but the ground truth can be “push”). Third, similar to existing approaches, HMN adopts word-by-word supervision for generated captions that cannot reflect the similarity between similar but not exactly the same expressions, such as “a woman on the beach next to a man” versus “a woman next to a man on the beach”.

In this paper, we improve HMN from three aspects. First, we extensively explore fusion schemes within and across visual action and object representations using different techniques, including transformers, (Bi)LSTMs, and their combination. Second, we devise an independent verb module as a complementary component to the predicate module, aiming to increase the flexibility of generating verbs. Third, we design a scene-graph-based reward to regulate our model via reinforcement learning, which is able to measure sentence similarity with less interference from the word order. Equipped with these novel designs, the proposed method performs favorably against the state-of-the-art approaches on three widely used benchmark datasets.

The contributions of this paper are summarized below:

We propose a hierarchical modular network to learn multi-level visual representations at four granularities before generating captions by associating them with their linguistic counterparts: entity, verb, predicate, and sentence.

We propose a transformer-based entity module to learn to select principal objects that are most likely to be mentioned in captions.

To better measure sentence similarity, we propose a reinforcement learning module where the scene-graph-based award is designed to encourage the generated captions to be semantically closer to their ground-truth with a larger tolerance to word orders. Our method performs favorably against the state-of-the-art models on three widely-used benchmarks: MSR-video to text (MSR-VTT), video-and-TEXt (VATEX) and microsoft research video description corpus (MSVD).
```

- **Related work:**

```
Templates and Deep Models: Recent years have witnessed significant advances in video captioning using classic and deep learning approaches. Caption templates [21] and [22] have been used to solve this problem at the early stage. These methods first generate words for objects and actions and then fit these words into predefined sentence templates to obtain complete captions. Despite the demonstrated success, one major limitation is that they cannot generate captions with flexible sentence patterns. In recent years, numerous methods based on convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have been developed to facilitate flexible video captioning [2], [4], [7]. Venugopalan et al. [1] propose a model based on long short-term memory (LSTM) [23] to sequentially generate words of a caption using CNN features of each frame as input. As video representation plays a key role in this task, Yao et al. [6] develop a temporal attention mechanism to model the global temporal structure in a video, which aggregates relevant video segments according to the state of a text-generating RNN. In addition to image and motion features, audio features are also utilized to enrich video representations [24], [25]. To find useful information from a large number of inputs, Chen et al. [5] present a PickNet model to select informative frames from a video, which can also reduce computational costs in later processes. On the other hand, Wang et al. [26] and Pei et al. [27] utilize memory networks to effectively organize numerous visual features leading to better captioning quality. Existing approaches mainly focus on designing complex video encoders to learn effective visual representation in an implicit manner via gradient back-propagation of the loss regarding the generated captions. In contrast, we propose to learn video representations via directly aligning the embeddings of video and captions.

Object-Centric Video Caption: Physical entities usually serve as subjects or objects in a caption, which plays crucial roles in captions. Much attention has been paid to exploiting objects in the scenes for caption generation. In [13], Zhang et al. propose to capture object dynamics from temporal trajectories via a GRU [28] module. Aafaq et al. [12] exploit object labels to enhance the semantics of visual representation. On the other hand, Zheng et al. [17] model the interactions between objects via an attention mechanism. In [9] and [10], relationships among objects are modeled by graph convolutional networks [29] to enhance object-level representation. These approaches can generate captions well when detailed video information is mined. However, one potential problem is that all detected objects are used, but in reality, only a small set of them are mentioned in captions. As such, the quality of the generated captions may be noisy. Different from these methods, we propose an entity module to highlight principal objects that are most likely to be mentioned in a caption, thereby reducing both the amount of noise caused by unimportant objects and the computational load of subsequent processes.

Transformers for Captioning: Numerous transformer models have been developed and applied to numerous natural language processing and vision problems [30], [31], [32], [33], [34], [35], [36], [37], [38], [39]. Similarly, transformers have also been explored in the video captioning task. In [9], two transformers are used as language decoders from object information and scene information. On the other hand, Li et al. [40] design a global gated graph reasoning module to replace the self-attention mechanism to capture short-term spatial relations of objects and long-term transformation dependencies of objects. Different from these methods and motivated in part by DETR [41], we propose a transformer-based entity module to predict principal objects from a large number of entities detected in a video.

Scene Graph for Captioning: Numerous captioning methods [42], [43], [44] have recently been developed based on scene graphs. In [42], the spatial location of objects and human-object-interaction labels are used to build scene graphs. In contrast, Zhong et al. [43] select important sub-graphs from the scene graph of the input image, and decoding each selected sub-graph into a single target sentence. Graph representations are enhanced by meta concepts for video captioning in [44]. We note that existing methods build scene graphs on the visual side to enhance visual representations for later caption decoders. In contrast, our scene graphs are built from the textual side, i.e., the generated captions and ground-truth captions, which are used to measure the similarity and thus as a loss to constrain the model.

Reinforcement Learning for Captioning: Video captioning methods based on reinforcement learning (RL) have been developed to effectively alleviate both the exposure bias [45] and non-differentiable loss [46], [47], [48], [49], [50], [51], [52]. In [11] and [51], the CIDEr [53] score of generated captions is directly used to optimize the captioning model. Differently, Liu et al. [46] design a context-aware policy to take contextual visual attention changes into consideration. Rennie et al. [48] propose an RL method, which alleviates the computational complexity of estimating the rewards and normalization by utilizing the output of its own test-time inference module. In [52], Zhang et al. cast the video captioning process as a language dependency tree generation procedure, and propose a tree-structured reinforcement learning algorithm to optimize the captioning model. Unlike these approaches, which compute rewards based on n-grams (n consecutive words), we devise a scene-graph-based reward to better describe video captions, such as the relationships among the objects and the attributes of objects. This design facilitates skipping some unimportant words in n-grams.

Large Models: Inspired by the success of image-text pre-training such as CLIP [54], some works have investigated video-text pre-training and achieved significant success [55], [56], [57]. UniVL [55] proposes to learn video and text representation on large-scale narrated videos with four self-supervised tasks, including masked language modeling, masked frame modeling, video-text alignment, and language reconstruction. On top of the success of CLIP, CLIP4Caption [56] leverages a video-text matching task to further optimize its text and visual encoders. In [57], a video event boundary prediction task is proposed together with textual description generation, aiming to enable the model to understand events in videos better. LAVENDER [58] proposes to unify multiple video-text tasks (e.g., text-to-video retrieval, video question answering, and video captioning) via using the same masked language modeling as the common interface for all pre-training and downstream tasks. On the other hand, large language models, such as InstructGPT [59], have been utilized in a zero-shot fashion to tackle the video captioning task. VidIL [60] utilizes image-language models (CLIP and BLIP [61]) to convert video content into frame-level captions and then uses a few prompts with InstructGPT to generate video captions. In [62], soft prompts are learned with a frozen GPT-2 [63] model to generate video-frame-directed captions. Due to the lack of in-domain training, zero-shot methods perform far behind traditional captioning methods. When using the same pre-trained CLIP feature, our method can achieve favorable performance over pre-training methods UniVL and CLIP4Caption.
```

# [26-26][v][EMKG] Towards generalized video captioning: An effective multi-modal knowledge graph perspective

- **Link:** https://www.sciencedirect.com/science/article/pii/S0950705126003102

- **Published in:** Knowledge-Based Systems, Volume 340, 12 May 2026, 115568

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 59.5 | 40.4 | 78.0 | 116.4 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 48.5 | 31.5 | 65.1 | 61.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
The complexity of video content, combined with the long-tailed distribution of training data, results in suboptimal performance of existing video captioning models in cross-dataset and novel scenarios. This limitation significantly hinders their generalization ability. To tackle these issues, we propose an Effective Multi-modal Knowledge Graph (EMKG) framework for video captioning, which incorporates Cross-modal Fine-grained Adaptive Fusion (CmFAF) and Hardest Sample Attribute-anchored Alignment (HSA-Aligner) to efficiently integrate knowledge into the model. Specifically, we adopt an object-centric multi-modal knowledge orchestration approach to construct ConceptVision Knowledge Graph (CVKG) composed of two subgraphs: ConceptCore (C3) and VisionVivid (V3). C3 integrates commonsense knowledge as well as dataset-specific information, whereas V3 covers video-related knowledge, such as visual objects and attributes. Before training, we utilize the pre-trained CLIP model to assess the similarity between video frames and words of different parts of speech, such as nouns and verbs, retrieving a content-relevant subgraph to serve as input to the model. To minimize the noise in the retrieved subgraph, we design a CmFAF module. CmFAF leverages the cross-modal context generated by fusing global context with captions retrieved from the training corpus to dynamically adjust the weights of nodes and relationships at a fine-grained level. In addition, to improve the alignment between visual objects and textual entities of CVKG, we present HSA-Aligner. By utilizing object attributes and strategically selecting hardest samples, HSA-Aligner boosts the alignment performance. Extensive experiments on the MSR-VTT and MSVD demonstrate that our proposed EMKG framework markedly surpasses the state-of-the-art methods while exhibiting superior generalization ability.
```

- **Introduction:**

```
Video captioning (VC), a machine learning-based technique, automatically generates natural language descriptions of video content. It has broad applications in diverse fields, including assistive technologies for individuals with disabilities [1], human-computer and human-robot interaction [2], [3], sports or movie commentary [4], [5], [6], [7] and data annotation [8].
Existing learning-based VC approaches are predominantly built upon encoder-decoder architectures [9], [10], [11], [12], [13], with the subsequent introduction of the Transformer [14] significantly enhancing model performance [15], [16], [17], [18]. Although these methods have shown commendable results in intra-dataset testing scenarios, they suffer from performance degradation when applied to cross-dataset testing or novel, rare scenarios. One primary reason is the oversight of the long-tailed word distribution, where certain objects or entities appear infrequently [19]. Consequently, video captioning models struggle to sufficiently learn cross-modal mappings for these long-tailed terms during training. This deficiency leads to inaccurate descriptions in cross-dataset testing or novel scenarios, ultimately limiting the models’ generalization. To address this challenge, methods based on knowledge or prior information have been proposed to enhance model performance. These methods can be categorized into two strategies. (1)Mining internal knowledge: This approach employs techniques such as object detection [20], [21], [22], retrieval [23], [24], topic modeling [25], and token distribution analysis [26]. Although these approaches may increase the generation probability of certain long-tailed words, they fail to endow the model with a commonsense understanding of videos, which may lead to reduced adaptability and limited generalization when faced with distribution shifts during testing. (2) Leveraging external knowledge: This strategy aims to equip the model with a deeper understanding of video content by incorporating extensive external knowledge, such as commonsense [19], [27] in ConceptNet [28] and teacher-model knowledge [21]. The aforementioned methods provide the model with textual commonsense knowledge and some information about long-tailed words [19], [27], but lack explicit modeling of visual information. Due to the complex and diverse nature of video content, models must not only extract internal knowledge and learn textual associations but also, more importantly, establish mappings between visual and textual modalities. A knowledge graph based solely on textual modality fails to establish a reliable mapping between text and visual information. As a result, captioning models struggle to accurately describe objects or actions in novel scenarios or rare video content.
Multi-modal knowledge graphs (MMKGs) inherently provide explicit relationships among different modalities and serve as bridges connecting diverse data types. For example, they enable the establishment of explicit relationships between textual and visual data. Moreover, the knowledge within these graphs can be flexibly organized within large-scale knowledge bases and extensive datasets, rather than being limited to a few training samples. By constructing a comprehensive multi-modal knowledge graph and leveraging its embedded knowledge, such as commonsense, concepts, appearances, attributes, and action types related to video targets, the video captioning model can enhance its understanding of video content. This is especially beneficial when processing out-of-domain videos or handling rare objects and actions. Ultimately, it is expected to generate more accurate and higher-quality textual descriptions while improving generalization performance, particularly in cross-domain scenarios. In the context of entity-aware news image captioning [28], [29], MMKGs have been preliminarily validated for enhancing identity recognition accuracy. However, their potential remains largely untapped in the more complex domain of video captioning, particularly in terms of generalization [30]. Therefore, leveraging MMKGs as a promising solution to enhance the generalization capability of video captioning models warrants further exploration and investigation.
In this paper, we propose a novel video captioning framework that enhances model generalization by efficiently injecting knowledge from MMKGs into the decoder through adaptive fusion and alignment, as illustrated in Fig. 1c. Specifically, we construct a multi-modal knowledge graph, termed ConceptVision Knowledge Graph (CVKG), which comprises two subgraphs: ConceptCore (C3) and VisionVivid (V3). C3 subgraph incorporates commonsense knowledge from ConceptNet[28] along with dataset-specific syntactic parsing to endow the model with both general and domain-specific knowledge. In contrast, V3 subgraph captures video-related knowledge, such as object visual features and attributes, thereby providing fine-grained cross-modal mappings. Moreover, we propose a hybrid query retrieval method that leverages a pretrained CLIP model to separately measure the similarity between video frames and noun or verb queries. Based on the computed similarities, we retrieve subgraphs from the CVKG that contain the most relevant noun or verb queries.

To maximize the utility of MMKGs and enhance model generalization, we propose the Effective Multi-modal Knowledge Graph (EMKG) framework, as depicted in Fig. 2. This framework comprises two core modules: the Cross-modal Fine-grained Adaptive Fusion (CmFAF) module and the Hardest Sample Attribute-Anchored Alignment (HSA-Aligner) module. CmFAF adaptively integrates CVKG by dynamically refining the weights of nodes and relationships at a fine-grained level, leveraging cross-modal context derived from global context and retrieved captions. This mechanism mitigates interference from irrelevant entities and relationships, thereby ensuring the incorporation of high-quality knowledge into the model. HSA-Aligner facilitates the alignment of multi-modal information within the knowledge graph, enabling the establishment of robust cross-modal relationships. Given that an object can exhibit multiple attributes (e.g., a brush may be blue or white), directly aligning its visual and textual representations may introduce ambiguity. To mitigate this issue, we incorporate object attributes into the textual representations, thereby refining the cross-modal alignment at a more granular level. Furthermore, to improve alignment effectiveness, we identify and utilize the hardest visual-text pairs. This increases the difficulty of the alignment learning process and strengthens the model’s ability to associate multi-modal representations with greater precision.

The contributions of this paper can be summarized as four-fold:
•
We propose the effective multi-modal knowledge graph framework, which explores the generalization capability of video captioning models from the perspective of efficiently leveraging multi-modal knowledge graphs.
•
We construct a multi-modal ConceptVision knowledge graph (CVKG) using an object-centric multi-modal knowledge orchestration approach. This approach integrates diverse forms of knowledge, including commonsense, objects, attributes, and conceptual information.
•
We propose the Cross-modal Fine-grained Adaptive Fusion (CmFAF), which dynamically adjusts the importance of subgraph components using cross-modal context. Additionally, we propose the Hardest Sample Attribute-Anchored Alignment (HSA-Aligner) to refine cross-modal alignment by utilizing attributes and the hardest sample mining.
•
Experimental results demonstrate that our approach surpasses state-of-the-art methods by 4.7% and 2.0% in absolute CIDEr scores on the widely used MSVD and MSR-VTT, respectively. Furthermore, it exhibits competitive performance in cross-dataset evaluations and models trained on partial training sets.
```

- **Related work:**

```
The proposed method leverages MMKGs to achieve robust video captioning. In this context, we first review current deep learning-based video captioning approaches, including end-to-end models and those employing frozen, pre-trained feature extractors. Subsequently, we discuss the advances of knowledge graphs in video captioning and summarize the current research status of multi-modal knowledge graphs.
2.1. Video captioning
Recent encoder-decoder-based video captioning methods have demonstrated substantial performance improvements and can be broadly classified into two categories. The first category includes models that are trained end-to-end [18], [31], [32], where all parameters, including those of the visual encoder, are optimized during the training process. Typically, the visual encoder is pre-trained on large datasets [33], [34], which feature a high number of parameters and robust feature extraction capabilities. The second category employs frozen pre-trained feature extractors to obtain frame-level features [24], [26], [35], [36], [37] as well as motion features [20], [21], [22], [25], [38], among others. Enhancements in these methods primarily focus on improving modality alignment capabilities [38], [39], [40] optimizing network architectures, and extracting effective prior knowledge. This prior knowledge encompasses concepts [24], [31], [41], retrievals [23], [42], [43], optical flow [18], sentiment [44], object detection [20], [21], [22], and topics [45]. Such information can serve as an intermediary in VC, enhancing the accuracy of generated captions with respect to concepts, sentiment, topics, and other attributes. Although these methods improve video captioning quality from multiple perspectives, they scarcely address model generalization. Since the features and prior information extracted are mostly derived from the video itself, the amount of learnable information is inherently limited. Moreover, constraints such as dataset distribution, size, and long-tailed issues hinder the establishment of comprehensive cross-modal mappings, resulting in poor generalization in rare scenarios and cross-dataset evaluations.
In contrast, we construct a multi-modal knowledge graph that explicitly defines relationships between various types of knowledge and modalities, thereby equipping the model with rich general and domain-specific contextual knowledge, as well as cross-modal mappings. This facilitates a more comprehensive understanding of the video and ultimately improves performance in complex scenarios.
2.2. Knowledge graph
KGs provide rich, structured factual knowledge and play a pivotal role in advancing various artificial intelligence applications, such as healthcare [46], [47], law [48], and customer service question answering [42]. In the task of image captioning, early studies on knowledge graph-based captioning can be traced back to the use of ConceptNet [28], [49], which supplies commonsense knowledge to support model training [29]. Similar efforts have employed more advanced architectures to harness the potential of KGs [19], [50], [51]. In practice, KGs are frequently utilized for entity-aware image captioning [52], [53], as they offer extensive supplementary knowledge. For instance, commonsense knowledge is divided into relevant and explanatory knowledge [52] to improve the entity and commonsense distributions in the final caption, thereby enhancing caption accuracy and quality. In the task of VC, TextKG [19] integrates dataset-specific knowledge with that extracted from ConceptNet, retrieves and selects pertinent information, and then feeds it into a dual-stream decoder to generate textual descriptions.
The aforementioned methods rely solely on text-based knowledge graphs to provide structural knowledge for the model. However, they fail to offer cross-modal insights. In fact, mappings between different modalities, such as visual and textual modalities, are particularly crucial for VC tasks. Moreover, their coarse utilization may introduce noise [19], potentially misleading the model and causing error accumulation, ultimately negatively impacting its performance. Consequently, when encountering rare objects or scenarios, the limitations of a solely textual knowledge graph and its coarse utilization may lead to erroneous captions [52]. In contrast to previous approaches, we construct a multi-modal knowledge graph and employ adaptive fusion to mitigate noise interference, thereby providing the model with comprehensive, high-quality knowledge for video understanding.
2.3. Multi-modal knowledge graph
MMKGs [54], [55] integrate information from multiple modalities, including images, videos, and text, to create comprehensive representations of entities and their relationships. Recent research on multi-modal knowledge graphs can be broadly summarized into follow aspects: construction, embedding, completion, reasoning and application. Furthermore, there is ongoing work on integrating graphs with large language models to alleviate some of their inherent limitations. In the following, we primarily introduce the construction, embedding and application aspects relevant to our work.
Construction of multi-modal knowledge graphs is typically tailored to specific domains. For example, MMPKUBase [56] for the Chinese language, which includes birds, mammals, ferns, and more, comprises over 50,000 entities and more than 1 million filtered images. VHAKG [57], focusing on daily activities, represents the content of daily life videos as event-centric knowledge and also captures frame-by-frame fine-grained changes, such as bounding boxes within video frames. AspectMMKG [58], which captures the multi-aspect nature of entities, is the first multi-modal knowledge graph to incorporate aspect-related images by associating images with specific entity aspects. ImgFact [59], which emphasizes the multi-modal representation of relations, grounds triplet facts in images by capturing both entities and their relationships. However, constructing these multi-modal knowledge graphs (MMKGs) is highly resource-intensive or heavily dependent on search engines, and they are primarily image-oriented, making them unsuitable for direct application to video captioning (VC) tasks. To address this, we propose an Object-Centric Multi-modal Knowledge Orchestration approach to efficiently construct a multi-modal knowledge graph, CVKG. CVKG incorporates diverse multi-modal knowledge, including objects and their attributes extracted using pre-trained object detectors, as well as extensive content-related commonsense knowledge and dataset-specific knowledge obtained through syntactic parsing.
Embeddings of multi-modal knowledge graphs focus on mapping entities, relations, and their associated multi-modal information (e.g., text, images, audio, video) into a low-dimensional continuous vector space. Existing approaches often use pre-trained models to extract modal information and subsequently map this information into a unified representation space. IKRL [60] utilizes VGG [61] to extract visual information from entities’ images and scores a triple by incorporating both visual and structural information using TransE [62]. However, this method is limited to using either textual or visual information, but not both simultaneously. To address this limitation, TransAE [63] employs TransE as the scoring function and leverages a multi-modal autoencoder to extract modal information. [64] and [65] use VGG [61] and GloVe [66] to separately extract visual and textual information, which are then fused into a unified multi-modal representation. APKGC [67] introduces a noise-enhanced approach with an attention penalty module, and AdaMKGC [68] addresses modal imbalances by integrating adaptive modality interactions. More recent works have applied decoupling methods [69], [70] to better capture rich semantics, either by enhancing the handling of semantic noise [71]. In KG-based captioning models such as TextKG [19], a GloVe model is used to extract the linguistic features of nodes. Instead of this approach, we leverage MMKGs by utilizing the powerful CLIP model to extract visual object features while assigning learnable embeddings to entities (including visual attributes) and relations. Additionally, we adaptively adjust the importance of entities and relations to enhance representation learning. Furthermore, we design HAS-Aligner to align different modalities, effectively mitigating modality discrepancies.
Recent applications of multi-modal knowledge graphs include recommendation systems [72], [73], as well as applications in the medical field such as medical conversational question answering [74], protein function and structure prediction [75], and representations of medical concepts [76]. NEGCL [77] leverages innovative contrastive learning and data augmentation to enhance multi-modal recommendation systems. MMKDGAT [72] integrates various attributes and visual information from remote sensing images and employs a deep relational attention mechanism for information aggregation, thereby improving active recommendation accuracy and addressing the cold-start recommendation problem. RECipe [73], a multipurpose recipe recommendation framework, addresses zero-shot inference for new users (i.e., the cold-start problem) as well as conditional recommendations with respect to recipe categories. LingYi [74] constructed a Chinese medical multi-modal knowledge graph (CM3KG) and compiled a large-scale Chinese medical CQA dataset (CMCQA), a system that helps alleviate physician workload and improve medical efficiency. ProteinKG65 [76] is a knowledge graph specifically built for protein science to enhance protein function and structure prediction. Multi-modal knowledge graph-enhanced entity-aware image captioning [53], in contrast to single-modality knowledge graphs [52], leverages MMKGs to provide cross-modal mapping relations, thereby addressing the long-tailed distribution of identity information. However, these knowledge graphs are difficult to apply directly to VC and cannot adequately accommodate the complex and variable content of videos.
To the best of our knowledge, our method is the first to explore the generalization capabilities of MMKGs in video captioning. Compared to the aforementioned approaches, constructing MMKGs and developing models for videos is more challenging due to the complexity and dynamic nature of videos. Videos contain vast amounts of information and numerous objects, making it difficult to establish cross-modal relationships. To address this, we leverage extensive knowledge gathered through data collection and parsing to train an efficient video captioning model based on MMKGs, which incorporates both adaptive fusion and alignment techniques.
```

# [23-24][v][BTKG] Bidirectional transformer with knowledge graph for video captioning

- **Link:** 

- **Published in:** Multimedia Tools and Applications

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 55.7 | 38.3 | 74.7 | 104.5 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 42.8 | 30.0 | 62.4 | 55.4 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Models based on transformer architecture have risen to prominence for video captioning. However, most models are only to improve either the encoder or the decoder, because when we improve the encoder and decoder simultaneously, the shortcomings of either side may be amplified. Based on the transformer architecture, we connect a bidirectional decoder and an encoder that integrates fine-grained spatio-temporal features, objects, and relationships between the objects in the video. Experiments show that improvements in the encoder amplify the information leakage of the bidirectional decoder and further produce a worse result. To tackle this problem, we generate pseudo reverse captions and propose a Bidirectional Transformer with Knowledge Graph (BTKG), which integrates the outputs of two encoders into the forward and backward decoders of the bidirectional decoder, respectively. In addition, we make fine-grained improvements on the interior of the different encoders according to four modal features of the video. Experiments on two mainstream benchmark datasets, i.e., MSVD and MSR-VTT, demonstrate the effectiveness of BTKG, which achieves state-of-theart performance in significant metrics. Moreover, the sentences generated by BTKG contain scene words and modifiers, that are more in line with human language habits.
```

- **Introduction:**

```
The task of video captioning aims to understand the scenes in a video and describe them with plausible sentences. It can be widely used in video retrieval, video recommendation, disabled supporting, and scene understanding. With the rapid development of deep learning, the encoder-decoder-based neural caption methods [1–4] have risen to prominence for video captioning. Especially with the emergence of Transformer [5], models based on transformer architecture [6–11] have been successful in advancing the state-of-the-art. However, the above models only improve either the encoder or the decoder, e.g., by integrating objects in the video and relationships between the objects into the encoder [7], by fusing audio, image, and motion features of the video into the encoder [9], and by employing a bidirectional decoder [8]. Therefore, the motivation for simultaneously improving the encoder and decoder of Transformer is natural, e.g., the decoder generates caption with higher accuracy through a bidirectional decoder while fusing more modal features of the video into the encoder. As shown in MEV of Fig.1, Zhang et al. [7] used the state-of-the-art 2-D CNN and 3D CNN pre-trained on a large dataset to extract visual spatio-temporal features [12]. They also acquired the relationships between objects extracted based on the object detector Mask R-CNN [13] and the knowledge graph model TransE [14]. Existing unidirectional decoders generate the target language sequence token-by-token from left to right, and they cannot make full use of the target-side future contexts, which can be produced in a right-to-left decoding direction [15]. Therefore, Wang et al. [8] proposed a bidirectional decoder to address the above problem. As shown in BDD of Fig. 1, the bidirectional decoder is composed of a Forward Decoder (FD) and a Backward Decoder (BD), because the forward decoder accept the reverse captions generated by the backward decoder, the forward decoder can considers the previously generated outputs and the following words of the ground-truth caption from the backward decoder. Nevertheless, the bidirectional decoder exists a problem of information leakage [16] . The leakage is that in the training stage, the forward decoder indirectly utilizes the reverse ground-truth caption, which is the input from the backward decoder. On the contrary, the forward decoder employs the reverse caption generated by the backward decoder in the testing stage. Therefore, if the backward decoder generates higher quality reverse caption, it is equivalent to that the forward decoder uses caption closer to the reverse ground-truth caption in the testing stage, i.e., as long as the more outstanding the performance of the backward decoder is, the less negligible negative impact of the leakage on the bidirectional decoder is. As shown in ODRA of Fig. 1, we acquire the objects (’car’, ’people’) from an object detector Mask R-CNN. Since objects are composed of a series of isolated entities and no reverse order concept between contexts, it is not very sensible for us to reverse them. The same goes for the relationships obtained from a knowledge graph model TransE. Therefore, if we integrate the objects and the relationships into the backward decoder, the quality of generating reverse caption may be reduced and further expand the leakage. As shown in Fig. 1, when we directly connect a bidirectional decoder and an encoder that fuses the objects and the relationships, experiments show that it does amplify the leakage. Therefore, we propose a Bidirectional Transformer with Knowledge Graph (BTKG) to mitigate the leakage. In Fig. 2, we respectively assign one encoder to each of the forward and backward decoders of the bidirectional decoder, which is different from the conventional model with only one encoder and decoder. Since the encoder integrating the objects and the relationships can reduce the generation quality of reverse caption, as shown in STE of Fig. 2, we remove the objects and relationships for the encoder connected to the backward decoder. In addition, we generate pseudo reverse captions to reduce the leakage by exploiting the characteristic of having multiple various captions for a video [8].
Moreover, different from the conventional encoder of the Transformer for sequential texts, the inputs of our encoders are frames, objects, and relationships in the video. Therefore, as shown in Fig. 2, we employ different encoders according to the various modals of the video. Because frames in the video have tighter correlations than words in a text, we first capture the correlations between smaller video clips than frames through an extended and multihead attention layer in STE. Secondly, each relationship is independent, so there is no need to employ an attention layer for capturing the correlations between them. Finally, since neither the objects nor the relationships are contextually ordered, neither utilizes a positional encoding. The main contributions are summarized in the following four aspects:  – To utilize the bidirectional decoder when integrating objects in a video and relationships between the objects into the encoder, we propose a Bidirectional Transformer with Knowledge Graph (BTKG). It allows two different encoders to connect the forward and backward decoders, respectively. – The inputs of our encoders are modal features of the video, which is different from the Transformer for sequential text. Therefore, we improve the encoders according to the different modal features. 
– To mitigate the information leakage of bidirectional decoder, we generate pseudo reverse captions. In addition, the extended and multi-head attention layer indirectly improves the performance of the encoder and the backward decoder and further alleviates the leakage. 
– We evaluate BTKG on two datasets, MSVD and MSR-VTT. In particular, our BTKG achieves state-of-the-art performance and outperforms the runner-up methods by a large margin in CIDEr-D, which is specially designed for captioning tasks. Moreover, the sentences generated by BTKG contain scene words and modifiers, that are more in line with human language habits.
```

- **Related work:**

```
2.1 Video captioning  Recently, captioning a short video in natural language has been challenging for machines. With the rapid development in deep learning, Donahue et al. are the first to adopt a deep neural network to solve the video captioning problem [1]. Later, many video captioning methods based on encoder-decoder architecture rose to prominence [1–4]. These methods encode the video by a Convolutional Neural Network (CNN) [17] and employ a Long ShortTerm Memory (LSTM) [18] to generate video captions. One of the first works that utilize an encoder-decoder framework is Venugopalan et al., where captions are generated by LSTM and visual features extracted by CNN [2]. With the emergence of the attention mechanism and Transformer [5], Zhou et al. also propose a Transformer method to utilize an attention mechanism to generate coherent captions [6]. Zhong et al. propose a three-layer hierarchical attention network based on a bidirectional decoding transformer that enhances multimodal features [19].  
2.2 Multimodal extraction  In order to exploit the temporal structure of the video, the utilization of multi-modalities in video encoding also attracted great attention. For instance, Song et al. extend the importance of implicitly distinguishing visual and non-visual words through a hierarchical attention mechanism [20]. In particular, Aafaq et al. leverage the state-of-the-art 2-D CNN and 3-D CNN (C3D [21]) pre-trained on a large dataset to extract visual spatio-temporal features [12]. With the success of object detection in Computer Vision, the bottom-up attention algorithm applies object detection to extract regional features, significantly improving the video captioning performance [22]. Yan et al. extract the local features of maintaining more accurate object information via object detector [23]. Zhang et al. respectively utilize object detection and knowledge graph to extract objects and relationships between the objects in the video and then fuse them with spatio-temporal features of the video to refine the fine-grained actions between the objects [7]. Zhou et al. proposed a novel MATNet to introduce a new way to learn rich spatiotemporal object representations with an interleaved encoder, which encourages knowledge propagation from motion to appearance in a hierarchical manner [24].  
2.3 Captions generation  Recently, many studies have also made some achievements in captions generation. Liu et al. tackled the captioning problem of the controllable video by injecting robust control signals into the selected objects [25]. For revising word and grammar errors, Xu et al. propose DRPN to refine the caption candidates [26]. In order to capture more visually grounded details from videos, Yang et al. designed a special decoding algorithm to realize coarse-tofine rather than the caption procedure word-by-word [27]. The decoder of the Transformer is an autoregressive decoder, which only considers the currently generated words. Wang et al. propose a bidirectional decoding model for generating captions by using the context [8].  
2.4 Summary  As mentioned above, the above methods almost only improve the encoder or the decoder. Experiments show that direct superposition of the encoder and decoder based on [7] and [8] may produce a worse results. Inspired by these works, we propose BTKG based on transformer architecture to effectively fuse the improved encoder and decoder. The inputs of our encoders are frames, objects, and relationships in the video, which is different from the Transformer for sequential text. Based on the difference, we improve the encoders according to the various modal features of the video.
```

# [23-23][v][TextKG] Text with Knowledge Graph Augmented Transformer for Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10205180

- **Published in:** 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 60.8 | 38.5 | 75.1 | 105.2 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 43.7 | 29.6 | 62.4 | 52.4|

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
Video captioning aims to describe the content of videos using natural language. Although significant progress has been made, there is still much room to improve the performance for real-world applications, mainly due to the long-tail words challenge. In this paper, we propose a text with knowledge graph augmented transformer (TextKG) for video captioning. Notably, TextKG is a two-stream transformer, formed by the external stream and internal stream. The external stream is designed to absorb additional knowledge, which models the interactions between the additional knowledge, e.g., pre-built knowledge graph, and the builtin information of videos, e.g., the salient object regions, speech transcripts, and video captions, to mitigate the longtail words challenge. Meanwhile, the internal stream is designed to exploit the multi-modality information in videos (e.g., the appearance of video frames, speech transcripts, and video captions) to ensure the quality of caption results. In addition, the cross attention mechanism is also used in between the two streams for sharing information. In this way, the two streams can help each other for more accurate results. Extensive experiments conducted on four challenging video captioning datasets, i.e., YouCookII, ActivityNet Captions, MSR-VTT, and MSVD, demonstrate that the proposed method performs favorably against the state-of-theart methods. Specifically, the proposed TextKG method outperforms the best published results by improving 18.7% absolute CIDEr scores on the YouCookII dataset.
```

- **Introduction:**

```
Video captioning aims to generate a complete and natural sentence to describe video content, which attracts much attention in recent years. Generally, most existing methods [21, 38, 41, 58] require a large amount of paired video and description data for model training. Several datasets, such as YouCookII [69], and ActivityNet Captions [19] are constructed to promote the development of video captioning field. Meanwhile, some methods [29, 40, 48, 72] also use the large-scale narrated video dataset HowTo100M [30] to pretrain the captioning model to further improve the accuracy. Although significant progress has been witnessed, it is still a challenge for video captioning methods to be applied in real applications, mainly due to the long-tail issues of words. Most existing methods [29, 40, 48, 72] attempt to design powerful neural networks, trained on the large-scale video-text datasets to make the network learn the relations between video appearances and descriptions. However, it is pretty tough for the networks to accurately predict the objects, properties, or behaviors that are infrequently or never appearing in training data. Some methods [14, 71] attempt to use knowledge graph to exploit the relations between objects for long-tail challenge in image or video captioning, which produces promising results. In this paper, we present a text with knowledge graph augmented transformer (TextKG), which integrates additional knowledge in knowledge graph and exploits the multi-modality information in videos to mitigate the longtail words challenge. TextKG is a two-stream transformer, formed by the external stream and internal stream. The external stream is used to absorb additional knowledge to help mitigate long-tail words challenge by modeling the interactions between the additional knowledge in pre-built knowledge graph, and the built-in information of videos, such as the salient object regions in each frame, speech transcripts, and video captions. Specifically, the information is first retrieved from the pre-built knowledge graphs based on the detected salient objects. After that, we combine the features of the retrieved information, the appearance features of detected salient objects, the features of speech transcripts and captions, then feed them into the external stream of TextKG to model the interactions. The internal stream is designed to exploit the multi-modality information in videos, such as the appearance of video frames, speech transcripts and video captions, which can ensure the quality of caption results. To share information between two streams, the cross attention mechanism is introduced. In this way, the two streams can obtain the required modal information from each other for generating more accurate results. The architecture of the proposed method is shown in Figure 1. Several experiments conducted on four challenging datasets, i.e., YouCookII [69], ActivityNet Captions [19], MSR-VTT [56], and MSVD [3] demonstrate that the proposed method performs favorably against the state-of-theart methods. Notably, our TextKG method outperforms the best published results by improving 18.7% and 3.2% absolute CIDEr scores in the paragraph-level evalution mode on the YouCookII and Activity-Net Captions datasets.
```

- **Related work:**

```
Video captioning attracts much attention of researchers in recent years. The best practice has been achieved by attention-based methods, which attempts to associate visual components with sentences in videos. Some of them focus on designing powerful network architectures. VLM [55] and VideoBERT [43] take the visual and text modalities as input, and use a shared transformer to construct a taskagnostic video-language model. ViLBERT [28] processes visual and linguistic information separately with two parallel streams, and then use the attention mechanism to model the interactions between visual and language features. Instead of using the separate encoder-decoder architecture, MART [21] designs a shared encoder-decoder network and augments it with the memory module. ActBert [72] uses local regional features to learn better visual-language alignment. WLT [35] takes audio features as an additional input, and uses context fusion to generate multimodal features. Meanwhile, some methods [10,14,15,62,65,71] focus on exploiting prior knowledge to provide semantic correlations and constraints between objects for image or video captioning, producing promising results. ORG-TRL [64] uses the knowledge information in the language model (BERT) to provide candidate words for video captioning. In contrast, we propose a two-stream transformer for video captioning, with the internal stream used to exploit multi-modality information in videos, and the external stream used to model the interactions between the additional knowledge and the built-in information of videos. These two streams use the cross-attention mechanism to share information in different modalities for generating more accurate results.  Vision-and-language representation learning is a hot topic in recent years. ViLBERT [28], LXMERT [46], UNITER [6], UNIMO [25] and Unified-VL [68] learn  the representations between image and text, while Univl [29], VideoBERT [43], ActBERT [72] and MV-GPT [40] learn the representations between videos and transcripts. Notably, most of these methods attempt to learn powerful vision-and-language representations by pre-training the models on the large-scale datasets, e.g., Howto100M [31] and WebVid-2M [1], and then finetune them on downstream tasks such as video captioning, video-text retrieval and visual question answering. In contrast, our TextKG method uses the speech transcripts as the text to model the visual and linguistic representations and integrate the additional knowledge in knowledge graph to mitigate long-tail words challenge in video captioning. Knowledge graph in NLP. Knowledge graph is an useful tool to indicate the real-world entities and their relations, which provides rich structured knowledge facts for language modeling. Large-scale knowledge graphs are used to train knowledge enhanced language models for various natural language processing (NLP) tasks. CoLAKE [44] proposes to inject the knowledge context of an entity, and to jointly learn the contextualized representation for both language and knowledge by a unified structure. ERNIE [63] enhances BERT architecture to better integrate the knowledge information and textual information. KEPLER [52] not only improves language models by integrating factual knowledge, but also generates text-enhanced knowledge representation. JAKET [59] proposes a joint pre-training framework to model knowledge graph and language simultaneously. Inspired by CoLAKE, our method jointly learns the representations of vision, language and knowledge, and enhances the joint visual-language representations by retrieving relevant knowledge in knowledge graphs.
```

# [23-23][v] Accurate and Fast Compressed Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/10378206

- **Published in:** 2023 IEEE/CVF International Conference on Computer Vision (ICCV)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 60.1 | 41.4 | 78.2 | 121.5 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 44.4 | 30.3 | 63.4 | 57.2 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 35.8 | 25.3 | 52.0 | 64.8 |

- **Abstract:**

```
Existing video captioning approaches typically require to first sample video frames from a decoded video and then conduct a subsequent process (e.g., feature extraction and/or captioning model learning). In this pipeline, manual frame sampling may ignore key information in videos and thus degrade performance. Additionally, redundant information in the sampled frames may result in low efficiency in the inference of video captioning. Addressing this, we study video captioning from a different perspective in compressed domain, which brings multi-fold advantages over the existing pipeline: 1) Compared to raw images from the decoded video, the compressed video, consisting of I-frames, motion vectors and residuals, is highly distinguishable, which allows us to leverage the entire video for learning without manual sampling through a specialized model design; 2) The captioning model is more efficient in inference as smaller and less redundant information is processed. We propose a simple yet effective end-to-end transformer in the compressed domain for video captioning that enables learning from the compressed video for captioning. We show that even with a simple design, our method can achieve state-ofthe-art performance on different benchmarks while running almost 2× faster than existing approaches.
```

- **Introduction:**

```
Video captioning is a representative example of applying deep learning to the fields of computer vision and natural language processing with a long list of applications, such as blind navigation, video event commentary, and humancomputer interaction. To generate captions for a video, the model needs to not only identify objects and actions in the video, but also be able to express them accurately in natural language. Despite significant progress, accurate and fast video captioning remains a challenge.  Video captioning requires both 2D appearance information, which reflects the objects in the video, and 3D action information, which reflects the actions. The interaction between these two types of information is crucial for accurately captioning the actions of objects in the video. Most of the existing methods [36, 38, 22] are shown in Fig. 1 (the upper branch), mainly including the three-steps: (1) Decoding the video and densely sampling frames. (2) Extracting the 2D/3D features of the video frames offline. (3) Training the model based on these 2D/3D features. In these methods, densely sampled video frames result in significant redundancy, which in turn increases the computation and inference time of the model. This is because the model needs to extract features from each video frame and use all of these features as input. Furthermore, extracting 2D appearance features, 3D action features, and region features for each video frame requires additional time. To address the speed issue and improve inference speed, some recent works [18, 29] have adopted an end-to-end approach that avoids extracting multiple visual features offline. As shown in Fig. 1 (The middle branch), the flow of their method is as follows: (1) Decoding the video and densely sample frames. (2) Take video frames directly as input and then end-to-end training model. These approaches involve a trainable visual feature extractor, rather than relying on multiple offline 2D/3D feature extractors. For example, SwinBERT [18] uses VidSwin [19] as the trainable feature extractor, while MV-GPT [29] uses ViViT [1]. While these two-steps methods address the time consumption associated with offline feature extraction, they do not alleviate the computational burden and time required to handle the redundancy of information. To address the above problems, we propose an end-toend video captioning method based on compressed video. Our work significantly simplifies the video caption pipeline by eliminating time-consuming video decoding and feature extraction steps. As in Fig. 1 (the lower branch), unlike previous methods, we take compressed video information as input and directly output a natural language description of the video. Compressed video is mainly composed of I-frame, motion vector and residual, and there is no redundant information between them, and they are all refined information. Therefore, the model needs less computation to process compressed domain information, and model inference is faster. At the same time, the end-to-end network structure in our proposed method can also avoid the time consumption caused by extracting multiple features. Besides, Our model is better at understanding the content of videos by utilizing the refined information in compressed domain, including the 2D feature from I-frame and the 3D action feature extracted from motion vector and residual.  As shown in Fig. 2, compared with other two-steps and three-steps methods, such as SwinBERT [18], HMN [36] and SGN [27], our method is not only faster, but also has competitive performance. Our model comprises two parts, as depicted in Fig. 4. One part consists of three encoders that extract features and an action encoder that fuses them, while the other part comprises a multimodal decoder that generates video captions. Specifically, we first extract the context feature, motion vector feature and residual feature of the compressed video through I-frame Encoder, Motion Encoder, and Residual Encoder, respectively. The context feature contains information about objects in the video, but action information is missing. In order to extract the action feature of the video, we fuse the motion vector feature, residual feature, and context feature through the action encoder. Then use the context feature and action feature as visual input of the multimodal decoder to generate video captions. The contributions of this paper are summarized below:  
1. We propose a simple and effective transformer that can take compressed video as input and directly generate a video description.  
2. Our experimental results demonstrate that our method is nearly 2× further than the fastest existing state-of-the-art method in inference time, while maintaining competitive results on three challenging video captioning datasets, e.g., MSVD, MSRVTT and VATEX.
```

- **Related work:**

```
Compressed vision task. The main idea of introducing compressed video into current computer vision tasks is to utilizing the motion vector and residual on the compressed domain to avoid fully decode all frames from the video and save the storage space at the same time. Early work mainly base on MPEG-4 video codec [33, 16, 12, 4]. CoViAR [33] proposed a back-tracking technique to trace motion vectors back to I-frame, which works on MPEG-4. MM-ViT [4] proposed a multi-modal transformer to process the I-frame, motion vector, residual and audio in the compressed video. Since the MPEG-4 codec is outdated, other works, e.g., MVCGC [13] and ATTP [14] , is designed to work on other coedcs like H.264 and H.265 to ensure generalizability. Comparing with MPEG-4, H.264 and H.265 allow a more flexible yet complicated compression, which makes it more challenging to learn from compressed domain. MVCGC [13] proposed a self-supervised method to learn video representations by utilizing the mutual information between RGB video frames and motion vectors. ATTP [14] designed a lightweight deep neural network to process the compressed video and achieve real time action recognition on embedded AI devices. Similarly, our work is conducted on H.264 video codec, which is currently one of the most popular video codecs. Video captioning. Video captioning aims to convert the content of videos into natural language descriptions, which requires the model to understand the objects in the video and the behavior of the objects. Some works focus on the design of the model structure. These methods usually extract features offline, and then models use these features to generate captions by designing different network architectures. HMN [36] proposed a hierarchical modular network that serves as a strong video encoder, which bridges videos and languages. ORG-TRL [38] proposes an object relational graph based encoder, which captures more detailed interaction features to enrich visual representation. SGN [27] designed a semantic grouping network to group video frames with discriminating word phrases of partially decoded caption. Some works explore additional information to help the model generate more accurate video captions. TextKG [9] propose a two-stream network capable of knowledge-assisted video description using knowledge graphs. Univl [20] learns powerful visionand-language representations by pre-training the models on large-scale datasets, e.g., HowTo100M [21] and WebVid2M [2]. Some other works focus more on end-to-end video captioning generation. SwinBERT [18] proposed an endto-end transformer-based model, which takes video frame patches directly as inputs and then uses VidSwin to extract visual features. MV-GPT [29] designed an encoder-decoder model end-to-end to generate the video caption from video frames and transcribed speech directly. We propose an endto-end video captioning model based on the compressed domain without decoding video frames and extracting features offline, which not only accelerates the generation of captions, but also performs favorably against the state-of-the-art methods.
```

# [22-22][v] SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning

- **Link:** https://ieeexplore.ieee.org/document/9879057

- **Published in:** 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 58.2 | 41.3 | 77.5 | 120.6 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 41.9 | 29.9 | 62.1 | 53.8 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 38.7 | 26.2 | 53.2 | 73.0 |

- **Abstract:**

```
The canonical approach to video captioning dictates a caption generation model to learn from offline-extracted dense video features. These feature extractors usually operate on video frames sampled at a fixed frame rate and are often trained on image/video understanding tasks, without adaption to video captioning data. In this work, we present SWINBERT, an end-to-end transformer-based model for video captioning, which takes video frame patches directly as inputs, and outputs a natural language description. Instead of leveraging multiple 2D/3D feature extractors, our method adopts a video transformer to encode spatialtemporal representations that can adapt to variable lengths of video input without dedicated design for different frame rates. Based on this model architecture, we show that video captioning can benefit significantly from more densely sampled video frames as opposed to previous successes with sparsely sampled video frames for video-and-language understanding tasks (e.g., video question answering). Moreover, to avoid the inherent redundancy in consecutive video frames, we propose adaptively learning a sparse attention mask and optimizing it for task-specific performance improvement through better long-range video sequence modeling. Through extensive experiments on 5 video captioning datasets, we show that SWINBERT achieves acrossthe-board performance improvements over previous methods, often by a large margin. The learned sparse attention masks in addition push the limit to new state of the arts, and can be transferred between different video lengths and between different datasets.
```

- **Introduction:**

```
Video captioning is the task of describing the visual content of a given video in natural language. As such, it requires an algorithm to understand and model the spatial-temporal dynamics in video, as well as the relationships between visual and textual elements, and to generate a sequence of output words. This has usually been tackled with transformer-based models that learn from offline extracted video representations [26, 30, 36, 51] (Figure 1 (a)). Specifically, multiple feature extractors, usually trained on image/video understanding tasks (e.g., image classification or action recognition), are employed to extract 2D appearance features and 3D motion features from densely sampled video frames. Although achieving promising results, there exists a discrepancy in both data domain and task formulation between these off-the-shelf feature extractors and downstream video captioning. However, end-to-end training with multiple feature extractors on such dense video frames is computationally intensive, or even infeasible.
More recently, CLIPBERT [26] points out the repetitive information presented in consecutive video frames is not necessary for downstream video-and-language tasks, and proposes a sparse sampling strategy that enables affordable end-to-end training to the raw pixel inputs. Although it has shown great success in video-and-language understanding tasks, such as video question answering [27] and text-tovideo retrieval [28, 60], it remains unclear whether these sparsely sampled video frames are sufficient to generate rich and descriptive captions. Moreover, CLIPBERT leverages a 2D Convolutional Neural Network together with mean pooling that operates directly on the raw video frames to learn video representations, which may lose temporal information that is essential to describe visual events in chronological order. In this work, we aim to find an end-to-end solution to the video captioning task. Inspired by the recent successes of Transformer-based models in computer vision [5, 8, 18, 34], especially for video understanding tasks [10], we propose SWINBERT (Figure 1 (b)), a pure Transformer-based model that directly takes raw video frames as inputs for endto-end caption generation. Unlike previous methods leveraging off-the-shelf 2D/3D feature extractors at a fixed frame rate, we employ a video Transformer capable of learning from variable lengths of video frame sequence without dedicated design for different frame rates. Based on this specific model design, we investigate how many video frames are sufficient for the video captioning task?. Our experiments show that the captioning performance (i.e., CIDEr score) can be greatly lifted by more densely sampled frames (e.g., Ours: 64 frames, vs. CLIPBERT: 16 frames), in contrast to previous success with sparsely sampled frames for video-and-language understanding tasks. Lastly, to avoid the redundancy that comes naturally in consecutive video frames, we further introduce a learnable Sparse Attention Mask as a regularizer that allows the model to focus more on video frame patches that contain more spatial-temporal movements (e.g., the main moving objects) than those staying unchanged for the entire video duration (e.g., the background). In contrast to prior models [26, 30, 36] with predefined attention structures, our model can learn adaptive attention maps to optimize for task-specific performance improvements through better video sequence modeling. Our extensive experimental results on 5 video captioning datasets demonstrate that our proposed model is effective in learning sparse attention patterns to improve long-range video sequence modeling, and consequently outperforms  previous state-of-the-art approaches by a large margin. To the best of our knowledge, SWINBERT is the first end-toend pure Transformer-based architecture for video captioning. Additionally, the proposed Sparse Attention Mask effectively regularizes model training and brings further performance improvements across all 5 datasets, which opens a new direction in removing redundancy in video inputs for video-and-language modeling. In summary, our contributions are three-fold.  
• We present SWINBERT, the first end-to-end fully Transformer-based model for video captioning.  
• We introduce the Sparse Attention Mask as a regularizer for improving long-range video sequence modeling, and quantitatively validate the effectiveness of the learnable sparse attention mask in caption generation.  
• Our method outperforms previous state-of-the-art methods by a large margin on 5 popular video captioning benchmarks. As shown in Table 1, SWINBERT achieves an absolute CIDEr improvement of +25.4 on MSVD, +55.4 on YouCook2, +0.9 on MSRVTT, +5.9 on TVC and +14.9 on VATEX.
```

- **Related work:**

```
Video Captioning. Recent researches [4, 36, 41, 45, 50] mainly focus on modeling the relationship between fixed video representations and the output textual descriptions via an encoder-decoder framework for video captioning. Specifically, these methods [14, 30, 33, 36, 64] employ an encoder to refine video representations from a set of fixed video frame features, and a language decoder operates on top of these refined video representations to learn visual-textual alignment for caption generation. Researchers [4, 30, 41] have focused on exploring different 2D/3D video representations, including IncepResNetV2 [52], ResNet [21], CLIP-ViT [18,46], SlowFast [19], C3D [20] and S3D [38, 59], for improving video captioning. In addition, object-level representations [24, 63, 65] have been explored to enrich captions with fine-grained objects and actions. Prior works [15] also studied frame selection schemes to capture informative visual inputs. Unlike previous studies that learn from multiple offline-extracted 2D/3D features with a fixed sampling rate, we introduce Video Swin Transformer [34] as the video encoder in our framework to encode spatial-temporal representations from raw video frames. Benefiting from the flexibility of the transformer architecture, our model can learn with variable number of video tokens and can be trained end-to-end.  
Video transformers. Dosovitskiy et al. [18] demonstrate that a pure-transformer based architecture can outperform its convolutional counterparts in ImageNet classification task [48]. Since then, there has been a growing interest in applying vision transformer (ViT) to the video domain. For example, ViViT [5] and TimeSformer [8] propose a new transformer architecture that can leverage spatial-temporal attention for improving representation learning. Video Swin Transformer (VidSwin) [34] further introduces locality inductive bias into the transformer self-attention, and achieves state-of-the-art performance on action recognition benchmark [10]. While recent studies [5, 8, 34] mainly focus on developing video transformer architecture for action recognition, video captioning has not been explored along this research direction, which is the focus of this work.  
Video and language. Recent studies [26, 30, 37–39, 62] have shown great success on multimodal representation learning for video-and-language understanding. Popular downstream tasks include video question answering [27], text-video retrieval [28, 60] and video captioning [57]. Among the literature, Frozen-in-time [6] is a relevant study that explores pure transformer-based model design, but they focus on text-video retrieval. Specifically, they employ two independent transformer encoders for visual and textual inputs, respectively. Retrieval is conducted by estimating the similarity between the outputs of their visual and textual encoders. With a similar spirit, CLIP4Clip [37] studied using the pre-trained CLIP [46] as a feature extractor for video retrieval. While existing architectures [6, 37] are effective for video retrieval, it cannot be directly applied to video captioning, which is the focus of this work.
```

# [25-25][v] KEDL: Knowledge enhancement and disentanglement learning for video captioning

- **Link:** https://www.sciencedirect.com/science/article/pii/S0950705125010482

- **Published in:** Knowledge-Based Systems, Volume 326, 27 September 2025, 114003

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| 63.7 | 41.5 | 77.4 | 111.8 |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| 46.0 | 30.3 | 63.4 | 56.3 |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| 32.7 | 23.3 | 49.6 | 53.0 |

- **Abstract:**

```
Video captioning, bridging computer vision and natural language, is crucial for various knowledge-based systems in the age of video streaming. Recent video captioning approaches have shown promise by integrating additional text-related knowledge to enhance understanding of video content and generate more informative captions. However, methods relying heavily on knowledge graphs face several limitations, including (i) a restricted capacity to reason complex relations among object words due to static logic rules, (ii) a lack of context awareness for spatio-temporal relation analysis in videos, and (iii) the complexity of manually constructing a knowledge graph. These limitations lead to insufficient semantic information and obstruct effective alignment between visual and textual modalities. To tackle these issues, we propose a novel knowledge enhancement and disentanglement learning method for video captioning. Our approach introduces a comprehensive and adaptable knowledge source to enhance text-related knowledge, thus directly improving caption generation. Specifically, we leverage a large language model to infer enriched semantic relations between object words and speech transcripts within video frames. By integrating visual, auditory, and textual information into universal tokens with task-specific prompts, our approach enhances semantic understanding and captures more diverse relations. Furthermore, we propose a novel modality-shared disentanglement learning strategy to better align modalities, enabling a more precise link of visual cues to their corresponding textual descriptions. Specifically, we disentangle two modalities into shared and specific features, leveraging shared features to ensure alignment while mitigating uncorrelated information. Extensive experiments demonstrate that our proposed method outperforms existing methods in both quantitative and qualitative results.
```

- **Introduction:**

```
Video captioning aims to generate natural language sentences that describe sophisticated video content, which is attracting significant attention due to the exponential growth of video data in current multimedia-based knowledge systems [1], [2], [3]. Traditional research for video captioning mainly followed a template-based paradigm [4], [5], [6], which relied on predefined language templates to form video captions from detected words. However, rigid grammar rules failed to capture all key video content and generate natural sentences. Recently, advances in deep neural networks [7], [8], [9] have led to more contextual and fluent video captions, bridging the gap between video understanding and language generation. They can be generalized into three categories:
(i) The first category of the methods [1], [10], [11], [12] focuses on designing the model structures. A traditional type of structure uses convolutional neural networks for extracting visual features from videos, along with recurrent neural networks to generate captions. Recent methods employ transformers and different attention mechanisms (e.g., cross-attention) to capture long-range dependencies. (ii) The second category of the methods [13], [14], [15] concentrates on end-to-end video captioning, where feature extraction and caption generation are trained jointly. These methods typically employ encoder–decoder architectures to generate coherent captions directly from video input. (iii) The third category of the methods [16], [17] focuses on incorporating additional textual information (e.g., salient objects and underlying relational reasoning) to improve video captioning accuracy since text-related information directly relates to caption generation in addition to visual information. Some methods leverage relations between objects as additional information to enrich the generated descriptions.
In this paper, we focus on the third category due to its promising performance and potential. However, existing methods within this category [16], [17] still struggle to integrate additional textual information into the video captioning process effectively. A notable challenge is their reliance on the capacity of Knowledge Graphs (KG), which structurally represent entities and their relations. Concretely, current approaches utilizing KG suffer from limited reasoning ability for complex relations among object words in videos constrained by static logic rules, lack of context awareness, and the necessity for manual construction [18]. Static logic rules are predefined and fixed, which makes KG only capture explicitly defined relations, restricting their ability to infer more complex or unseen relations, especially in multimodal contexts. Furthermore, objects and events change over time in videos, requiring the model to update them dynamically. KG rely on static entity relations, making them unable to capture spatio-temporal interactions. Finally, manually constructing a KG is complex and inflexible, as designing and selecting appropriate types (e.g., attributes and functions) directly impacts overall performance. These limitations restrict the analysis of spatio-temporal relations in videos [19], leading to a deficiency in enriched semantic information and, consequently, sub-optimal captioning performance. Furthermore, ensuring the alignment between additional textual knowledge and the visual contents of videos presents another challenge. The coordination of multimodal representations is crucial for video captioning. Existing approaches either utilize modalities independently or conduct insufficient interaction. Each modality contains individual modality-specific features and mismatched information, e.g., the spatio-temporal redundancy in video modality, and the semantically uncorrelated relations in text modality due to inherent reasoning ambiguities among visual objects, making it challenging to enhance coherence between modalities.
To tackle the aforementioned limitations, we propose a novel Knowledge Enhancement and Disentanglement Learning (KEDL) method for video captioning, which includes four core components: a video encoder, a text encoder, a modality-shared variation extractor, and a text decoder. We introduce a Large Language Model (LLM), a neural network trained on vast text data to understand and generate language, to endow additional text knowledge into our method. Additionally, we employ disentanglement learning to refine semantic alignment between visual and textual modalities in videos. Specifically, instead of relying on traditional KG [16], we propose to leverage an LLM (i.e., GPT-3.5-Turbo) to enhance relational reasoning and obtain enriched semantic information and diverse relations since LLMs have showcased superior reasoning capabilities due to their learned common knowledge from large-scale corpora [20]. Compared to KG, LLMs are off-the-shelf and powerful tools without fine-tuning and handcrafted construction. Further, we utilize the Application Programming Interface (API) of an LLM, a software interface that enables program communication, to perform relational reasoning, eliminating the need for retraining from scratch. Concretely, keywords are inputted to the LLM for reasoning relations rather than directly using it to generate captions, which minimizes the risks of information loss or misunderstandings. Additionally, as shown in Fig. 1, instead of treating each type of text-related information (e.g., the relation-completion visual object words, speech transcripts) as an independent knowledge source, KEDL integrates them into universal tokens accompanied by task-specific prompts to fully utilize the reasoning and context awareness capability of the LLM for generating video captions.
Furthermore, we propose a modality-shared disentanglement learning strategy to enhance the interaction and alignment between video and text modalities. Disentanglement learning aims to separate representations into distinct features through architectural designs, objective functions, and learning constraints. To enhance cross-modal alignment and suppress mismatched information, we explore modality-shared and modality-specific features achieved by dedicated extractors. This strategy improves the semantic interactions between modalities and improves the quality and relevance of incorporated text data by filtering out irrelevant content. This strategy also addresses the potential limitations of semantically uncorrelated outputs caused by LLMs for video captioning. Finally, our framework is mainly constructed by employing both video and text encoders pre-trained on video-text matching data, which can effectively combine the feature representations of additional textual knowledge with visual contents, facilitating more contextually aware caption generation. The contributions of this paper can be summarized as follows:
• We propose to leverage an LLM to reason relations, enabling the model to generate context-rich and diverse descriptions by seamlessly integrating object words and speech transcripts into universal tokens, which are guided by task-specific prompts. Consequently, our model is capable of considering visual, textual, and auditory information simultaneously in videos, providing comprehensive additional text-related knowledge for video captioning.
• We propose a disentanglement learning method that extracts both modality-shared and modality-specific information to ensure well-aligned semantics within the video-text embedding space. This method improves the quality and relevance of incorporated additional text data by filtering out irrelevant information and enabling a precise link between visual contents and their corresponding textual descriptions.
• Different from conventional encoder–decoder methods that rely solely on a pre-trained video encoder to acquire visual information for video captioning, our method employs both the pre-trained video and text encoders. This configuration facilitates the learning of a shared embedding space, effectively leveraging both modalities to enhance video captioning.
```

- **Related work:**

```
2.1. Video captioning
Early video captioning methods relied on linguistic templates [4], [5], [6], which utilized predefined structures to generate captions by dividing sentences into several phases, e.g., subject, verb, and object. Specifically, individual words, e.g., object-detected words, were inserted into templates to form captions. For example, Kojima et al. [4] proposed describing human activities in videos using concept hierarchies of actions. However, these rule-based approaches were inflexible and struggled to generalize across diverse video content [21]. The development of deep learning techniques has significantly enhanced video captioning by learning comprehensive semantic representations. Recurrent neural networks and long short-term memory networks [9] enabled sequential modeling of visual features for captioning. Further, attention mechanisms [22] allowed models to focus on relevant video frames dynamically. More recently, transformer-based architectures [23] and large-scale pre-training techniques [24] further enhanced the ability to generate coherent and contextual captions, bridging the gap between visual understanding and natural language description.
To improve the overall effectiveness and accuracy of video captioning, some research efforts have concentrated on designing powerful model structures that enhance both feature extraction and caption generation. For example, Ryu et al. [1] introduced a semantic grouping network for clustering video frames based on distinct word phrases from decoded captions to reduce redundancy. Ye et al. [10] proposed a novel hierarchical modular network that served as a video encoder connecting videos with natural language. Later, diverse sequence learning methods that leverage encoder–decoder frameworks are leveraged. Lei et al. [25] designed a shared encoder–decoder network for coherent video paragraph captioning, which was augmented with a memory module. Zhu et al. [26] leveraged local regional features to enhance the alignment between visual and language modalities. Additionally, Shen et al. [27] proposed an end-to-end compressed model to eliminate the need for decoding video frames and offline feature extraction. Meanwhile, various methods are dedicated to integrating additional text information (e.g., salient object words, and their related relational reasoning) to enhance the precision of captions. As such text-related information is directly relevant to caption generation, in addition to merely analyzing the visual contents from videos. For instance, Zhang et al. [17] proposed an object-relational graph with teacher-recommended learning for video captioning, which utilized knowledge from Bidirectional Encoder Representations from Transformers [28] (BERT), a pre-training language model that captures bidirectional context, to suggest candidate words. Similarly, Gu et al. [16] introduced a two-stream network that incorporated additional textual data from KG to enrich video descriptions.
While existing approaches have shown promise, they often suffer from limited reasoning capacities, static logic rules, and a lack of context awareness. Motivated by these limitations, we propose a more robust approach that incorporates additional text-related knowledge to enhance video captioning.
2.2. Multimodal pre-trained models and large language models in video captioning
The complexity of video content requires models that can effectively integrate information from diverse modalities, such as visual, textual, and auditory cues. The rise of multimodal pre-training, particularly visual-language pre-trained models, e.g., Unified Video and Language pre-training [29] (UniVL), which integrates video and language learning, and Contrastive Language-Image Pre-training [24] (CLIP), which aligns image and text using contrastive learning , has driven significant advancements by exploring effective visual representations through supervised or self-supervised methods [30]. CLIP [24] learned visual representations during a pre-trained stage, enabling the zero-shot transfer of the learned representations to downstream tasks. In video captioning, CLIP4Caption [31] framework enhanced CLIP with a video-text matching network, where the pre-trained video encoder was employed for generating captions during the fine-tuning phase. Moreover, CLIP4VLA model [32] enriched multimodal representations by incorporating audio data with textual and visual inputs during the CLIP pre-training phase. Furthermore, LLMs have proven to be highly effective across a wide range of tasks, including natural language inference and question answering [33], [34]. LLMs (e.g., GPT-3 [35]) exhibited impressive few-shot performance on various tasks without further fine-tuning, enabling them to serve as powerful tools for data generation. For instance, Brooks et al. [36] used GPT-3 to generate instructions and edit captions in a dataset, which was subsequently employed to train a model for image editing.
Meanwhile, some recent approaches conduct a straightforward usage of LLMs for video-language tasks, such as the video-language pre-training [37] and the fine-tuning strategies for downstream tasks [38]. However, these methods do not leverage an LLM as a plug-and-play knowledge source for relational reasoning, and the integration of multiple modalities into universal tokens to fully utilize the reasoning capabilities of an LLM remains underexplored.
2.3. Disentanglement learning
Our research is intricately connected to the field of disentanglement learning, which revolves around separating different aspects of data into distinct and independent low-dimensional latent vector spaces. This area has garnered significant attention for its pivotal role in enhancing the interpretability of deep learning models, allowing for better understanding and manipulation of complex data representations. In recent years, the versatility of disentanglement has been evident across diverse domains, including images, videos, and speech [39]. For example, Zheng et al. [40] implemented separate encoders to isolate the appearance and structure features, enabling the network to generate high-quality composite images across different identities, thereby facilitating online learning. Recently, Sun et al. [41] leveraged shared and private encoders to conduct fine-grained disentanglement learning for text and audio modalities.
Different from these approaches, our method utilizes modality-shared information rather than all disentangled components, since the modality-specific information and inherent semantically uncorrelated information of modalities can be disentangled into modality-specific variations, leading to negative effects and hindering effective alignment. Meanwhile, we leverage a modality reconstruction strategy to encourage the learning of modality-specific variations, which is different from existing methods based on complex objective-guided manner [41], [42].
```

# [][] Paper-name

- **Link:** 

- **Published in:** 

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
```

- **Introduction:**

```
```

- **Related work:**

```
```


---

# [][] 

- **Link:** 

- **Published in:** 

- **Main results:**

| MSVD-B@4 | MSVD-M | MSVD-R | MSVD-C | 
| --- | --- | --- | --- |
| |

| MSRVTT-B@4 | MSRVTT-M | MSRVTT-R | MSRVTT-C | 
| --- | --- | --- | --- |
| |

| VATEX-B@4 | VATEX-M | VATEX-R | VATEX-C |
| --- | --- | --- | --- |
| |

- **Abstract:**

```
```

- **Introduction:**

```
```

- **Related work:**

```
```
