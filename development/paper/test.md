## 4.2. Video Representation and Multimodal Feature Extraction

**Video Representation.** As mentioned in Section [$\sout{???}$](), to improve computational efficiency and reduce redundancy, we process the video based on its compression structure rather than frame-by-frame. Given an input video $X$, we represent it as a sequence of Groups of Pictures (GOPs):

$$X = \left[ \text{GOP}^{(1)}, \dots, \text{GOP}^{(G)} \right],$$

where $G$ is the maximum number of sampled GOPs per video. Each GOP strictly begins with an I-frame, followed by a series of predictive frames (P/B-frames):

$$\text{GOP}^{(g)} = \left[\text{I}^{(g)}, \text{P/B}^{(g, 1)}, \dots, \text{P/B}^{(g, \text{KeyInt}-1)}\right].$$

Here, $\text{KeyInt}$ (Keyframe Interval) is a hyperparameter defining the maximum distance between two consecutive I-frames during the video compression process. Therefore, a single GOP contains at most $\text{KeyInt}-1$ consecutive P/B-frames.

**Feature Extraction Process.** After partitioning the video into GOPs, the next critical step is extracting robust feature representations from each GOP. Historically, using 2D appearance features (spatial information) and 3D motion features (temporal information) has been the standard and widespread practice for building multimodal representations. While effective, relying only on these visual signals often leaves a "semantic gap" between low-level visual content and high-level natural language descriptions. To bridge this gap and enhance semantic understanding, we introduce a third modality: the semantic feature.

To ensure data consistency, we apply the feature extraction process uniformly across every GOP. For a given $\text{GOP}^{(g)}$, we extract three different types of information:

**Appearance and Semantic Features (from I-frame):** Because the I-frame contains the most complete spatial details of the scene, it is the ideal candidate to capture not only the physical appearance but also the high-level semantic meaning of the entire GOP. Therefore, we use the I-frame as the single source to extract both features through a collaborative pipeline using BLIP-2 [$\sout{CITE}$]() and SRoBERTa [$\sout{CITE}$](). First, we feed the I-frame into the Image Encoder of BLIP-2 and extract the $\text{[CLS]}$ token to serve as our appearance token ($a^{(g)}$). Next, all of the encoder's outputs are reused in the full BLIP-2 pipeline (passing through the Querying Transformer and the Large Language Model) to generate a descriptive text caption for that exact I-frame. Finally, this generated caption is processed by SRoBERTa to produce a dense vector, which acts as our semantic token ($s^{(g)}$).

**Motion Feature (from GOP sequence):** Although the I-frame provides strong static context, it lacks the temporal dynamics necessary to capture the local motion and action transitions within each GOP. To add continuous temporal information, we use MViTv2 [$\sout{CITE}$]() to process the sequence of frames within the $\text{GOP}^{(g)}$. This model captures the motion transitions and outputs a motion token $m^{(g)}$.

By applying this pipeline to all $G$ GOPs in the video, we successfully collect three distinct features. Mathematically, the video $X$ is now represented by:

1. Appearance feature: $F_A = [a^{(1)}, \dots, a^{(G)}]$, where $a^{(g)} \in \mathbb{R}^{d_A}$ and $F_A \in \mathbb{R}^{G \times d_A}$.
2. Semantic feature: $F_S = [s^{(1)}, \dots, s^{(G)}]$, where $s^{(g)} \in \mathbb{R}^{d_S}$ and $F_S \in \mathbb{R}^{G \times d_S}$.
3. Motion feature: $F_M = [m^{(1)}, \dots, m^{(G)}]$, where $m^{(g)} \in \mathbb{R}^{d_M}$ and $F_M \in \mathbb{R}^{G \times d_M}$.

Here, $d_A$, $d_S$, and $d_M$ are the hidden dimensions of the respective pre-trained models. Once extracted, these three distinct features serve as the direct inputs for the subsequent Multimodal Feature Embedding module.
