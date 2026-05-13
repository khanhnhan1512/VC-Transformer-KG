# 5. Experiments

## 5.1. Datasets

We evaluate BiDecT on three widely used video captioning benchmarks. Key statistics are summarized in Table [$\sout{???}$]().

| Dataset | #Captions/Video | Train | Validation | Test |
|---|---|---|---|---|
| MSVD    | ~40          | 1,200  | 100   | 670   |
| MSR-VTT | 20           | 6,513  | 497   | 2,990 |
| VATEX   | 10 (English) | 25,985 | 3,000 | 5,808 |

Table [$\sout{???}$](). Statistics of the three benchmark datasets used in our experiments. For VATEX, we use only English captions and report results on the public test set. The VATEX split reflects the videos that are accessible in our experimental setup.
<br><br>

**MSVD** [$\sout{CITE}$ Collecting Highly Parallel Data for Paraphrase Evaluation]() consists of 1,970 short video clips averaging 10 seconds each, with approximately 40 human-annotated captions per clip. We follow the standard split used in prior work.

**MSR-VTT** [$\sout{CITE}$ MSR-VTT: A Large Video Description Dataset for Bridging Video and Language]() is a generic benchmark of 10,000 clips spanning 20 diverse categories, with 20 captions per clip and an average duration of approximately 15 seconds. We adopt the conventional split used in prior work.

**VATEX** [$\sout{CITE}$ VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research]() is a large multilingual dataset of 41,269 video clips, each approximately 10 seconds long and annotated with 10 English and 10 Chinese captions. We use only the English captions and evaluate on the public test set. Because some of the original videos are no longer accessible, our actual split comprises 25,985 training, 3,000 validation, and 5,808 test videos, compared to the official splits of 25,991 / 3,000 / 6,000.

## 5.2. Evaluation Metrics

We report results using four standard metrics for video captioning: BLEU@4 (B4) [$\sout{CITE}$ Bleu: a method for automatic evaluation of machine translation](), METEOR (M) [$\sout{CITE}$ Meteor: An automatic metric for mt evaluation with improved correlation with human judgments](), ROUGE-L (R) [$\sout{CITE}$ Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics](), and CIDEr (C) [$\sout{CITE}$ Cider: Consensus-based image description evaluation](). BLEU@4 measures modified n-gram precision up to 4-grams with a brevity penalty. METEOR evaluates word-level alignment using exact, stem, and synonym matches. ROUGE-L measures sentence-level overlap based on the longest common subsequence. CIDEr uses TF-IDF-weighted n-gram similarity to measure agreement with human reference captions. Following common practice in video captioning, we use CIDEr as our primary evaluation metric.

## 5.3. Implementation Details

**Video Preprocessing.** All raw videos are resized to 240 pixels on their shortest edge and re-encoded using the H.264 codec via FFmpeg, with $\text{KeyInt}$ set to 45. For each video, we keep at most $G$ GOPs, uniformly sampled over time if the video contains more than $G$ GOPs. Based on the 75th percentile of the GOP count distribution in each training set, we set $G$ to 8, 10, and 8 for MSVD, MSR-VTT, and VATEX, respectively.

**Feature Extraction.** As described in Section [$\sout{???}$](), all features are extracted offline using pre-trained models. Table [$\sout{???}$]() summarizes the specific model and feature dimension for each modality.

| Feature | Model | Input Source | Dimension |
|---|---|---|---|
| Appearance | BLIP-2 OPT-2.7B [$\sout{CITE}$ BLIP-2]() | I-frame | $d_A = 1{,}408$ |
| Semantic | all-RoBERTa-large-v1 (SRoBERTa) [$\sout{CITE}$ Sentence-BERT]() | BLIP-2 caption | $d_S = 1{,}024$ |
| Motion | MViTv2-small [$\sout{CITE}$ MViTv2]() | 16 sampled frames | $d_M = 768$ |

Table [$\sout{???}$](). Feature extractors used in BiDecT.
<br><br>

**Model Configuration.** We set the model dimension $d_{model} = 512$, the feed-forward dimension $d_{ff} = 2048$, and the number of attention heads to 4. Both the backward and forward decoders consist of 3 stacked Transformer layers ($N_{BD} = N_{FD} = 3$). The maximum caption length is set to 20 for all datasets.

**Training and Inference.** The model is trained for 16 epochs with a batch size of 64. We use the Adam optimizer (AMSGrad variant) with an initial learning rate of $1 \times 10^{-4}$, weight decay of $0.5 \times 10^{-5}$, and gradient clipping at 5.0. The learning rate is linearly warmed up during the first 3 epochs and then reduced by a factor of 0.5 when validation loss does not improve for 3 consecutive epochs. We set the loss weighting hyperparameter $\lambda$ to 0.6, with dropout of 0.1 and label smoothing of 0.15. Caption generation uses beam search with a beam size of 4 for both decoders. The checkpoint with the best validation CIDEr score is selected for evaluation.

**Hardware.** All experiments are conducted on NVIDIA A100 GPUs.

## 5.4. Comparison with State-of-the-Art Methods

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Year</th>
      <th colspan="4">MSVD</th>
      <th colspan="4">MSR-VTT</th>
    </tr>
    <tr>
      <th>B4</th><th>M</th><th>R</th><th>C</th>
      <th>B4</th><th>M</th><th>R</th><th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SwinBERT <u>SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning</u></td>
      <td>2022</td>
      <td>58.2</td><td>41.3</td><td>77.5</td><td>120.6</td>
      <td>41.9</td><td>29.9</td><td>62.1</td><td>53.8</td>
    </tr>
    <tr>
      <td>TextKG <u>Text with Knowledge Graph Augmented Transformer for Video Captioning</u></td>
      <td>2023</td>
      <td>60.8</td><td>38.5</td><td>75.1</td><td>105.2</td>
      <td>43.7</td><td>29.6</td><td>62.4</td><td>52.4</td>
    </tr>
    <tr>
      <td>CoCap <u>Accurate and Fast Compressed Video Captioning</u></td>
      <td>2023</td>
      <td>60.1</td><td>41.4</td><td>78.2</td><td>121.5</td>
      <td>44.4</td><td>30.3</td><td>63.4</td><td>57.2</td>
    </tr>
    <tr>
      <td>CARE <u>Concept-Aware Video Captioning: Describing Videos With Effective Prior Information</u></td>
      <td>2023</td>
      <td>56.3</td><td>39.1</td><td>75.6</td><td>106.9</td>
      <td>48.7</td><td><u>31.5</u></td><td>65.2</td><td>59.7</td>
    </tr>
    <tr>
      <td>BTKG <u>Bidirectional transformer with knowledge graph for video captioning</u></td>
      <td>2024</td>
      <td>55.7</td><td>38.3</td><td>74.7</td><td>104.5</td>
      <td>42.8</td><td>30.0</td><td>62.4</td><td>55.4</td>
    </tr>
    <tr>
      <td>IVRC <u>Rethink video retrieval representation for video captioning</u></td>
      <td>2024</td>
      <td>58.8</td><td>40.3</td><td>77.4</td><td>116.0</td>
      <td>43.7</td><td>30.2</td><td>63.0</td><td>57.1</td>
    </tr>
    <tr>
      <td>OmniViD <u>OmniViD: A Generative Framework for Universal Video Understanding</u></td>
      <td>2024</td>
      <td>59.7</td><td>42.2</td><td>78.1</td><td>122.5</td>
      <td>44.3</td><td>29.9</td><td>62.7</td><td>56.6</td>
    </tr>
    <tr>
      <td>IcoCap <u>IcoCap: Improving Video Captioning by Compounding Images</u></td>
      <td>2024</td>
      <td>59.1</td><td>39.5</td><td>76.5</td><td>110.3</td>
      <td>47.0</td><td>31.1</td><td>64.9</td><td>60.2</td>
    </tr>
    <tr>
      <td>KG-VCN <u>Fully exploring object relation interaction and hidden state attention for video captioning</u></td>
      <td>2025</td>
      <td><u>64.9</u></td><td>39.7</td><td>77.2</td><td>107.1</td>
      <td>45.0</td><td>28.7</td><td>62.5</td><td>51.9</td>
    </tr>
    <tr>
      <td>Track4Cap <u>Frame-by-Frame Multi-Object Tracking-Guided Video Captioning</u></td>
      <td>2025</td>
      <td>62.1</td><td><u>42.5</u></td><td><u>79.8</u></td><td><u>127.2</u></td>
      <td>44.6</td><td>30.5</td><td>63.6</td><td>57.7</td>
    </tr>
    <tr>
      <td>DSSM-KG <u>DSSM-KG: Dual-Stream State-Space Modeling with Adaptive Knowledge Injection for Video Captioning</u></td>
      <td>2025</td>
      <td>57.9</td><td>40.0</td><td>77.0</td><td>111.7</td>
      <td>47.6</td><td>31.2</td><td>65.1</td><td>59.3</td>
    </tr>
    <tr>
      <td>UHCL <u>Unified hierarchical contrastive learning for video captioning</u></td>
      <td>2026</td>
      <td>59.3</td><td>40.5</td><td>77.5</td><td>114.6</td>
      <td><strong>49.4</strong></td><td><strong>31.7</strong></td><td><u>65.6</u></td><td>61.7</td>
    </tr>
    <tr>
      <td>CapDistill <u>Dual-hierarchical knowledge distillation for video captioning</u></td>
      <td>2026</td>
      <td>61.6</td><td>42.4</td><td>79.1</td><td>127.0</td>
      <td>45.3</td><td>30.2</td><td>63.8</td><td>57.9</td>
    </tr>
    <tr>
      <td>MK-VC <u>Scene adaptive dynamic multi-modal knowledge for video captioning</u></td>
      <td>2026</td>
      <td>59.3</td><td>40.6</td><td>77.8</td><td>115.1</td>
      <td>48.1</td><td><strong>31.7</strong></td><td>65.5</td><td><u>62.0</u></td>
    </tr>
    <tr>
      <td>QPDC <u>Ask and focus more: Question-prompt uncertainty allocation for dual-controllable video captioning</u></td>
      <td>2026</td>
      <td>59.4</td><td>38.4</td><td>74.8</td><td>105.2</td>
      <td>42.4</td><td>28.9</td><td>61.9</td><td>51.8</td>
    </tr>
    <tr>
      <td>EMKG <u>Towards generalized video captioning: An effective multi-modal knowledge graph perspective</u></td>
      <td>2026</td>
      <td>59.5</td><td>40.4</td><td>78.0</td><td>116.4</td>
      <td>48.5</td><td><u>31.5</u></td><td>65.1</td><td>61.3</td>
    </tr>
    <tr>
      <td><strong>BiDecT (Ours)</strong> <u>https://www.kaggle.com/code/vmphat/bidect-exp?scriptVersionId=301167326</u></td>
      <td></td>
      <td><strong>67.7</strong></td><td><strong>45.8</strong></td><td><strong>82.9</strong></td><td><strong>138.0</strong></td>
      <td><u>49.2</u></td><td><strong>31.7</strong></td><td><strong>65.9</strong></td><td><strong>64.8</strong></td>
    </tr>
  </tbody>
</table>

Table [$\sout{???}$](). Comparison with state-of-the-art video captioning methods on the test split of MSVD and MSR-VTT. The best and second-best results in each column are denoted by **bold** and <u>underline</u>, respectively.
<br><br>

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Year</th>
      <th colspan="4">VATEX</th>
    </tr>
    <tr>
      <th>B4</th><th>M</th><th>R</th><th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SwinBERT <u>SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning</u></td>
      <td>2022</td>
      <td><u>38.7</u></td><td><u>26.2</u></td><td><u>53.2</u></td><td><u>73.0</u></td>
    </tr>
    <tr>
      <td>CoCap <u>Accurate and Fast Compressed Video Captioning</u></td>
      <td>2023</td>
      <td>35.8</td><td>25.3</td><td>52.0</td><td>64.8</td>
    </tr>
    <tr>
      <td>CARE <u>Concept-Aware Video Captioning: Describing Videos With Effective Prior Information</u></td>
      <td>2023</td>
      <td>37.5</td><td>25.1</td><td>52.4</td><td>63.1</td>
    </tr>
    <tr>
      <td>IVRC <u>Rethink video retrieval representation for video captioning</u></td>
      <td>2024</td>
      <td>32.8</td><td>24.0</td><td>50.3</td><td>57.4</td>
    </tr>
    <tr>
      <td>IcoCap <u>IcoCap: Improving Video Captioning by Compounding Images</u></td>
      <td>2024</td>
      <td>37.4</td><td>25.7</td><td>53.1</td><td>67.8</td>
    </tr>
    <tr>
      <td>RLHMN <u>Learning Hierarchical Modular Networks for Video Captioning</u></td>
      <td>2024</td>
      <td>35.3</td><td>23.1</td><td>50.9</td><td>58.8</td>
    </tr>
    <tr>
      <td>KG-VCN <u>Fully exploring object relation interaction and hidden state attention for video captioning</u></td>
      <td>2025</td>
      <td>33.3</td><td>22.9</td><td>49.5</td><td>53.3</td>
    </tr>
    <tr>
      <td>KEDL <u>KEDL: Knowledge enhancement and disentanglement learning for video captioning</u></td>
      <td>2025</td>
      <td>32.7</td><td>23.3</td><td>49.6</td><td>53.0</td>
    </tr>
    <tr>
      <td>CapDistill <u>Dual-hierarchical knowledge distillation for video captioning</u></td>
      <td>2026</td>
      <td>31.6</td><td>23.9</td><td>49.6</td><td>55.3</td>
    </tr>
    <tr>
      <td>QPDC <u>Ask and focus more: Question-prompt uncertainty allocation for dual-controllable video captioning</u></td>
      <td>2026</td>
      <td>36.8</td><td>24.9</td><td>52.0</td><td>61.3</td>
    </tr>
    <tr>
      <td><strong>BiDecT (Ours)</strong> <u>https://www.kaggle.com/code/vmphat/bidect-exp?scriptVersionId=301599735</u></td>
      <td></td>
      <td><strong>41.4</strong></td><td><strong>27.0</strong></td><td><strong>54.9</strong></td><td><strong>78.2</strong></td>
    </tr>
  </tbody>
</table>

Table [$\sout{???}$](). Comparison with state-of-the-art video captioning methods on the public test set of VATEX. The best and second-best results in each column are denoted by **bold** and <u>underline</u>, respectively.
<br><br>

Tables [$\sout{???}$]() and [$\sout{???}$]() compare BiDecT with recent state-of-the-art methods on MSVD, MSR-VTT, and VATEX. All comparison results are taken directly from the respective original papers. The compared methods span a range of design strategies, including end-to-end video Transformers, knowledge graph-augmented pipelines, contrastive learning frameworks, and tracking-guided approaches. BiDecT achieves the highest CIDEr score, our primary evaluation metric, on all three benchmarks and ranks first or second on every reported metric.

On MSVD, BiDecT achieves the best score on all four metrics. The improvement is most pronounced on CIDEr, where BiDecT (138.0) surpasses the second-best method, Track4Cap [$\sout{CITE}$ Frame-by-Frame Multi-Object Tracking-Guided Video Captioning]() (127.2), by 10.8 points, with additional margins of +3.3 on METEOR and +3.1 on ROUGE-L over the same method. Track4Cap augments frame-level features with a multi-object tracking module and an object relation encoder, forming a more complex intermediate pipeline. BiDecT also outperforms KG-VCN [$\sout{CITE}$ Fully exploring object relation interaction and hidden state attention for video captioning](), which holds the second-highest BLEU@4 (64.9), by 2.8 points despite not relying on graph-based object modeling. These results suggest that rich multimodal features projected directly into the decoder can be more effective than features processed through additional intermediate stages.

On MSR-VTT, a more diverse benchmark spanning 20 video categories, performance differences among top methods are narrower. BiDecT achieves the highest CIDEr (64.8), the highest ROUGE-L (65.9), and ties for the highest METEOR (31.7) with UHCL [$\sout{CITE}$ Unified hierarchical contrastive learning for video captioning]() and MK-VC [$\sout{CITE}$ Scene adaptive dynamic multi-modal knowledge for video captioning](). On BLEU@4, BiDecT (49.2) is within 0.2 points of UHCL (49.4), which employs triamese decoders with hierarchical contrastive learning. Among knowledge graph-based approaches, MK-VC attains the next-highest CIDEr (62.0) through dynamic fusion of commonsense and video-related knowledge, yet BiDecT leads by 2.8 points without relying on external knowledge construction pipelines.

On VATEX, the largest benchmark in our evaluation, BiDecT achieves the best score on all four metrics. CIDEr reaches 78.2, surpassing the second-best method, SwinBERT [$\sout{CITE}$ SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning]() (73.0), by 5.2 points. An observation worth noting is that SwinBERT, an end-to-end model from 2022, holds the second-best position across all metrics on this dataset, outperforming several more recent methods that incorporate additional processing modules. BiDecT's consistent improvement over SwinBERT (+2.7 BLEU@4, +0.8 METEOR, +1.7 ROUGE-L, +5.2 CIDEr) suggests that the proposed encoder-free multimodal design generalizes effectively to larger-scale datasets.

Across all three benchmarks, several patterns emerge when examining the results through the lens of our design choices. First, BiDecT consistently outperforms BTKG [$\sout{CITE}$ Bidirectional transformer with knowledge graph for video captioning](), its most architecturally similar counterpart that also adopts bidirectional decoding but incorporates encoder modules and knowledge graph augmentation (e.g., by +33.5 CIDEr on MSVD and +9.4 on MSR-VTT). Second, BiDecT outperforms CoCap [$\sout{CITE}$ Accurate and Fast Compressed Video Captioning](), which also exploits compressed video structure, on all three datasets, indicating that GOP-based extraction combined with complementary appearance, semantic, and motion features provides a more effective video representation. Third, BiDecT achieves higher CIDEr than every knowledge graph-augmented method in the comparison, suggesting that semantic features derived from a pre-trained vision-language model can serve as an effective alternative to explicit knowledge graph construction pipelines.

## 5.5. Ablation Study

To analyze the contribution of each component and the sensitivity of key hyperparameters, we conduct all experiments in Sections 5.5 and 5.6 on MSR-VTT unless otherwise stated.

**Effect of Bidirectional Decoding.** Figure [$\sout{???}$]() compares the full BiDecT configuration (FD+BD) with a variant that removes the backward decoder entirely (FD-only), reducing the model to a standard unidirectional decoder that conditions solely on the multimodal representation $\overrightarrow{E}$. Introducing the backward decoder yields consistent improvements across all four metrics. The gain is most pronounced on CIDEr (+4.52), our primary metric, followed by ROUGE-L (+2.53), BLEU@4 (+1.75), and METEOR (+1.00). Notably, the two metrics most sensitive to overall caption quality, CIDEr and ROUGE-L, show larger gains than BLEU@4 and METEOR, suggesting that the global backward context $\overleftarrow{H}$ is particularly effective at improving caption coherence rather than isolated word choices.

![](figures/bidirectional_decoding_comparison.svg)<br>
Figure [$\sout{???}$](). Effect of bidirectional decoding on the MSR-VTT test set. FD-only denotes the model without the backward decoder, in which the forward decoder has no access to backward context $\overleftarrow{H}$. FD+BD corresponds to the full BiDecT configuration. Delta values above each bar pair indicate the absolute improvement.
<br><br>

**Effect of Pseudo Reverse Caption.**

**Effect of Input Modalities.**

<table>
  <thead>
    <tr>
      <th rowspan="2">App</th>
      <th rowspan="2">Sem</th>
      <th rowspan="2">Mot</th>
      <th rowspan="2">Type Emb</th>
      <th rowspan="2">Ind. Emb.</th>
      <th colspan="4">MSR-VTT</th>
    </tr>
    <tr>
      <th>B4</th><th>M</th><th>R</th><th>C</th>
    </tr>
  </thead>
  <tbody>
    <!-- Single modality (no type embedding needed) -->
    <tr>
      <td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>✓</td>
      <td>46.86</td><td>30.98</td><td>64.55</td><td>61.79</td>
    </tr>
    <tr>
      <td>✗</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td>
      <td>45.39</td><td>30.42</td><td>63.62</td><td>60.32</td>
    </tr>
    <tr>
      <td>✗</td><td>✗</td><td>✓</td><td>✗</td><td>✓</td>
      <td>40.10</td><td>28.24</td><td>59.60</td><td>51.02</td>
    </tr>
    <!-- Two modalities (with type embedding) -->
    <tr>
      <td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>✓</td>
      <td>48.35</td><td>31.56</td><td>65.07</td><td>63.32</td>
    </tr>
    <tr>
      <td>✓</td><td>✗</td><td>✓</td><td>✓</td><td>✓</td>
      <td>47.62</td><td>31.18</td><td>64.74</td><td>62.34</td>
    </tr>
    <tr>
      <td>✗</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td>
      <td>47.11</td><td>30.92</td><td>64.27</td><td>61.17</td>
    </tr>
    <!-- All 3 modalities, no type embedding -->
    <tr>
      <td>✓</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td>
      <td>47.27</td><td>30.82</td><td>64.50</td><td>61.18</td>
    </tr>
    <!-- Separator: embedding module design -->
    <tr style="border-top: 2px solid #000;">
      <td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td>
      <td>48.74</td><td>31.45</td><td>65.55</td><td>62.53</td>
    </tr>
    <!-- Full model (BiDecT) -->
    <tr>
      <td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td>
      <td><strong>49.23</strong></td><td><strong>31.73</strong></td>
      <td><strong>65.87</strong></td><td><strong>64.83</strong></td>
    </tr>
  </tbody>
</table>

Table [$\sout{???}$](). Ablation study on input modalities and embedding module design, evaluated on the MSR-VTT test set. App / Sem / Mot denote appearance, semantic, and motion features. Type Emb indicates whether modality-specific type embeddings are applied. Ind. Emb. indicates whether the backward and forward decoders use independent feature embedding modules; ✗ denotes a single shared module. The last row corresponds to the full BiDecT configuration.
<br><br>

**Effect of Layer Normalization Strategy.**



## 5.6. Hyperparameter Analysis

**Effect of Decoder Depth.**

**Effect of Attention Heads.**

**Effect of Loss Weight (λ).**

**Effect of GOP Count.**

<table>
  <thead>
    <tr>
      <th rowspan="2">Max GOPs</th>
      <th rowspan="2">Time (s)</th>
      <th colspan="4">MSR-VTT</th>
    </tr>
    <tr>
      <th>B4</th><th>M</th><th>R</th><th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6 (P25)</td>
      <td>405.96</td>
      <td>48.37</td><td>31.18</td><td>65.22</td><td>63.46</td>
    </tr>
    <tr>
      <td>8 (P50)</td>
      <td>407.92</td>
      <td>48.10</td><td>31.25</td><td>64.97</td><td>63.86</td>
    </tr>
    <tr>
      <td><strong>10 (P75)†</strong></td>
      <td><strong>411.46</strong></td>
      <td><strong>49.23</strong></td><td><strong>31.73</strong></td>
      <td><strong>65.87</strong></td><td><strong>64.83</strong></td>
    </tr>
    <tr>
      <td>12 (P90)</td>
      <td>414.40</td>
      <td>48.37</td><td>31.28</td><td>65.27</td><td>64.08</td>
    </tr>
  </tbody>
</table>

Table [$\sout{???}$](). Effect of the maximum number of GOPs per video on MSR-VTT, with KeyInt fixed at 45. Pxx denotes the xx-th percentile of the GOP count distribution in the MSR-VTT training set. Time (s) denotes total inference time on the test set. † indicates the configuration adopted in our experiments.
