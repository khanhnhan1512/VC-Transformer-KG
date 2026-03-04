# Section: 5. Experiments

## 5.1. Experimental settings

### 5.1.1. Datasets

To evaluate the effectiveness and generalization capability of our proposed BiDecT model, we conduct experiments on three widely used video captioning benchmarks: MSVD, MSR-VTT, and VATEX.

**MSVD** <cite> comprises 1970 video clips, each paired with approximately 40 candidate captions. On average, these clips have a duration of about 10 seconds. Following the standard split adopted by prior work, we allocate 1200 samples for training, 100 for validation, and the remaining 670 for testing.

**MSR-VTT** <cite> is a generic, large-scale video captioning dataset containing 10000 videos from 20 diverse categories. Each clip is described by 20 human-annotated captions. Consistent with the conventional split, we divide the data into 6513 samples for training, 497 for validation, and 2990 for testing.

**VATEX** <cite> is a massive dataset comprising approximately 41269 video clips, where each clip is manually annotated with 10 English and 10 Chinese captions. In this work, we solely utilize English captions. The official split allocates 25991 videos to the training set, 3000 to the validation set, and 6000 to the public test set. We exclude the private test set (containing about 6278 videos) due to the lack of accessible ground-truth captions. Since a portion of the original YouTube videos is no longer available, we utilize the hosted data provided by the Common Visual Data Foundation and Yang et al. <cite>. Consequently, our final split consists of 25985 training videos, 3000 validation videos, and 5808 testing videos.

### 5.1.2. Evaluation Metrics

To comprehensively and objectively evaluate the quality of the generated captions, we employ four standard metrics **computed via the Microsoft COCO Evaluation Server** <cite>: BLEU@4 (B4) <cite>, METEOR (M) <cite>, ROUGE-L (R) <cite>, and CIDEr (C) <cite>.

These metrics provide complementary perspectives on caption quality. Specifically, **BLEU@4** measures n-gram precision to assess sentence fluency. **METEOR** evaluates semantic alignment by considering exact matches, stems, and synonyms. **ROUGE-L** focuses on structural similarity based on the longest common subsequences.

Most importantly, we prioritize CIDEr, a metric specifically designed for captioning tasks, as it captures the consensus between generated captions and human references to reflect the accuracy of key information.

### 5.1.3. Implementation Details

- Preprocessing raw video + feature extraction on GOP
- Transformer model configuration
