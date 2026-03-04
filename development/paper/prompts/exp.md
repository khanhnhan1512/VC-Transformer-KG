# Section: 5. Experiments

## 5.1. Experimental settings

### 5.1.1. Datasets

To evaluate the effectiveness and generalization capability of our proposed BiDecT model, we conduct experiments on three widely used video captioning benchmarks: MSVD, MSR-VTT, and VATEX.

**MSVD** `<cite>` comprises 1970 video clips, each paired with approximately 40 candidate captions. On average, these clips have a duration of about 10 seconds. Following the standard split adopted by prior work, we allocate 1200 samples for training, 100 for validation, and the remaining 670 for testing.

**MSR-VTT** `<cite>` is a generic, large-scale video captioning dataset containing 10000 videos from 20 diverse categories. Each clip is described by 20 human-annotated captions. Consistent with the conventional split, we divide the data into 6513 samples for training, 497 for validation, and 2990 for testing.

**VATEX** `<cite>` is a massive dataset comprising approximately 41269 video clips, where each clip is manually annotated with 10 English and 10 Chinese captions. In this work, we solely utilize English captions. The official split allocates 25991 videos to the training set, 3000 to the validation set, and 6000 to the public test set. We exclude the private test set (containing about 6278 videos) due to the lack of accessible ground-truth captions. Since a portion of the original YouTube videos is no longer available, we utilize the hosted data provided by the Common Visual Data Foundation and Yang et al. `<cite-MultiCap>`. Consequently, our final split consists of 25985 training videos, 3000 validation videos, and 5808 testing videos.

### 5.1.2. Evaluation Metrics

To comprehensively and objectively evaluate the quality of the generated captions, we employ four standard metrics **computed via the Microsoft COCO Evaluation Server** `<cite>`: BLEU@4 (B4) `<cite>`, METEOR (M) `<cite>`, ROUGE-L (R) `<cite>`, and CIDEr (C) `<cite>`.

These metrics provide complementary perspectives on caption quality. Specifically, **BLEU@4** measures n-gram precision to assess sentence fluency. **METEOR** evaluates semantic alignment by considering exact matches, stems, and synonyms. **ROUGE-L** focuses on structural similarity based on the longest common subsequences.

Most importantly, we prioritize CIDEr, a metric specifically designed for captioning tasks, as it captures the consensus between generated captions and human references to reflect the accuracy of key information.

### 5.1.3. Implementation Details

```
- Preprocessing raw video + feature extraction on GOP
- Transformer model configuration
```

- Stage 1: Pre-processing raw videos
    - Before extracting features, all raw videos from all datasets are resized to `240` on its smallest edge and compressed using the H.264 codec with KeyInt set to `40`. We use the code provided by CoCap `<cite>`.
    - Chúng ta lần lượt sample `9, 13, 9` GOPs cho các preprocessed video trong MSVD, MSR-VTT, và VATEX dataset. Đây là các giá trị ở 75th percentile sau khi phân tích số lượng GOP trong training set của mỗi dataset. Mỗi GOP sẽ có 1 I-frame và khoảng `39` P-/B-frame.
- Stage 2: Multimodel Feature Extraction
    - Quá trình rút trích Multimodel Feature sẽ được thực hiện tương đồng trên tất cả GOP.
    - Đầu tiên, chúng ta sẽ dùng model **BLIP2-OPT-2.7b** trong **Hugging Face Transformers library** `<cite>` để generate 1 short caption cho I-frame của GOP. Trong quá trình này, bên cạnh caption cho I-frame, chúng ta cũng sẽ sử dụng [CLS] token được tạo ra bởi Image Encoder để làm **visual feature** cho GOP. Mỗi [CLS] token là 1 vector có dimension $d_V=1408$
    - Để trích xuất **semantic feature**, caption từ I-frame sau đó sẽ trở thành đầu vào cho mô hình **all-RoBERTa-large-v1** từ **Sentence Transformers library** `<cite>`để tạo ra một vector sentence embedding có dimension $d_S=1024$.
    - Để trích xuất motion feature cho mỗi GOP, chúng ta sẽ sử dụng mô hình **MViTv2-small** được cung cấp sẵn trong **Torchvision library**. Vì mục tiêu của chúng ta là rút trích đặc trưng nên chúng ta sẽ bỏ qua layer cuối cùng của mô hình (dùng để dự đoán). Đối với mỗi GOP, chúng ta sẽ uniform sampling 16 frame (cách đều nhau) để làm đầu vào cho mô hình này. Kết quả thu được là 1 vector motion feature có dimension $d_M=768$.
    - Vì quá trình rút trích đặc trưng này tốn rất nhiều thời gian, nên sau khi kết thúc quá trình rút trích đặc trưng, chúng ta sẽ lưu trữ chúng với định dạng HDF5. Khi cần sử dụng chúng để huấn luyện mô hình, chúng ta chỉ cần đơn giản là load lại dữ liệu từ các file HDF5 này để tiết kiệm thời gian.
- Stage 3: Training model and evaluation
    - Our model is implemented using PyTorch framework
    - For the transformer network, model dimension $d_model$ is set as `512`, hidden dimension in feed-forward module $d_ff$ is set as `2048`, number of heads in multi-head attention module is set as `4`. The layers of both decoders (backward and forward decoder) are set as `3` layers.
    - For the training process, we set the batch size to `64` and the training epochs to `16` for all three datasets. We use Adam optimizer with initial learning rate of $1e-4$, $\beta_1=0.9$, $\beta_2=0.999$, weight_decay = $0.5e-5$ to train our proposed model and select the best model parameters according to the model performance on the validation set. We also adopt warmup strategy in the first `3` epochs in the training process. During training, the hyper-parameter $\lambda$ used to balance the preferences between the two decoders is set to `0.6`. Other regularization techniques include dropout rate is `0.1` and label smoothing ratio is `0.15`.
    - In the decoder module, we remove punctuation in every caption and filter words whose counts are less than `3` for all datasets. The maximum length of caption is set as `20` for all three datasets. In order to generate a better caption, we leverage the beam search with a size of `4` for both decoders.
    - All experiments were run on the `NVIDIA A100` GPUs.
