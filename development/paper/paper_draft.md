# 3. Background

## 3.1. Transformer Architecture Overview

Kiến trúc Transformer tiêu chuẩn ban đầu được thiết kế bao gồm một khối encoder và một khối decoder, mỗi khối được cấu tạo từ việc xếp chồng các layer giống hệt nhau (stack of identical layers). Tuy nhiên, trong nhiều biến thể hiện đại, kiến trúc này có thể được điều chỉnh (ví dụ: chỉ sử dụng decoder) để phù hợp với từng bài toán cụ thể. Cấu trúc cốt lõi bên trong các layer này dựa trên cơ chế chú ý (attention mechanism) và các mạng truyền thẳng (Feed-Forward Networks - FFN), được hỗ trợ bởi các kết nối tắt (residual connections) và tầng chuẩn hóa (layer normalization).

### 3.1.1. Attention Mechanism: Self-Attention and Cross-Attention

Cốt lõi của Transformer là cơ chế **Scaled Dot-Product Attention**. Cơ chế này tính toán attention score bằng cách ánh xạ một tập hợp các giá trị Query ($Q$), Key ($K$) và Value ($V$). Công thức toán học tổng quát được biểu diễn như sau:

$$
\begin{align}
\text{Attention}(Q, K, V) &= \text{Softmax}\left(\frac{Q * K^T}{\sqrt{d_k}}\right) * V 
\end{align}
$$

Trong đó, $\sqrt{d_k}$ là hệ số tỷ lệ (scaling factor) dựa trên số chiều của Key, giúp tránh tình trạng giá trị của tích vô hướng quá lớn làm cho đạo hàm của hàm $\text{Softmax}$ bị triệt tiêu (vanishing gradient).

Thay vì chỉ thực hiện hàm attention một lần, các mô hình Transformer thường sử dụng **Multi-Head Attention (MHA)** để cho phép mô hình đồng thời chú ý (attend) đến các thông tin từ nhiều không gian biểu diễn (representation subspaces) khác nhau. The MHA is calculated as follows:

$$ 
\begin{align}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) * W^O 
\end{align}
$$

$$ 
\begin{align}
\text{where } \text{head}_i &= \text{Attention}(Q*W_i^Q, K*W_i^K, V*W_i^V)
\end{align}
$$

where $\text{MultiHead}$ denotes multi-head attention, $\text{Concat}$ denotes concatenation operation, $h$ is the number of heads. Trong cơ chế MHA, mô hình sẽ học các ma trận chiếu (projection matrices) $W_i^Q$, $W_i^K$, $W_i^V$ cho mỗi *head* $i$, và ma trận chiếu đầu ra (output projection matrix) $W^O$ để tổng hợp thông tin.

**Phân biệt Self-Attention và Cross-Attention.** 
Based on the source of $Q, K, V$, attention modules in the Transformer can be categorized into two types: self‑attention and cross‑attention.

**Self-Attention.** In a self‑attention module, $Q, K, V$ are all derived from the same input sequence (e.g., the hidden states of previously generated tokens). Self‑attention enables the model to capture internal dependencies among elements within the same sequence.

**Cross-Attention.** Cross‑attention occurs when $Q, K, V$ originate from different sources. In this setting, $Q$ is taken from one representation (e.g., the current decoder state), while $K$ and $V$ are taken from another (e.g., encoder outputs). Cross‑attention acts as a routing mechanism that helps the attention module gather necessary semantic context from other information streams.

### 3.1.2. Feed-Forward Network with GELU Activation

Thành phần quan trọng thứ hai trong mỗi layer Transformer là Position-wise Feed-Forward Network. Mạng này thường bao gồm hai phép biến đổi tuyến tính (two linear transformations) xen kẽ bởi một hàm kích hoạt phi tuyến (non-linear activation function). Trong các kiến trúc Transformer truyền thống, hàm kích hoạt phổ biến nhất được sử dụng là ReLU. Tuy nhiên, trong nghiên cứu này, chúng ta sẽ sử dụng hàm kích hoạt GELU thay cho ReLU, following Google BERT and OpenAI GPT. Như vậy, quá trình tính toán của FFN với GELU activation có thể được biểu diễn bằng công thức sau:

$$
\begin{align}
\text{FFN}(X) &= \text{GELU}(X*W_1 + b_1)*W_2 + b_2
\\
\text{GELU}(x) &= x \cdot \Phi(x)
\end{align}
$$

Trong đó, $W_1$, $W_2$, $b_1$, và $b_2$ đều là các tham số có thể học được (learnable parameters), $\Phi(x)$ là hàm phân phối tích lũy (Cumulative Distribution Function - CDF) của phân phối chuẩn tắc (Standard Gaussian).

## 3.2. Peri-Normalization in Transformer

Trong quá trình huấn luyện các mô hình Transformer, việc lựa chọn chiến lược chuẩn hóa (normalization strategy) đóng vai trò cốt lõi trong việc kiểm soát tính ổn định của gradient và tốc độ hội tụ. Until recently, hai chiến lược **Post-LN** và **Pre-LN** vẫn được sử dụng phổ biến, mặc dù chúng vẫn tồn tại một số hạn chế. Cụ thể, chiến lược Post-LN (chuẩn hóa sau khi cộng dồn residual) giúp giới hạn phương sai (variance) của hidden states nhưng lại làm suy yếu luồng gradient trong các mạng sâu, dẫn đến hiện tượng triệt tiêu gradient (vanishing gradients) và làm chậm quá trình hội tụ. Ngược lại, Pre-LN (chuẩn hóa trước khi đưa vào module) cải thiện tốt sự lưu thông của gradient, nhưng lại không chuẩn hóa luồng thông tin chính (main residual path). Điều này dẫn đến sự tích lũy phương sai theo cấp số nhân qua các layer, gây ra hiện tượng "massive activations" (các giá trị kích hoạt khổng lồ) làm mất ổn định quá trình huấn luyện.

Gần đây, một chiến lược thứ ba đã lặng lẽ xuất hiện trong các mô hình mã nguồn mở (như Gemma, OLMo) nhằm khắc phục nhược điểm của cả hai phương pháp trên. Kim et al. đã hệ thống hóa và đặt tên cho thiết kế này là **Peri-LN**.

Về mặt định nghĩa, **Peri-LN** places normalization layers peripherally around modules in Transformer architecture. Nói cách khác, Peri-LN bao gồm việc bổ sung thêm một layer normalization (Output-LN) ở đầu ra của module, kết hợp với cấu trúc Pre-LN truyền thống.

For clarity, the locations of normalization layers in the Post-, Pre-, and Peri-LN architectures are illustrated in Figure ???. Về mặt toán học, cho trước $x_l$ là hidden state đầu vào của layer thứ $l$, vị trí của các normalization layer $\text{Norm}$ trong ba kiến trúc được biểu diễn như sau:

*   **Post-LN:** $y_l = \text{Norm} \big(x_l + \text{Module}(x_l)\big)$
*   **Pre-LN:** $y_l = x_l + \text{Module}\big(\text{Norm}(x_l)\big)$
*   **Peri-LN:** $y_l = x_l + \text{Norm}\Big(\text{Module}\big(\text{Norm}(x_l)\big)\Big)$

Thiết kế của Peri-LN hợp nhất cả pre-normalization và output-normalization để điều chỉnh phương sai từ cả hai đầu của module. Nhờ có thêm một normalization layer ở đầu ra, Peri-LN hoạt động như một cơ chế tự điều chuẩn (self-regularizing mechanism), tạo ra một hệ số cản (damping factor) giúp giới hạn sự gia tăng đột biến của gradient ngay cả khi module sinh ra các giá trị kích hoạt lớn.

Through in-depth analysis, Peri-LN đã được chứng minh có khả năng liên tục đạt được sự tăng trưởng phương sai cân bằng hơn (more balanced variance growth), luồng gradient ổn định hơn (steadier gradient flow), và cải thiện độ ổn định hội tụ (convergence stability) so với cả Post-LN và Pre-LN. Do đó, our proposed Transformer architecture will use Peri-LN strategy.

# 4. Methods

- structure of compressed video
- multimodal features extraction: blip2 + mvitv2 + sentence-transfomer
- Bidirectional Decoder Transformer: multimodal feature embedding + backward decoder + forward decoder
- Optimization and Inference

## 4.1. Cấu trúc của các digital video hiện đại

<!-- #### ba loại khung hình cơ bản của 1 video -->

In modern video codecs such as H.264 and H.265, leveraging temporal redundancy between consecutive frames is crucial for optimizing storage and bandwidth. Based on data dependencies, frames within a video are generally categorized into three primary types: I-frame, P-frame and B-frame.

**I-frame (Intra-coded frame)**, còn được gọi là keyframe, is encoded independently using intra-prediction, meaning it does not rely on any other frames for reconstruction. Consequently, I-frame contains complete appearance information, capturing the full context and objects within a scene. Trong khi đó, 2 loại khung hình còn lại (P-frame và B-frame) sẽ được mã hóa using inter-prediction by referring to the other frames. 
	
**P-frame (Predictive coded frame)** là khung hình dự đoán một hướng. Nó chỉ lưu trữ sự khác biệt về hình ảnh so với một khung hình tham chiếu đứng trước nó (có thể là I-frame hoặc một P-frame khác).
	
**B-frame (Bipredictive coded frame)** là khung hình dự đoán hai hướng. B-frame có khả năng tham chiếu thông tin từ cả khung hình trong quá khứ và khung hình trong tương lai (xét theo trình tự hiển thị) để tìm ra sự tương đồng lớn nhất. Điều này đồng nghĩa với việc B-frame cung cấp tỷ lệ nén cao nhất.

<!-- #### Cấu trúc Group of Pictures (GOP) và Cơ chế Token hóa -->

**Group of Pictures** Các khung hình I-frame, P-frame và B-frame không được sắp xếp một cách tùy tiện mà tuân theo một mô hình lặp lại có tính chu kỳ được gọi là cấu trúc GOP (Group of Pictures). Như được trình bày trong hình ???, mỗi GOP luôn bắt đầu bằng một I-frame và bao gồm tất cả các P-frame và B-frame theo sau cho đến khi gặp một I-frame tiếp theo. A digital video consists of a continuous sequence of GOPs.

Thông thường, một GOP là một đơn vị bitstream được encode và decode một cách độc lập, điều này có nghĩa là các frame trong 1 GOP sẽ không refer đến bất kỳ frame nào trong GOP khác. This autonomy allows us to treat each GOP as a distinct "unit of information".

In our Transformer-based architecture for video captioning, rather than processing every individual frame - which is computationally expensive - we represent each GOP as a single token. This approach enables the model to efficiently capture local motion dynamics within each GOP before aggregating them into the global semantics of the entire video.

<!-- #### Sampling Strategy -->

To ensure both representativeness and computational efficiency, we employ a uniform sampling method to extract (tối đa) $N$ GOPs from each video (Giá trị tối ưu của $N$ sẽ được lựa chọn trong quá trình thực nghiệm). The internal structure of each GOP is determined by the $\text{KeyInt}$ hyperparameter (the distance between I-frames) during the encoding process.

Specifically, each GOP comprises $M$ frames, where: the first frame is always an I-frame and the remaining $\text{KeyInt} - 1$ frames are P/B-frames. By controlling the parameters $N$ and $\text{KeyInt}$, we can achieve an optimal balance between the granularity of the extracted features and the model's overall execution speed.

## 4.2. Multimodal Feature Extraction

2D visual features and 3D motion features have been widely used by prior work for multimodal features extraction in video captioning task. Do đó, trong nghiên cứu này, chúng ta cũng sẽ tiếp tục khai thác hai loại đặc trưng này. Tuy nhiên, để nâng cao khả năng biểu diễn và nắm bắt ngữ nghĩa của video, chúng ta sẽ bổ sung thêm một loại đặc trưng thứ ba: semantic feature.

Như đã trình bày ở phần trước, mỗi video có thể được biểu diễn dưới dạng một chuỗi các GOP. Giả sử, xét một video $X$ gồm $N$ GOP, chúng ta có thể biểu diễn video này dưới dạng một chuỗi các GOP như sau:
$$X = [\text{GOP}^{(1)}, \dots, \text{GOP}^{(N)}]$$

Quá trình trích xuất đặc trưng được **thực hiện đồng nhất** trên mọi GOP để đảm bảo tính nhất quán của dữ liệu đầu vào cho mô hình video captioning của chúng ta.

<!-- Đặc trưng diện mạo và Ngữ nghĩa từ I-frame -->

Trong mỗi GOP, I-frame đóng vai trò là anchor point chứa đựng lượng thông tin thị giác phong phú nhất. Thay vì phải trích xuất đặc trưng cho toàn bộ khung hình (có thể gây ra sự dư thừa về dữ liệu và không thực sự đóng góp quá nhiều vào hiệu suất tổng thể), chúng ta chỉ tập trung vào việc trích xuất đặc trưng từ I-frame. Chúng ta sẽ xây dựng một pipeline đơn giản, tận dụng sức mạnh của hai pre-trained models BLIP-2 và all-RoBERTa-large-v1, để thu thập hai loại đặc trưng: visual feature và semantic feature.

**Visual feature** được trích xuất từ $[\text{CLS}]$ token của Image Encoder trong BLIP-2. Mỗi $[\text{CLS}]$ token là một vector cung cấp thông tin thô về các đối tượng và bối cảnh không gian.

Đồng thời, chúng ta cũng tận dụng khả năng sinh ngôn ngữ của BLIP-2 để tạo ra một câu mô tả ngắn (caption) cho I-frame. Để chuyển đổi thông tin văn bản (sentence or sequence of words) này thành một vector trong - Embedding space (mà mô hình có thể hiểu được), câu mô tả này được đưa qua mô hình all-RoBERTa-large-v1 để thu được một vector sentence embedding. Vector này cung cấp một lớp dẫn dắt ngữ nghĩa cấp cao, giúp mô hình nắm bắt được các khái niệm trừu tượng có trong khung hình và chúng ta gọi đây là **semantic feature**.

<!-- Đặc trưng chuyển động không gian - thời gian -->

Để thu thập các tương tác động (spatio-temporal motion dynamics) diễn ra giữa các khung hình, chúng ta sử dụng pre-trained model MViTv2 để rút trích **motion feature**. Mô hình này được áp dụng trên toàn bộ chuỗi khung hình trong mỗi GOP để rút trích ra một vector tóm tắt các thay đổi về mặt cử động, hành động của đối tượng từ các P-frame và B-frame, bổ khuyết cho thông tin tĩnh từ I-frame.

<!-- Biểu diễn toán học của chuỗi đặc trưng -->

Như vậy, đối với mỗi video, sau khi kết thúc quá trình trích xuất multimodal feature cho tất cả $N$ GOP, chúng ta sẽ thu được ba loại đặc trưng là: (1) visual feature $F_V = [v^{(1)},\dots, v^{(N)}]$ where $v^{(n)} \in \mathbb{R}^{d_V}$ and $F_V \in \mathbb{R}^{N \times d_V}$; (2) semantic feature $F_S = [s^{(1)},\dots, s^{(N)}]$ where $s^{(n)} \in \mathbb{R}^{d_S}$ and $F_S \in \mathbb{R}^{N \times d_S}$; và (3) motion feature $F_M = [m^{(1)},\dots, m^{(N)}]$ where $m^{(n)} \in \mathbb{R}^{d_M}$ and $F_M \in \mathbb{R}^{N \times d_M}$. Trong đó, $d_V$, $d_S$, và $d_M$ lần lượt là kích thước (dimension) của không gian đặc trưng, và giá trị cụ thể của chúng phụ thuộc vào kiến trúc của các mô hình pre-trained mà chúng ta sử dụng. Chúng ta sẽ đề cập đến các giá trị này trong phần thực nghiệm.

## 4.3. Phương pháp đề xuất: Bidirectional Decoder Transformer (BiDecT)

Kiến trúc Bidirectional Decoder Transformer (BiDecT) của chúng ta sẽ sử dụng hướng tiếp cận tập trung vào việc khai thác tối đa ngữ cảnh hai chiều trong quá trình sinh mô tả (caption) cho video. Điểm khác biệt cốt lõi của BiDecT so với các nghiên cứu trước đây là việc loại bỏ hoàn toàn khối Encoder trung gian, thay vào đó trực tiếp tích hợp các đặc trưng đa phương thức (multimodal feature) vào hệ thống Bidirectional Decoder.

<!-- Triết lý thiết kế kiến trúc -->

Khác với các mô hình Bidirectional transformer truyền thống vốn dựa trên cấu trúc Encoder-Decoder phức tạp, kiến trúc BiDecT được tinh gọn để chỉ bao gồm hai thành phần chính: Backward Decoder (giải mã ngược) và Forward Decoder (giải mã xuôi).

Lý do cho việc loại bỏ hoàn toàn module Encoder là dựa trên giả thuyết rằng các đặc trưng đa phương thức được trích xuất từ các mô hình SOTA đã mang đủ hàm lượng thông tin biểu diễn (representation power). Do đó, việc đưa trực tiếp các đặc trưng này vào khối Decoder thông qua cơ chế Cross-Attention không chỉ giúp giảm bớt khối lượng tính toán mà còn hạn chế việc mất mát thông tin khi phải đi qua quá nhiều tầng (layer) biến đổi của Encoder.

<!-- Module Multimodal Feature Embedding tách biệt -->

Để đảm bảo tính độc lập và khả năng học đặc thù cho từng chiều giải mã, chúng ta xây dựng hai module Multimodal Feature Embedding riêng biệt. Mặc dù hai module này có cấu trúc tương tự nhau, việc cài đặt bộ trọng số riêng biệt là cực kỳ quan trọng. Điều này giúp luồng thông tin ngược (backward flow) không bị lẫn lộn với luồng thông tin xuôi (forward flow), cho phép mỗi Decoder tối ưu hóa việc học theo đặc thù của chiều dữ liệu tương ứng.

<!-- Cơ chế học hai chiều (Bidirectional Learning Mechanism) -->

Module **Backward Decoder (BD)** đảm nhận việc sinh ra câu mô tả theo chiều từ phải sang trái (Right-to-Left - R2L) $Y_{BD}$ cho video. Đặc biệt, trong quá trình này, Backward Decoder không chỉ học cách dự đoán từ tiếp theo (theo chiều ngược) mà còn nén toàn bộ thông tin ngữ nghĩa của video thành một context feature $\overleftarrow{H}$. Đặc trưng này đại diện cho cái nhìn "tổng quan từ phía sau", giúp định hình cấu trúc câu và ý tưởng cốt lõi trước khi quá trình giải mã chính diễn ra.

Module **Forward Decoder** (FD) nhận đầu vào là: Các đặc trưng đa phương thức trích xuất từ video và context feature $\overleftarrow{H}$ thu được từ Backward Decoder để tạo ra caption cuối cùng theo chiều từ trái sang phải (Left-to-Right - L2R) $Y_{FD}$, đây chính là caption thực sự cho video. Sự kết hợp này tạo ra một cơ chế dẫn dắt kép: đặc trưng video cung cấp nội dung thị giác, trong khi $\overleftarrow{H}$ cung cấp định hướng ngữ nghĩa. Kết quả là Forward Decoder có thể sinh ra các câu mô tả có tính mạch lạc cao, bám sát diễn biến thời gian và mang cấu trúc ngôn ngữ tự nhiên hơn.

### 4.3.1. Multimodal features Embedding

Module này đóng vai trò là **interface layer** giữa quá trình trích xuất đặc trưng thô và khối Transformer Decoder. Nhiệm vụ chính của module là ánh xạ các đặc trưng từ các không gian vector khác nhau về một không gian biểu diễn chung (Common Representation Space) có số chiều (dimension) là $d_{model}$, đồng thời tích hợp các thông tin về ngữ cảnh thời gian và loại dữ liệu.

<!-- 4.1. Chiếu tuyến tính và Đồng bộ hóa số chiều (Linear Projection) -->

Do các pretrained models có cấu trúc khác nhau, nên các đặc trưng đầu vào $F_V$, $F_S$ và $F_M$ thường có số chiều không đồng nhất. Chúng ta sẽ áp dụng các phép biến đổi tuyến tính (linear transformation) riêng biệt cho từng loại đặc trưng để đưa chúng về cùng một không gian vector có kích thước $d_{model}$:

$$
\begin{align}
F'_V &= F_V * W_V + b_V \\
F'_S &= F_S * W_S + b_S \\
F'_M &= F_M * W_M + b_M
\end{align}
$$

Trong đó, ba ma trận trọng số $W_V \in \mathbb{R}^{d_V \times d_{model}}$, $W_S \in \mathbb{R}^{d_S \times d_{model}}$, $W_M \in \mathbb{R}^{d_M \times d_{model}}$ và ba vector bias $b_V, b_S, b_M \in \mathbb{R}^{d_{model}}$ đều là các tham số có thể học được (learnable parameters). Kết quả thu được là các tập đặc trưng đã được chiếu về cùng một không gian vector: $F'_V, F'_S, F'_M \in \mathbb{R}^{N \times d_{model}}$.

<!-- Tích hợp Type Embedding và Positional Embedding -->

Lấy cảm hứng từ cơ chế biểu diễn đầu vào của BERT, chúng ta bổ sung thêm hai thành phần bổ trợ để làm giàu thông tin cho các vector đặc trưng là: type embedding và positional embedding.

**Type Embeddings** $\text{(TypeEmb)}$ giúp mô hình phân biệt được nguồn gốc của từng loại đặc trưng (Modality). Chúng ta sử dụng một bảng tra cứu (lookup table) để gán một vector nhúng định danh riêng cho từng loại đặc trưng Visual, Semantic và Motion. Điều này ngăn chặn sự nhầm lẫn về mặt ngữ nghĩa khi các đặc trưng được trộn lẫn trong chuỗi đầu vào.

Về **Positional Embeddings** $\text{(PosEmb)}$, because transformers don’t inherently have a built-in sense of sequential order like recurrent models, chúng ta sử dụng các hàm Sine và Cosine để mã hóa thứ tự thời gian của các GOP. Việc cộng chung một vector vị trí cho cả ba đặc trưng trong cùng một GOP giúp mô hình nhận diện được tính đồng thời (temporal synchrony) của chúng.

Quá trình tổng hợp và chuẩn hóa được thực hiện qua các layer **Layer Normalization** $\text{(LayerNorm)}$ riêng biệt nhằm ổn định thang đo dữ liệu giữa các phương thức (modality):

$$
\begin{align}
E_V &= \text{LayerNorm}_V(F'_V + \text{TypeEmb}_V + \text{PosEmb}) \\
E_S &= \text{LayerNorm}_S(F'_S + \text{TypeEmb}_S + \text{PosEmb}) \\
E_M &= \text{LayerNorm}_M(F'_M + \text{TypeEmb}_M + \text{PosEmb})
\end{align}
$$

Trong đó, $\text{TypeEmb}$ và $\text{LayerNorm}$ đều có các tham số có thể học được (learnable) trong quá trình huấn luyện. Kết quả thu được là các chuỗi embedding đã được chuẩn hóa: $E_V, E_S, E_M \in \mathbb{R}^{N \times d_{model}}$.

<!-- Biểu diễn chuỗi tổng hợp (Final Representation) -->

Cuối cùng, các vector nhúng của từng GOP được nối xen kẽ để tạo thành một chuỗi embedding tổng hợp $E \in \mathbb{R}^{3N \times d_{model}}$. 

Về mặt toán học, nếu ta biểu diễn mỗi chuỗi embedding đã được chuẩn hóa của từng loại đặc trưng như sau:

$$
\begin{align}
E_V &= [e_V^{(1)},\dots, e_V^{(N)}]
\\
E_S &= [e_S^{(1)},\dots, e_S^{(N)}]
\\
E_M &= [e_M^{(1)},\dots, e_M^{(N)}]
\end{align}
$$

thì chuỗi embedding tổng hợp:

$$
\begin{align}
E &= \text{Concat}(E_V, E_S, E_M) \\
  &= [e_{V}^{(1)}, e_{S}^{(1)}, e_{M}^{(1)}, \dots, e_{V}^{(N)}, e_{S}^{(N)}, e_{M}^{(N)}]
\end{align}
$$

Cấu trúc này đảm bảo rằng với mỗi GOP, Decoder sẽ nhận được đầy đủ các thông tin từ diện mạo, ngữ nghĩa đến chuyển động. Quy trình này được cài đặt độc lập cho cả Backward Decoder ($\overleftarrow{E}$) và Forward Decoder ($\overrightarrow{E}$) với các bộ trọng số riêng biệt, giúp tối ưu hóa việc học theo từng chiều giải mã cụ thể.

### 4.3.2. Backward decoder (BD)

The Backward Decoder (BD) follows the standard Transformer decoder architecture but taking the reference caption presented in reverse order as a part of input to capture the right-to-left context.

Với $\overleftarrow{E}$ là multimodal feature embedding dành riêng cho nhánh BD. Ta ký hiệu chuỗi từ được dự đoán (the predicted words) trong $t'-1$ time steps đầu tiên là $\overleftarrow{\hat{Y}_{<t'}}=\{\overleftarrow{\hat{y}_1},\dots,\overleftarrow{\hat{y}_{t'-1}}\}$ và ma trận vector nhúng (word embedding vectors) tương ứng là $\overleftarrow{C_{<t'}} = \{\overleftarrow{{c}_1},\dots,\overleftarrow{{c}_{t'-1}}\}$. Để dự đoán từ $\overleftarrow{\hat{y}_{t'}}$ tại time step $t'$, quy trình tính toán tổng quát của BD có thể được mô tả như sau:

$$
\begin{align}
\overleftarrow{A_{<t'}} &= \text{Attention}(\overleftarrow{C_{<t'}}, \overleftarrow{C_{<t'}}, \overleftarrow{C_{<t'}})
\\
\overleftarrow{D_{t'}} &= \text{Attention}(\overleftarrow{A_{<t'}}, \overleftarrow{E}, \overleftarrow{E})
\\
\overleftarrow{H_{t'}} &= \text{FFN}(\overleftarrow{D_{t'}})
\end{align}
$$

Trong đó, $\overleftarrow{A_{<t'}}$, $\overleftarrow{D_{t'}}$ và $\overleftarrow{H_{t'}}$ lần lượt là đầu ra của self-attention module, cross-attention module và feed-forward module. Các module $\text{Attention}$ và $\text{FFN}$ này đều tích hợp Peri-LN strategy và có bổ sung residual connection. 

Then, we use a linear layer followed by softmax activation function to calculate the probability distribution of the predicted word at the current time step $t'$:

$$
\begin{align}
P(\overleftarrow{\hat{y}_{t'}} \mid \overleftarrow{\hat{y}_{<t'}}, \overleftarrow{E}) &= \text{Softmax}(\text{Linear}(\overleftarrow{H_{t'}}))
\end{align}
$$

When the predicted word is the end marker $\langle \text{S} \rangle$, the generation of the reverse caption terminates. More importantly, the backward decoder produces the final hidden state $\overleftarrow{H}$, which encodes semantic context in a right-to-left manner. This hidden representation is subsequently utilized by the forward decoder.

### 4.3.3. Forward decoder (FD)

The forward decoder in our model is adapted from the conventional Transformer decoder, operating in a left-to-right manner. Its decoding process is guided by both the multimodal feature representation $\overrightarrow{E}$ and the right-to-left semantic context $\overleftarrow{H}$. Compared to the backward decoder, the forward decoder incorporates an additional cross-attention module that attends to the hidden states generated by the backward decoder. This design allows the forward decoder to leverage the semantic context of the ground-truth caption every time it predicts the next word.

Given the words generated at the previous $t − 1$ time step, denoted as $\overrightarrow{\hat{Y}_{<t}}=\{\overrightarrow{\hat{y}_1},\dots,\overrightarrow{\hat{y}_{t-1}}\}$, and their corresponding word vectors $\overrightarrow{C_{<t}} = \{\overrightarrow{{c}_1},\dots,\overrightarrow{{c}_{t-1}}\}$, the word $\overrightarrow{\hat{y}_{t}}$ predicted at time step $t$ by the forward decoder can be formulated as follows:

$$
\begin{align}
\overrightarrow{A_{<t}} &= \text{Attention}(\overrightarrow{C_{<t}}, \overrightarrow{C_{<t}}, \overrightarrow{C_{<t}}) \\

\overrightarrow{D_{t}} &= \text{Attention}(\overrightarrow{A_{<t}}, \overrightarrow{E}, \overrightarrow{E}) \\

\overrightarrow{D'_{t}} &= \text{Attention}(\overrightarrow{D_{t}}, \overleftarrow{H}, \overleftarrow{H}) \\

\overrightarrow{H_{t}} &= \text{FFN}(\overrightarrow{D'_{t}})
\end{align}
$$

Trong đó, $\overrightarrow{A_{<t}}$, $\overrightarrow{D_{t}}$, $\overrightarrow{D'_{t}}$ và $\overrightarrow{H_{t}}$ lần lượt là đầu ra của self-attention module, cross-attention module với multimodal feature embedding $\overrightarrow{E}$, cross-attention module với hidden state $\overleftarrow{H}$ của backward decoder và feed-forward module. Tương tự như trong BD, các module $\text{Attention}$ và $\text{FFN}$ trong FD đều tích hợp cấu trúc Peri-LN strategy và residual connection. Module cross-attention bổ sung ($\overrightarrow{D'_{t}}$) đóng vai trò là cầu nối cốt lõi giúp FD hòa trộn thông tin từ tương lai (right-to-left) vào quá trình sinh từ hiện tại.

Then, the probability distribution of the predicted word is calculated as:

$$
\begin{align}
P(\overrightarrow{\hat{y}_{t}} \mid \overrightarrow{\hat{y}_{<t}}, \overrightarrow{E}, \overleftarrow{H}) &= \text{Softmax}(\text{Linear}(\overrightarrow{H_{t}}))
\end{align}
$$

When the predicted word is the end marker $\langle \text{S} \rangle$, the generation of the caption terminates. The forward decoder produces the final caption, which is expected to be more accurate and fluent than the one generated by the backward decoder alone, as it benefits from both the multimodal feature representation and the robust semantic context provided by the backward decoder.

## 4.4. Optimization and Inference

We train the BiDecT model using the cross-entropy losses on both decoders. The training process is designed to optimize the parameters of both decoders simultaneously, allowing them to generate high-quality captions in both directions.

Let $D$ be the training set of videos with provided ground-truth captions. For each video $X$, its ground-truth caption $\overrightarrow{Y}=\{\overrightarrow{y_1},\dots,\overrightarrow{y_l}\}$ of length $l$, and the pseudo reverse caption $\overleftarrow{Y}=\{\overleftarrow{y_1},\dots,\overleftarrow{y_{l'}}\}$ of length $l'$ from the training set $D$, hàm loss tổng thể $L$ của mô hình BiDecT được định nghĩa như sau:

$$
\begin{align}
L = (1-\lambda)\,L_{BD} + \lambda\,L_{FD},
\end{align}
$$

$$
\begin{align}
L_{BD} = \sum_{(X,\overleftarrow{Y})\in D}\;\sum_{t'=1}^{l'} -\log \,P(\overleftarrow{y_{t'}} \mid \overleftarrow{y_{<t'}},\,X; \;\theta_{BD}),
\end{align}
$$

$$
\begin{align}
L_{FD} = \sum_{(X,\overrightarrow{Y})\in D}\;\sum_{t=1}^{l} -\log \,P(\overrightarrow{y_{t}} \mid \overrightarrow{y_{<t}},\,X;\;\theta_{BD},\,\theta_{FD}),
\end{align}
$$

where $L_{BD}$ and $L_{FD}$ denote the cross-entropy losses of the backward and forward decoders, respectively; and $\overleftarrow{y_{t'}}$ and $\overrightarrow{y_{t}}$ denote the reference words for the backward and forward decoders; and $\lambda \in [0,1]$ is a hyperparameter that balances the two terms. The trainable parameters of the backward and forward decoders are denoted by $\theta_{BD}$ and $\theta_{FD}$, respectively; each $\theta$ also includes the trainable parameters of the corresponding multimodal feature embedding module. The first term, $L_{BD}$, encourages the model to learn a high-quality right-to-left semantic representation. The second term, $L_{FD}$, models the forward caption generation process. For implementation, we apply **label smoothing** to reduce overconfidence.

**Pseudo Reverse Captions.** Following prior studies on bidirectional decoding, we must avoid trivial information leakage that would arise if reverse candidates were obtained by simply reversing the exact forward ground-truth captions. This leakage occurs when the forward decoder (FD) can exploit future answers provided by the backward decoder (BD) instead of learning to integrate contextual cues.

To prevent this, the reverse captions used during training are **pseudo reverse captions**. Each video in the dataset typically has multiple ground-truth captions. For each video, we first reverse all of its forward captions (i.e., flip the word order) to form a pool of candidate reversed captions specific to that video. These reversed captions are then randomly shuffled and reassigned within the same pool for that video, so that the reverse caption paired with a given video is not necessarily the exact reversal of its corresponding forward caption. Crucially, reversed captions are not exchanged across different videos. In other words, the backward decoder is not fed the exact future ground-truth during training. This intra-video randomization mitigates information leakage and prevents the forward decoder from exploiting memorized exact reversals, thereby encouraging the model to learn genuine bidirectional semantic context.

**Testing / Inference.** Once BiDecT is trained, the caption generation during testing proceeds in two sequential stages. In order to ensure consistency between model training and testing, we apply beam search decoding in both stages. In the first stage, the backward decoder utilizes beam search to sequentially generate a reverse caption and terminates when the end marker $\langle S\rangle$ is predicted with maximum probability, producing the final reverse hidden state. In the second stage, the forward decoder applies beam search in a left-to-right manner to produce the final, best video caption.
