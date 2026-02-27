# 3. Methods

- structure of compressed video
- multimodal features extraction: blip2 + mvitv2 + sentence-transfomer
- Bidirectional Decoder Transformer: offline feature extraction + backward decoder + forward decoder
- Training: loss

## 3.1. Cấu trúc của các digital video hiện đại

<!-- #### ba loại khung hình cơ bản của 1 video -->

In modern video codecs such as H.264 and H.265, leveraging temporal redundancy between consecutive frames is crucial for optimizing storage and bandwidth. Based on data dependencies, frames within a video are generally categorized into three primary types: I-frame, P-frame and B-frame.

**I-frame (Intra-coded frame)**, còn được gọi là keyframe, is encoded independently using intra-prediction, meaning it does not rely on any other frames for reconstruction. Consequently, I-frame contains complete appearance information, capturing the full context and objects within a scene. Trong khi đó, 2 loại khung hình còn lại (P-frame và B-frame) sẽ được mã hóa using inter-prediction by referring to the other frames. 
	
**P-frame (Predictive coded frame)** là khung hình dự đoán một hướng. Nó chỉ lưu trữ sự khác biệt về hình ảnh so với một khung hình tham chiếu đứng trước nó (có thể là I-frame hoặc một P-frame khác).
	
**B-frame (Bipredictive coded frame)** là khung hình dự đoán hai hướng. B-frame có khả năng tham chiếu thông tin từ cả khung hình trong quá khứ và khung hình trong tương lai (xét theo trình tự hiển thị) để tìm ra sự tương đồng lớn nhất. Điều này đồng nghĩa với việc B-frame cung cấp tỷ lệ nén cao nhất.

<!-- #### Cấu trúc Group of Pictures (GOP) và Cơ chế Token hóa -->

**Group of Pictures**$\thickspace$ Các khung hình I-frame, P-frame và B-frame không được sắp xếp một cách tùy tiện mà tuân theo một mô hình lặp lại có tính chu kỳ được gọi là cấu trúc GOP (Group of Pictures). Như được trình bày trong hình ???, mỗi GOP luôn bắt đầu bằng một I-frame và bao gồm tất cả các P-frame và B-frame theo sau cho đến khi gặp một I-frame tiếp theo. A digital video consists of a continuous sequence of GOPs.

Thông thường, một GOP là một đơn vị bitstream được encode và decode một cách độc lập, điều này có nghĩa là các frame trong 1 GOP sẽ không refer đến bất kỳ frame nào trong GOP khác. This autonomy allows us to treat each GOP as a distinct "unit of information".

In our Transformer-based architecture for video captioning, rather than processing every individual frame - which is computationally expensive - we represent each GOP as a single token. This approach enables the model to efficiently capture local motion dynamics within each GOP before aggregating them into the global semantics of the entire video.

<!-- #### Sampling Strategy -->

To ensure both representativeness and computational efficiency, we employ a uniform sampling method to extract (tối đa) $N$ GOPs from each video (Giá trị tối ưu của $N$ sẽ được lựa chọn trong quá trình thực nghiệm). The internal structure of each GOP is determined by the $\text{KeyInt}$ hyperparameter (the distance between I-frames) during the encoding process.

Specifically, each GOP comprises $M$ frames, where: the first frame is always an I-frame and the remaining $\text{KeyInt} - 1$ frames are P/B-frames. By controlling the parameters $N$ and $\text{KeyInt}$, we can achieve an optimal balance between the granularity of the extracted features and the model's overall execution speed.

## 3.2. Multimodal Feature Extraction

2D visual features and 3D motion features have been widely used by prior works for multimodal features extraction in video captioning task. Do đó, trong nghiên cứu này, chúng ta cũng sẽ tiếp tục khai thác hai loại đặc trưng này. Tuy nhiên, để nâng cao khả năng biểu diễn và nắm bắt ngữ nghĩa của video, chúng ta sẽ bổ sung thêm một loại đặc trưng thứ ba: semantic feature.

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

## 3.3. Phương pháp đề xuất: Bidirectional Decoder Transformer (BiDecT)

Kiến trúc Bidirectional Decoder Transformer (BiDecT) của chúng ta sẽ sử dụng hướng tiếp cận tập trung vào việc khai thác tối đa ngữ cảnh hai chiều trong quá trình sinh mô tả (caption) cho video. Điểm khác biệt cốt lõi của BiDecT so với các nghiên cứu trước đây là việc loại bỏ hoàn toàn khối Encoder trung gian, thay vào đó trực tiếp tích hợp các đặc trưng đa phương thức (multimodal feature) vào hệ thống Bidirectional Decoder.

<!-- Triết lý thiết kế kiến trúc -->

Khác với các mô hình Bidirectional transformer truyền thống vốn dựa trên cấu trúc Encoder-Decoder phức tạp, kiến trúc BiDecT được tinh gọn để chỉ bao gồm hai thành phần chính: Backward Decoder (giải mã ngược) và Forward Decoder (giải mã xuôi).

Lý do cho việc loại bỏ hoàn toàn module Encoder là dựa trên giả thuyết rằng các đặc trưng đa phương thức được trích xuất từ các mô hình SOTA đã mang đủ hàm lượng thông tin biểu diễn (representation power). Do đó, việc đưa trực tiếp các đặc trưng này vào khối Decoder thông qua cơ chế Cross-Attention không chỉ giúp giảm bớt khối lượng tính toán mà còn hạn chế việc mất mát thông tin khi phải đi qua quá nhiều tầng (layer) biến đổi của Encoder.

<!-- Module Multimodal Feature Embedding tách biệt -->

Để đảm bảo tính độc lập và khả năng học đặc thù cho từng chiều giải mã, chúng ta xây dựng hai module Multimodal Feature Embedding riêng biệt. Mặc dù hai module này có cấu trúc tương tự nhau, việc cài đặt bộ trọng số riêng biệt là cực kỳ quan trọng. Điều này giúp luồng thông tin ngược (backward flow) không bị lẫn lộn với luồng thông tin xuôi (forward flow), cho phép mỗi Decoder tối ưu hóa việc học theo đặc thù của chiều dữ liệu tương ứng.

<!-- Cơ chế học hai chiều (Bidirectional Learning Mechanism) -->

Module **Backward Decoder (BD)** đảm nhận việc sinh ra câu mô tả theo chiều từ phải sang trái (Right-to-Left - R2L) $Y_{BD}$ cho video. Đặc biệt, trong quá trình này, Backward Decoder không chỉ học cách dự đoán từ tiếp theo (theo chiều ngược) mà còn nén toàn bộ thông tin ngữ nghĩa của video thành một context feature $F_{\overleftarrow{C}}$. Đặc trưng này đại diện cho cái nhìn "tổng quan từ phía sau", giúp định hình cấu trúc câu và ý tưởng cốt lõi trước khi quá trình giải mã chính diễn ra.

Module **Forward Decoder** (FD) nhận đầu vào là: Các đặc trưng đa phương thức trích xuất từ video và context feature $F_{\overleftarrow{C}}$ thu được từ Backward Decoder để tạo ra caption cuối cùng theo chiều từ trái sang phải (Left-to-Right - L2R) $Y_{FD}$, đây chính là caption thực sự cho video. Sự kết hợp này tạo ra một cơ chế dẫn dắt kép: đặc trưng video cung cấp nội dung thị giác, trong khi $F_{\overleftarrow{C}}$ cung cấp định hướng ngữ nghĩa. Kết quả là Forward Decoder có thể sinh ra các câu mô tả có tính mạch lạc cao, bám sát diễn biến thời gian và mang cấu trúc ngôn ngữ tự nhiên hơn.

### 3.3.1. Multimodal features Embedding

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

Cấu trúc này đảm bảo rằng với mỗi GOP, Decoder sẽ nhận được đầy đủ các thông tin từ diện mạo, ngữ nghĩa đến chuyển động. Quy trình này được cài đặt độc lập cho cả Backward Decoder ($E_{BD}$) và Forward Decoder ($E_{FD}$) với các bộ trọng số riêng biệt, giúp tối ưu hóa việc học theo từng chiều giải mã cụ thể.

### 3.3.2. Backward decoder (BD)

### 3.3.3. Forward decoder (FD)

