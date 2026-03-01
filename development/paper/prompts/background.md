Bây giờ bạn hãy giúp tôi đề xuất, thảo luận, chỉnh sửa những chỗ mà tôi viết sai và phát triển các ý tưởng để tôi có thể hoàn thành bản phác thảo cho paper của mình. Bạn hãy giúp tôi trình bày bằng tiếng Việt một cách mạch lạc, chi tiết, súc tích. Đối với một số từ tiếng Anh đặc thù, chuyên ngành thì bạn có thể trình bày bằng tiếng Anh để không làm mất ngữ nghĩa của từ đó.

Dưới đây là bản phác thảo các ý tưởng mà tôi muốn trình bày trong paper của mình:

# Section: 3. Background

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
