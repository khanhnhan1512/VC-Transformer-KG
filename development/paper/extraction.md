## 4.2. Multimodal Feature Extraction

- **Intro:** Trình bày rằng 2D appearance feature (hay spatial information) và 3D motion feature (temporal information) đã được ứng dụng rộng rãi trong rất nhiều nghiên cứu trước để rút trích đặc trưng đa phương thức cho video. Giải thích ngắn gọn vai trò của mỗi loại feature này.
    - Do đó trong nghiên cứu này chúng ta vẫn sẽ rút trích 2 loại feature này. Tuy nhiên, để nâng cao khả năng biểu diễn và nắm bắt ngữ nghĩa của video, chúng ta sẽ bổ sung thêm một loại đặc trưng thứ ba: semantic feature.
- **Video Representation:** Nhấn mạnh việc biểu diễn mỗi video thành 1 chuỗi GOP và quá trình rút trích đặc trưng được thực hiện đồng nhất trên mỗi GOP. 
    - Như đã trình bày ở phần trước, mỗi video có thể được biểu diễn dưới dạng một chuỗi các GOP. Giả sử, xét một video $X$, chúng ta có thể biểu diễn video này dưới dạng một chuỗi các GOP như sau: $$X = \left[ \text{GOP}^{(1)}, \dots, \text{GOP}^{(G)} \right]$$ $$\text{GOP}^{(g)} = \left[\text{I}^{(g)}, \text{P/B}^{(g, 1)}, \dots, \text{P/B}^{(g, \text{KeyInt}-1)}\right]$$
    - Trong đó $G$ là số lượng GOP tối đa được lấy từ mỗi video. Each $\text{GOP}^{(g)}$ strictly begins with exactly one I-frame and is followed by at most $\text{KeyInt}-1$ continuous P/B-frames. $\text{KeyInt}$, là viết tắt của Keyframe Interval, is a hyperparameter cho biết khoảng cách tối đa giữa 2 I-frame liên tiếp during the video encoding process.
- **Feature Extraction:** Quá trình trích xuất đặc trưng được thực hiện đồng nhất trên mọi GOP để đảm bảo tính nhất quán của dữ liệu đầu vào cho mô hình video captioning của chúng ta. Cho trước GOP $g$, chúng ta:
    - **Rút trích đặc trưng từ I-frame:** Nhấn mạnh vai trò của I-frame trong mỗi GOP là chứa đựng lượng thông tin thị giác phong phú nhất. Chúng ta rút trích 2 loại đặc trưng appearance feature và semantic feature từ I-frame bằng cách xây dựng 1 pipeline xử lý nhờ sự hợp tác của 2 mô hình. Đầu tiên, ta dùng vision-langle model BLIP-2 để rút trích vector biểu diễn cho I-frame và đồng thời tạo caption cho I-frame. Câu caption này sau đó được đưa sang SRoBERTa để tạo vector embedding cho sentence.
        - Nhấn mạnh rằng BLIP-2 được tạo thành từ 3 module chính là Image Encoder để trích xuất các đặc trưng hình ảnh chất lượng cao từ ảnh đầu vào. Tiếp theo là module Querying Transformer rút trích các đặc trưng hình ảnh phù hợp nhất với ngôn ngữ để cung cấp cho module cuối cùng là LLM để sinh ra câu mô tả cho ảnh.
        - Chúng ta sử dụng $\text{[CLS]}$ token từ đầu ra của Image Encoder để làm appearance token, biểu diễn thông tin cho GOP. Vector sentence embedding làm semantic token.
    - **Đặc trưng motion:** Dùng mô hình MViTv2 để rút trích motion token từ các frame trong GOP, bổ khuyết thông tin tĩnh của I-frame.

---

2D visual features and 3D motion features have been widely used by prior work for multimodal features extraction in video captioning task. Do đó, trong nghiên cứu này, chúng ta cũng sẽ tiếp tục khai thác hai loại đặc trưng này. Tuy nhiên, để nâng cao khả năng biểu diễn và nắm bắt ngữ nghĩa của video, chúng ta sẽ bổ sung thêm một loại đặc trưng thứ ba: semantic feature.

Như đã trình bày ở phần trước, mỗi video có thể được biểu diễn dưới dạng một chuỗi các GOP. Giả sử, xét một video $X$ gồm $N$ GOP, chúng ta có thể biểu diễn video này dưới dạng một chuỗi các GOP như sau:
$$X = [\text{GOP}^{(1)}, \dots, \text{GOP}^{(N)}]$$

Quá trình trích xuất đặc trưng được **thực hiện đồng nhất** trên mọi GOP để đảm bảo tính nhất quán của dữ liệu đầu vào cho mô hình video captioning của chúng ta.

<!-- Đặc trưng diện mạo và Ngữ nghĩa từ I-frame -->

Trong mỗi GOP, I-frame đóng vai trò là anchor point chứa đựng lượng thông tin thị giác phong phú nhất. Thay vì phải trích xuất đặc trưng cho toàn bộ khung hình (có thể gây ra sự dư thừa về dữ liệu và không thực sự đóng góp quá nhiều vào hiệu suất tổng thể), chúng ta chỉ tập trung vào việc trích xuất đặc trưng từ I-frame. Chúng ta sẽ xây dựng một pipeline đơn giản, tận dụng sức mạnh của hai pre-trained models BLIP-2 và SRoBERTa, để thu thập hai loại đặc trưng: visual feature và semantic feature.

**Visual feature** được trích xuất từ $[\text{CLS}]$ token của Image Encoder trong BLIP-2. Mỗi $[\text{CLS}]$ token là một vector cung cấp thông tin thô về các đối tượng và bối cảnh không gian.

Đồng thời, chúng ta cũng tận dụng khả năng sinh ngôn ngữ của BLIP-2 để tạo ra một câu mô tả ngắn (caption) cho I-frame. Để chuyển đổi thông tin văn bản (sentence or sequence of words) này thành một vector trong embedding space (mà mô hình có thể hiểu được), câu mô tả này được đưa qua mô hình SRoBERTa để thu được một vector sentence embedding. Vector này cung cấp một lớp dẫn dắt ngữ nghĩa cấp cao, giúp mô hình nắm bắt được các khái niệm trừu tượng có trong khung hình và chúng ta gọi đây là **semantic feature**.

<!-- Đặc trưng chuyển động không gian - thời gian -->

Để thu thập các tương tác động (spatio-temporal motion dynamics) diễn ra giữa các khung hình, chúng ta sử dụng pre-trained model MViTv2 để rút trích **motion feature**. Mô hình này được áp dụng trên toàn bộ chuỗi khung hình trong mỗi GOP để rút trích ra một vector tóm tắt các thay đổi về mặt cử động, hành động của đối tượng từ các P-frame và B-frame, bổ khuyết cho thông tin tĩnh từ I-frame.

<!-- Biểu diễn toán học của chuỗi đặc trưng -->

Như vậy, đối với mỗi video, sau khi kết thúc quá trình trích xuất multimodal feature cho tất cả $N$ GOP, chúng ta sẽ thu được ba loại đặc trưng là: (1) visual feature $F_V = [v^{(1)},\dots, v^{(N)}]$ where $v^{(n)} \in \mathbb{R}^{d_V}$ and $F_V \in \mathbb{R}^{N \times d_V}$; (2) semantic feature $F_S = [s^{(1)},\dots, s^{(N)}]$ where $s^{(n)} \in \mathbb{R}^{d_S}$ and $F_S \in \mathbb{R}^{N \times d_S}$; và (3) motion feature $F_M = [m^{(1)},\dots, m^{(N)}]$ where $m^{(n)} \in \mathbb{R}^{d_M}$ and $F_M \in \mathbb{R}^{N \times d_M}$. Trong đó, $d_V$, $d_S$, và $d_M$ lần lượt là kích thước (dimension) của không gian đặc trưng, và giá trị cụ thể của chúng phụ thuộc vào kiến trúc của các mô hình pre-trained mà chúng ta sử dụng. Chúng ta sẽ đề cập đến các giá trị này trong phần thực nghiệm.
