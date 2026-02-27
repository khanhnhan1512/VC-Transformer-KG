Bây giờ bạn hãy giúp tôi đề xuất, thảo luận, chỉnh sửa những chỗ mà tôi viết sai và phát triển các ý tưởng để tôi có thể hoàn thành bản phác thảo cho paper của mình. Bạn hãy giúp tôi trình bày bằng tiếng Việt một cách mạch lạc, chi tiết, súc tích. Đối với một số từ tiếng Anh đặc thù, chuyên ngành thì bạn có thể trình bày bằng tiếng Anh để không làm mất ngữ nghĩa của từ đó.

Dưới đây là bản phác thảo các ý tưởng mà tôi muốn trình bày trong paper của mình:

- Section: Multimodal features Embedding
- Module này đóng vai trò như một layer chiếu tuyến tính (linear projection layer) để đưa các đặc trưng đã trích xuất từ video về cùng số chiều (dimension) với model $d_{model}$ của chúng ta. Đồng thời cũng bổ sung thêm một số thông tin quan trọng khác như type embedding và positional embedding để giúp mô hình có thể phân biệt được nguồn gốc của từng đặc trưng cũng như nắm bắt được mối quan hệ thời gian giữa chúng. Việc này là cần thiết để đảm bảo rằng các đặc trưng này có thể được tích hợp một cách hiệu quả vào kiến trúc Transformer của chúng ta thông qua cơ chế Cross-Attention.
- Ý tưởng thiết kế module này được lấy cảm hứng từ BERT input representation.
- Đầu tiên, chúng ta sẽ áp dụng một phép biến đổi tuyến tính (linear transformation) riêng biệt cho từng loại đặc trưng để đưa chúng về cùng một không gian vector có kích thước $d_{model}$. Với mỗi loại đặc trưng $F_V$, $F_S$, và $F_M$, chúng ta sẽ thực hiện các phép tính sau:
$$
\begin{align}
F'_V &= F_V * W_V + b_V \\
F'_S &= F_S * W_S + b_S \\
F'_M &= F_M * W_M + b_M
\end{align}
$$
Trong đó, ba ma trận $W_V \in \mathbb{R}^{d_V \times d_{model}}$, $W_S \in \mathbb{R}^{d_S \times d_{model}}$, $W_M \in \mathbb{R}^{d_M \times d_{model}}$ và ba vector bias $b_V, b_S, b_M \in \mathbb{R}^{d_{model}}$ đều là các tham số của mô hình được tối ưu hóa trong quá trình huấn luyện. Sau phép biến đổi này, các đặc trưng đã được chiếu về cùng một không gian vector có kích thước $d_{model}$: $F'_V, F'_S, F'_M \in \mathbb{R}^{N \times d_{model}}$.

- Sau đó, chúng ta sẽ bổ sung thêm **type embeddings** để giúp mô hình phân biệt được nguồn gốc của từng đặc trưng (tương tự như cách mà BERT sử dụng segment embeddings để phân biệt các câu trong một đoạn văn). Đồng thời ta cũng bổ sung thêm **positional embeddings** để giữ lại thông tin về thứ tự của các GOP trong chuỗi video và đảm bảo rằng mô hình có thể nắm bắt được mối quan hệ thời gian giữa các đặc trưng này. Mỗi feature embedding sau đó cũng sẽ được chuẩn hóa bằng một **layer normalization** riêng biệt để đảm bảo rằng chúng có cùng thang đo (scale) trước khi được đưa vào khối Decoder của mô hình.

$$
\begin{align}
E_V &= \text{LayerNorm}_V(F'_V + \text{TypeEmb}_V + \text{PosEmb}) \\
E_S &= \text{LayerNorm}_S(F'_S + \text{TypeEmb}_S + \text{PosEmb}) \\
E_M &= \text{LayerNorm}_M(F'_M + \text{TypeEmb}_M + \text{PosEmb})
\end{align}
$$

Trong đó, $E_V, E_S, E_M \in \mathbb{R}^{N \times d_{model}}$. $\text{PosEmb}$ là positional embeddings, không cần tối ưu hóa trong quá trình huấn luyện mà sẽ được tính toán với công thức như trong mô hình Transformer gốc. Ngược lại, $\text{TypeEmb}$ là một layer được xây dựng để gán một vector embedding riêng biệt cho mỗi loại đặc trưng (visual, semantic, motion) và nó sẽ được học trong quá trình huấn luyện. Mỗi phép biến đổi $\text{LayerNorm}$ cũng sẽ có các tham số được học trong quá trình huấn luyện.

- Cuối cùng chúng ta sẽ nối (concatenation) $E_V, E_S, E_M$ lại với nhau để tạo thành một chuỗi embedding tổng hợp $E \in \mathbb{R}^{3N \times d_{model}}$, trong đó mỗi GOP sẽ được biểu diễn bằng ba vector embedding liên tiếp nhau tương ứng với ba loại đặc trưng khác nhau.

Về mặt toán học, nếu ta biểu diễn:
$$
\begin{align}
E_V &= [e_V^{(1)},\dots, e_V^{(N)}]
\\
E_S &= [e_S^{(1)},\dots, e_S^{(N)}]
\\
E_M &= [e_M^{(1)},\dots, e_M^{(N)}]
\end{align}
$$

thì:
$$
\begin{align}
E &= \text{Concat}(E_V, E_S, E_M) \\
  &= [e_{V}^{(1)}, e_{S}^{(1)}, e_{M}^{(1)}, \dots, e_{V}^{(N)}, e_{S}^{(N)}, e_{M}^{(N)}]
\end{align}
$$

- Toàn bộ quá trình mà ta đã trình bày phía trên sẽ được áp dụng hoàn toàn tương tự nhau cho cả Backward Decoder và Forward Decoder. Ta sẽ ký hiệu $E_{BD}$ và $E_{FD}$ lần lượt là multimodal feature embedding được sử dụng trong Backward Decoder và Forward Decoder.
