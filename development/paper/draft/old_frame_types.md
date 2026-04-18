In modern video compression standards, such as H.264 and H.265, reducing temporal redundancy between consecutive frames is a key mechanism for optimizing storage. Based on data dependencies and coding techniques, frames within a compressed video sequence are generally categorized into three primary types: I-frame, P-frame, and B-frame [$\sout{CITE}$-Compressed Video Contrastive Learning]().

**I-frame (Intra-coded frame).** An I-frame, also `known as a keyframe`, is encoded independently using `intra-prediction`. It does `not rely on any other frames `for reconstruction. As a result, an I-frame contains complete appearance information, capturing the full spatial context and original objects within a scene.

**P-frame (Predictive frame).** A P-frame uses `uni-directional inter-prediction`. To save storage space, it only records the `visual differences relative to a preceding reference frame` (which can be either an I-frame or a previously decoded P-frame). These differences are efficiently encoded using motion vectors and residual errors.

**B-frame (Bi-predictive frame).** A B-frame uses `bi-directional` inter-prediction. It references information from `both past and future frames` to find the best matching blocks, maximizing storage efficiency. Similar to P-frames, its content is represented by motion vectors and residuals.

---

I-frames, or keyframes, encode the complete spatial context and appearance of a scene entirely through spatial intra-prediction without referencing any other frames.

spatial intra-prediction, with no dependency on other frames. This self-contained nature makes them inherently less compression-efficient than temporally inter-predicted frames.

```
I-frames, or keyframes, rely exclusively on spatial intra-prediction without referencing any other frames. Because they must reconstruct the complete spatial context and appearance of a scene independently, their compression efficiency is relatively limited. To achieve higher compression rates, P-frames and B-frames introduce temporal inter-prediction. Specifically, P-frames use uni-directional inter-prediction to record only the pixel-level differences relative to a preceding reference frame. To further improve storage efficiency, B-frames employ bi-directional inter-prediction by referencing information from both past and future frames. Consequently, rather than blindly storing redundant spatial context, both P-frames and B-frames focus exclusively on encoding the temporal dynamics and motion variations of the scene.
```

---

- I-frames, or keyframes, rely exclusively on spatial intra-prediction without referencing any other frames
- P-frames and B-frames incorporate temporal inter-prediction to record pixel-level differences relative to reference frames.
- While P-frames only refer to preceding frames, B-frames can refer to both past and future frames
- Consequently, if I-frames focus on capturing the complete spatial context and appearance of a scene, both P-frames and B-frames are specifically designed to capture the temporal dynamics and motion variations of the scene.

---

**Group of Pictures (GOP) Structure.** The I-frames, P-frames, and B-frames are not arranged randomly; rather, they follow a periodic, repeating pattern known as the Group of Pictures (GOP) structure. As illustrated in Figure [$\sout{???}$](), each GOP strictly begins with an I-frame and includes all subsequent P-frames and B-frames until the next I-frame appears. 

These complementary frame types are not arranged arbitrarily; instead, they are organized into a periodic structural pattern called a Group of Pictures (GOP). As illustrated in Figure [$\sout{???}$](), each GOP strictly begins with an I-frame and encompasses all subsequent P-frames and B-frames prior to the next I-frame. Typically, a GOP serves as an independently decodable unit within the video bitstream (often referred to as a closed GOP). This means that frames within one specific GOP do not refer to any frames located in adjacent GOPs. Because of this structural independence, we can naturally view a compressed video as a continuous sequence of GOPs rather than a sequence of individual frames, treating each GOP as a distinct structural “unit of information”.

**Discussion.** In traditional video captioning frameworks, models often process densely sampled individual frames. However, this dense sampling strategy is highly computationally expensive and often introduces massive redundant visual information, which can easily overwhelm the video captioning network. By shifting the perspective and utilizing the GOP structure as the fundamental input unit, we can effectively eliminate temporal redundancy while preserving the most critical spatio-temporal dynamics required to generate accurate captions.