import numpy as np
import pandas as pd
import h5py
import av
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def extract_keyframes(video_path: str):
    """
    Return a list of PIL.Image corresponding to keyframes in the video.
    """
    keyframes = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        if frame.key_frame:
            img = frame.to_image()
            keyframes.append(img)
    container.close()
    return keyframes


def extract_visual_embeddings(frames_pil, processor, vision_model, device, batch_size):
    """
    Args:
        frames_pil: list of PIL.Image
        processor: Blip2Processor
        vision_model: Blip2ForConditionalGeneration.vision_model
    """

    all_cls_tokens = []
    all_avg_tokens = []
    vision_model.to(device)
    vision_model.eval()

    with torch.no_grad():
        for i in range(0, len(frames_pil), batch_size):
            # Get batch of images
            batch_imgs = frames_pil[i:i+batch_size]

            # Processor will resize/normalize according to model's default transforms
            inputs = processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            # Inference
            outputs = vision_model(pixel_values=pixel_values)

            # === Extract CLS token ===
            cls_token = outputs.pooler_output  # (B, hidden_dim)
            cls_token = cls_token.cpu().numpy().astype(np.float32)
            all_cls_tokens.append(cls_token)

            # === Extract Average Pooling token ===
            # outputs.last_hidden_state: (B, seq_len, hidden_dim)
            avg_token = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
            avg_token = avg_token.cpu().numpy().astype(np.float32)
            all_avg_tokens.append(avg_token)

    if (len(all_cls_tokens) == 0) or (len(all_avg_tokens) == 0):
        raise ValueError("No features extracted")

    vis_emb_cls = np.vstack(all_cls_tokens)  # shape (N_frames, hidden_dim)
    vis_emb_avg = np.vstack(all_avg_tokens)  # shape (N_frames, hidden_dim)
    return vis_emb_cls, vis_emb_avg


# --- Full pipeline: captions to embeddings ---
def captions_to_embeddings_pipeline(
    image_inputs: List[Image.Image],
    blip_processor: Blip2Processor,
    blip_model: Blip2ForConditionalGeneration,
    embed_model: SentenceTransformer,
    device: str,
    batch_size: int
) -> np.ndarray:
    """
    Process images and generate embeddings using BLIP and SentenceTransformer.

    Args:
        image_inputs: List of PIL.Image objects.
        blip_processor: BLIP processor for image preprocessing.
        blip_model: BLIP model for image captioning.
        embed_model: SentenceTransformer model for text embedding.
        device: Device to run the model on ('cpu' or 'cuda').
        batch_size: Batch size for processing images.
    Returns:
        Numpy array of embeddings with shape (N, embedding_dim).
    """

    # Initialize models
    blip_model.to(device)
    blip_model.eval()
    embed_model.to(device)
    embed_model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(image_inputs), batch_size):
            batch_images: List[Image.Image] = image_inputs[i:i+batch_size]

            # Preprocess images
            inputs = blip_processor(images=batch_images, return_tensors="pt").to(device)

            # Generate captions
            outputs = blip_model.generate(**inputs, max_new_tokens=32, num_beams=4)
            captions = blip_processor.batch_decode(outputs, skip_special_tokens=True)

            # Generate embeddings
            embeddings = embed_model.encode(captions)

            all_embeddings.append(embeddings)

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings extracted")

    return np.vstack(all_embeddings)  # shape (N, embedding_dim)


# --- Load processor and models ---
print("Loading models...")
cache_dir = "/media02/lnthanh01/vmphat/raw_data/cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", cache_dir=cache_dir).to(device)
embed_model = SentenceTransformer(
    "all-roberta-large-v1", device=device, cache_folder=cache_dir)
blip_vision_model = blip_model.vision_model  # encoder for images


# --- Get path to all videos ---
print("Loading video paths...")
video_paths = sorted(Path("./MSRVTT/videos/all/").glob("*.mp4"))
assert len(video_paths) == 10_000
print(f">> Found {len(video_paths)} videos.")


# --- Extract keyframes, image embeddings, embeddings from image captions ---
corpus: str = "MSRVTT"
vid_to_keyframe_count: Dict[str, int] = {}
vis_emb_cls_hdf5 = h5py.File(f"./{corpus}_Blip2ClsKF.hdf5", "w")
vis_emb_avg_hdf5 = h5py.File(f"./{corpus}_Blip2AvgKF.hdf5", "w")
cap_emb_hdf5 = h5py.File(f"./{corpus}_ImgCapBlip2KF.hdf5", "w")

print("Extracting features for each video...")
for video_path in tqdm(video_paths, desc="Processing videos"):
    # === Extract keyframes ===
    keyframes = extract_keyframes(str(video_path))
    vid_to_keyframe_count[video_path.stem] = len(keyframes)

    # === Extract visual embeddings (CLS and AVG) ===
    vis_emb_cls, vis_emb_avg = extract_visual_embeddings(
        frames_pil=keyframes,
        processor=blip_processor,
        vision_model=blip_vision_model,
        device=device,
        batch_size=8
    )
    vis_emb_cls_hdf5.create_dataset(video_path.stem, data=vis_emb_cls)
    vis_emb_avg_hdf5.create_dataset(video_path.stem, data=vis_emb_avg)

    # === Extract caption embeddings from images ===
    caption_embs = captions_to_embeddings_pipeline(
        image_inputs=keyframes,
        blip_processor=blip_processor,
        blip_model=blip_model,
        embed_model=embed_model,
        device=device,
        batch_size=8
    )
    cap_emb_hdf5.create_dataset(video_path.stem, data=caption_embs)

# Close HDF5 files
vis_emb_cls_hdf5.close()
vis_emb_avg_hdf5.close()
cap_emb_hdf5.close()

# Save keyframe counts to CSV
df = pd.DataFrame(
    list(vid_to_keyframe_count.items()),
    columns=["video_id", "keyframe_count"]
)
df.to_csv(f"./{corpus}_keyframe_counts.csv", index=False)

print(">> Done!")
