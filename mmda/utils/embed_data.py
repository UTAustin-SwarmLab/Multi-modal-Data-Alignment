"""This module contains functions to extract features from images, audio, and text using various models."""

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    ViTImageProcessor,
    ViTModel,
)


def cosplace_img(img_files: list, batch_size: int = 32) -> np.ndarray:
    """Extract image features using CosPlace model specifically trained for the Pittsburgh dataset.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    transforms_list = []
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transforms_list)
    model = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i : i + batch_size]:
                image = Image.open(img_file).convert("RGB")
                image = transform(image).unsqueeze(0)
                images.append(image)
            batch = torch.cat(images, dim=0).cuda()
            outputs = model(batch)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0)


def clap_audio(
    audio_np: list[np.ndarray],
    batch_size: int = 32,
    sample_rate: int = 48_000,
    max_length_s: int = 10,
) -> np.ndarray:
    """Extract audio features using CLAP model.

    Args:
        audio_np: numpy array of audio
        batch_size: batch size
        sample_rate: sample rate of the audio
        max_length_s: maximum length of the audio
    Returns:
        audio features
    """
    model = AutoModel.from_pretrained("laion/larger_clap_general")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "laion/larger_clap_general"
    )
    model = model.cuda()
    audio_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(audio_np), batch_size)):
            audio_batch = audio_np[i : i + batch_size]
            inputs = feature_extractor(
                audio_batch,
                return_tensors="pt",
                sampling_rate=sample_rate,
                padding=True,
                max_length_s=max_length_s,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            audio_feature_batch = model.get_audio_features(**inputs)
            audio_features.append(audio_feature_batch.detach().cpu().numpy())
    return np.concatenate(audio_features, axis=0)


def clap_text(text: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract text features using CLAP model.

    Args:
        text: list of text
        batch_size: batch size
    Returns:
        text features
    """
    model = AutoModel.from_pretrained("laion/larger_clap_general")
    tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
    model = model.cuda()
    text_features = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            text_batch = text[i : i + batch_size]
            inputs = tokenizer(text_batch, return_tensors="pt", padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_feature_batch = model.get_text_features(**inputs)
            text_features.append(text_feature_batch.detach().cpu().numpy())
    return np.concatenate(text_features, axis=0)


# clip_imgs in batch with gpu
def clip_imgs(
    img_files: list[str], batch_size: int = 32, noise: bool = False
) -> np.ndarray:
    """Extract image features using CLIP model.

    Args:
        img_files: list of image files
        batch_size: batch size
        noise: add noise to the image
    Returns:
        image features
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(img_files), batch_size)):
            batch = []
            for img_file in img_files[i : i + batch_size]:
                if noise and isinstance(img_file, str):
                    image = Image.open(img_file)
                    image.filter(ImageFilter.GaussianBlur(5))
                    image = preprocess(image).unsqueeze(0)
                elif not noise and isinstance(img_file, str):
                    image = preprocess(Image.open(img_file)).unsqueeze(0)
                elif not noise and isinstance(img_file, Image.Image):
                    image = preprocess(img_file).unsqueeze(0)
                batch.append(image)
            batch = torch.cat(batch, dim=0)
            batch = batch.cuda()
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_embeddings.append(image_features.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0)


# clip text in batch with gpu
def clip_text(text: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract text features using CLIP model.

    Args:
        text: list of text
        batch_size: batch size
    Returns:
        text features
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )
    model = model.cuda()

    text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size]
            batch = tokenizer(batch)
            batch = batch.cuda()
            batch = model.encode_text(batch)
            batch /= batch.norm(dim=-1, keepdim=True)
            text_features.append(batch.detach().cpu().numpy())
    return np.concatenate(text_features, axis=0)


def gtr_text(text: list[str]) -> np.ndarray:
    """Extract text features using GTR model.

    Args:
        text: list of text
    Returns:
        text features
    """
    model = SentenceTransformer("sentence-transformers/gtr-t5-large")
    model = model.cuda()
    return model.encode(text)


def vit(img_files: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract image features using Vision Transformer model.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    print("Loading VIT")
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch32-384")
    model = ViTModel.from_pretrained("google/vit-large-patch32-384")

    model = model.cuda()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i : i + batch_size]:
                image = Image.open(img_file).convert("RGB")
                images.append(image)
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())

    return np.concatenate(img_embeddings, axis=0)


def dinov2(img_files: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract image features using DINO model.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    print("Loading DINO")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant")
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    image = Image.open(img_file)
                    images.append(image)
            elif isinstance(img_files[i], Image.Image):
                images = img_files[i : i + batch_size]
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())

    return np.concatenate(img_embeddings, axis=0)
