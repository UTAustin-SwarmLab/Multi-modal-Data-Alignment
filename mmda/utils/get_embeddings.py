from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    ViTImageProcessor,
    ViTModel,
)


def clap_audio(audio_files : List[np.ndarray], batch_size : int = 32, sample_rate:int = 48_000, max_length_s:int = 10) -> np.ndarray:
    model = AutoModel.from_pretrained("laion/larger_clap_general")
    feature_extractor = AutoFeatureExtractor.from_pretrained("laion/larger_clap_general")
    model = model.cuda()
    audio_features = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(audio_files), batch_size)):
            audio_batch = audio_files[i:i+batch_size]
            inputs = feature_extractor(audio_batch, return_tensors="pt", sampling_rate=sample_rate, padding=True, max_length_s=max_length_s)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            audio_feature_batch = model.get_audio_features(**inputs)
            audio_features.append(audio_feature_batch.detach().cpu().numpy())
    audio_features = np.concatenate(audio_features, axis=0)

    return audio_features

def clap_text(text: List[str], batch_size : int = 32) -> np.ndarray:
    model = AutoModel.from_pretrained("laion/larger_clap_general")
    tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
    model = model.cuda()
    text_features = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            text_batch = text[i:i+batch_size]
            inputs = tokenizer(text_batch, return_tensors="pt", padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_feature_batch = model.get_text_features(**inputs)
            text_features.append(text_feature_batch.detach().cpu().numpy())
    text_features = np.concatenate(text_features, axis=0)

    return text_features

# clip_imgs in batch with gpu
def clip_imgs(img_files : List[str], batch_size : int = 32, noise = False) -> np.ndarray:
    model, _, preprocess= open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(img_files), batch_size)):
            batch = []
            for img_file in img_files[i:i+batch_size]:
                if noise:
                    image = Image.open(img_file)
                    image.filter(ImageFilter.GaussianBlur(5))
                    image = preprocess(image).unsqueeze(0)
                else:
                    image = preprocess(Image.open(img_file)).unsqueeze(0)
                batch.append(image)
            batch = torch.cat(batch, dim=0)
            batch = batch.cuda()
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_embeddings.append(image_features.detach().cpu().numpy())
    img_embeddings = np.concatenate(img_embeddings, axis=0)

    return img_embeddings

# clip text in batch with gpu
def clip_text(text: List[str], batch_size : int = 32) -> np.ndarray:
    model, _, preprocess= open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    model = model.cuda()

    text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i:i+batch_size]
            batch = tokenizer(batch)
            batch = batch.cuda()
            batch = model.encode_text(batch)
            batch /= batch.norm(dim=-1, keepdim=True)
            text_features.append(batch.detach().cpu().numpy())
    text_features = np.concatenate(text_features, axis=0)

    return text_features

def gtr_text(text: List[str]) -> np.ndarray:
    model = SentenceTransformer('sentence-transformers/gtr-t5-large')
    model = model.cuda()
    text_features = model.encode(text)
    return text_features

def vit(img_files : List[str], batch_size : int = 32) -> np.ndarray:
    print("Loading VIT")
    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch32-384')
    model = ViTModel.from_pretrained('google/vit-large-patch32-384')
    
    
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i:i+batch_size]:
                image = Image.open(img_file).convert('RGB')
                images.append(image)
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())

    img_embeddings = np.concatenate(img_embeddings, axis=0)

    return img_embeddings

def dinov2(img_files : List[str], batch_size : int = 32) -> np.ndarray:
    print("Loading DINO")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
    model = AutoModel.from_pretrained('facebook/dinov2-giant')
    model = model.cuda()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i:i+batch_size]:
                image = Image.open(img_file)
                images.append(image)
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())

    img_embeddings = np.concatenate(img_embeddings, axis=0)

    return img_embeddings
