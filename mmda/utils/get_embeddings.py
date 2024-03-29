import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    ViTImageProcessor,
    ViTModel,
)


# clip_imgs in batch with gpu
def clip_imgs(img_files : list, batch_size : int = 32, noise = False):
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2B-39B-b160k')
    # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    model, _, preprocess= open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    # model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    # tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
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
def clip_text(text, batch_size : int = 32):
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

def gtr_text(text, batch_size : int = 32):
    model = SentenceTransformer('sentence-transformers/gtr-t5-large')
    model = model.cuda()
    text_features = model.encode(text)
    return text_features

def vit(img_files : list, batch_size : int = 32):
    # pass
    print("Loading VIT")
    # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    # model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
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

def dinov2(img_files : list, batch_size : int = 32):
    # pass
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


def get_embedding_function_mapping():
    return {
        "clip": clip_imgs,
        "dinov2": dinov2,
        "vit": vit,
    }

