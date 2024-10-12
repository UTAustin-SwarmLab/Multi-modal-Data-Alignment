# noqa
import torch
from imagebind.data import (
    load_and_transform_audio_data,
    load_and_transform_text,
    load_and_transform_vision_data,
)
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class ImageBindInference:
    def __init__(self, device:int = 0):
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    def inference_audio(self, image_paths, audio_paths):
        inputs = {
            ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
            return embeddings[ModalityType.AUDIO]

    def inference_image(self, image_paths):
        inputs = {
            ModalityType.VISION: load_and_transform_vision_data(image_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
            return embeddings[ModalityType.VISION]
        
    def inference_text(self, text_list):
        inputs = {
            ModalityType.TEXT: load_and_transform_vision_data(text_list, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
            return embeddings[ModalityType.TEXT]