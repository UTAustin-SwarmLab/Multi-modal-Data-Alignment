"""Utility functions for executing llava model."""

import multiprocessing as mp
from io import BytesIO
from multiprocessing import Pool

import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


def load_image(image_file: str) -> Image.Image:
    """Load image from file or url.

    Args:
        image_file: image file path or url
    Returns:
        image: PIL image
    """
    if image_file.startswith(("http", "https")):
        response = requests.get(image_file)  # noqa: S113
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def query_llava(input_tuple_data: tuple[DictConfig, list[str], list[str]]) -> list[str]:
    """Get text descriptions from llava model.

    Args:
        input_tuple_data: input tuple data: (cfg_llava, img_paths, prompt_list)

    Returns:
        text_descriptions: list of text descriptions
    """
    cfg_llava, img_paths, prompt_list = input_tuple_data
    disable_torch_init()
    model_name = get_model_name_from_path(cfg_llava.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        cfg_llava.model_path,
        cfg_llava.model_base,
        model_name,
        cfg_llava.load_8bit,
        cfg_llava.load_4bit,
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if cfg_llava.conv_mode is not None and conv_mode != cfg_llava.conv_mode:
        print(
            f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is\
                  {cfg_llava.conv_mode}, using {cfg_llava.conv_mode}"
        )
    else:
        cfg_llava.conv_mode = conv_mode

    text_descriptions = []

    ### adapt to newest llava 1.2.1
    count = -1
    for image_file in tqdm(img_paths):
        count += 1
        conv = conv_templates[cfg_llava.conv_mode].copy()
        llava_imput = prompt_list[count]
        llava_imput = DEFAULT_IMAGE_TOKEN + "\n" + llava_imput
        conv.append_message(conv.roles[0], llava_imput)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = [load_image(image_file)]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=cfg_llava.temperature > 0,
                temperature=cfg_llava.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        text_descriptions.append([image_file, outputs])
    return text_descriptions


def llava_caption(cfg: DictConfig, img_paths: list[str]) -> list[str]:
    """Query llava model to get text descriptions of images.

    Args:
        cfg: config
        img_paths: image file paths
    Returns:
        img_captions: list of text descriptions
    """
    mp.set_start_method("spawn")
    if cfg.dataset == "pitts":
        prompt_list = (
            [
                "Describe the static objects in the image like buildings. \
                Do not include dynamic objects such as people, cars, or animals."
            ]
            * len(img_paths)
        )
    elif cfg.dataset == "sop":
        prompt_list = ["Describe the object in the image."] * len(img_paths)
    elif cfg.dataset == "KITTI":
        prompt_list = [
            "Describe the static objects and the numbers of objects in the image within 20 words."
        ] * len(img_paths)
    # TODO: Add more datasets
    else:
        msg = f"Dataset {cfg.dataset} not supported."
        raise ValueError(msg)
    try:
        num_processes = cfg.llava.num_processes
        p = Pool(processes=num_processes)
        print("num_processes:", num_processes)
        data = p.map(
            query_llava,
            [
                (
                    cfg.llava,
                    img_paths[
                        int(i * len(img_paths) / num_processes) : int(
                            (i + 1) * len(img_paths) / num_processes
                        )
                    ],
                    prompt_list[
                        int(i * len(prompt_list) / num_processes) : int(
                            (i + 1) * len(prompt_list) / num_processes
                        )
                    ],
                )
                for i in range(num_processes)
            ],
        )
        img_captions = data.copy()
    except RuntimeError:
        print("-----------------------------------------------")
        print("RuntimeError.")
        img_captions = query_llava((cfg.llava, img_paths, prompt_list))
    return img_captions


def llava_img_text_align(
    cfg: DictConfig, img_paths: list[str], text_descriptions: list[str]
) -> list[str]:
    """Query llava model to see if the texts are aligned with the provided images.

    Args:
        cfg: config
        img_paths: image file paths
        text_descriptions: text descriptions (captions) of images
    Returns:
        llava_yes_no_answer: list of text descriptions
    """
    assert len(img_paths) == len(text_descriptions), "Image and text length mismatch."
    prompt_list = []
    for i in range(len(text_descriptions)):
        if cfg.dataset == "cosmos":
            prompt = f'Does the following text describe the given image? GPE means any location. Answer in yes/no. \
                    "{text_descriptions[i][1]}"'
        else:
            prompt = f'Does the following text describe the given image? Answer in yes/no. "{text_descriptions[i]}"'
        prompt_list.append(prompt)

    mp.set_start_method("spawn", force=True)
    try:
        num_processes = cfg.llava.num_processes
        p = Pool(processes=num_processes)
        print("num_processes:", num_processes)
        data = p.map(
            query_llava,
            [
                (
                    cfg.llava,
                    img_paths[
                        int(i * len(img_paths) / num_processes) : int(
                            (i + 1) * len(img_paths) / num_processes
                        )
                    ],
                    prompt_list[
                        int(i * len(prompt_list) / num_processes) : int(
                            (i + 1) * len(prompt_list) / num_processes
                        )
                    ],
                )
                for i in range(num_processes)
            ],
        )
        llava_yes_no_answer = data.copy()
    except RuntimeError:
        print("-----------------------------------------------")
        llava_yes_no_answer = query_llava((cfg.llava, img_paths, prompt_list))
    return llava_yes_no_answer
