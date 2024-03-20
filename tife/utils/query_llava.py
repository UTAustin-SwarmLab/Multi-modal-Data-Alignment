import multiprocessing as mp
from io import BytesIO
from multiprocessing import Pool
from typing import List, Tuple

import requests
import torch
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
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
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def get_text_descriptions(input_tuple_data: Tuple[DictConfig, List[str], List[str]])->List[str]:
    cfg, img_paths, texts = input_tuple_data
    disable_torch_init()
    model_name = get_model_name_from_path(cfg.llava.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        cfg.llava.model_path,
        cfg.llava.model_base,
        model_name,
        cfg.llava.load_8bit,
        cfg.llava.load_4bit,
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
    if cfg.llava.conv_mode is not None and conv_mode != cfg.llava.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, cfg.llava.conv_mode, cfg.llava.conv_mode
            )
        )
    else:
        cfg.llava.conv_mode = conv_mode

    text_descriptions = []

    ### adapt to newest llava 1.2.1
    count = -1
    for image_file in tqdm(img_paths):
        count += 1
        conv = conv_templates[cfg.llava.conv_mode].copy()
        inp = f"Does the following text describe the given image? Answer in yes/no. \"{texts[count]}\""
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = [load_image(image_file)]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
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
                do_sample=True if cfg.llava.temperature > 0 else False,
                temperature=cfg.llava.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        text_descriptions.append([image_file, outputs])
        # print(count, image_file, outputs)

    return text_descriptions


def query_llava(cfg, img_paths, text_descriptions):
    mp.set_start_method("spawn")
    try:
        num_processes = cfg.llava.num_processes
        p = Pool(processes=num_processes)
        print("num_processes:", num_processes)

        data = p.map(
            get_text_descriptions,
            [
                (
                    cfg,
                    img_paths[
                        int(i * len(img_paths) / num_processes) : int(
                            (i + 1) * len(img_paths) / num_processes
                        )
                    ],
                    text_descriptions[
                        int(i * len(text_descriptions) / num_processes) : int(
                            (i + 1) * len(text_descriptions) / num_processes
                        )
                    ],
                )
                for i in range(num_processes)
            ],
        )
        return_data = data.copy()
    except RuntimeError:
        print("-----------------------------------------------")
        print("RuntimeError.")
        return_data = get_text_descriptions((cfg, img_paths, text_descriptions))

    return return_data
