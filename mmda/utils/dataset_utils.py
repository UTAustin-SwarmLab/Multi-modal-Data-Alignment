"""Utility functions for loading and processing datasets."""

import ast
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import datasets
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import hydra
from mmda.liploc.dataloaders.KittiBothDataset import KITTIBothDataset
from mmda.utils.liploc_model import CFG, load_eval_filenames
from mmda.utils.video_audio_utils import process_video_ids


def load_msrvtt(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the MSR-VTT dataset (https://github.com/microsoft/MSR-VTT-Tools).

    Dataset link: https://www.kaggle.com/datasets/vishnutheepb/msrvtt
    Args:
        cfg_dataset: configuration file

    Returns:
        sen_ids: list of sentence_ids
        captions: list of captions. len 59800
        video_info_sen_order: list of video information (in the order of sen_ids)
        video_dict: a dict of video information (video_id to video_info). len: 2990
    """
    # load MSR-VTT json file
    with Path(cfg_dataset.paths.dataset_path, "test_videodatainfo.json").open() as f:
        json_data = json.load(f)  # each video has 20 sentences
        # "id": int, "video_id": str, "category": int, "url": str, "start time": float, "end time": float, "split": str
        # time of the YT video (here the video is already cut into clips).
        videos = json_data["videos"]
        # "sen_id": int, "video_id": str, "caption": str
        sentences = json_data["sentences"]

    # if no video_dict.pkl, extract audio from videos
    if not Path(cfg_dataset.paths.dataset_path, "video_dict.pkl").exists():
        list_video_ids = [video_json["video_id"] for video_json in videos]
        video_dict = {}
        for video_json in videos:
            video_id = video_json["video_id"]
            start_time = video_json["start time"]
            end_time = video_json["end time"]
            split = video_json["split"]
            category = video_json["category"]
            url = video_json["url"]
            video_dict[video_id] = {
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "split": split,
                "category": category,
                "url": url,
            }
        num_processes = 64
        p = Pool(processes=num_processes)
        print("num_processes:", num_processes)
        _ = p.map(
            process_video_ids,
            [
                (
                    cfg_dataset,
                    list_video_ids[
                        int(i * len(list_video_ids) / num_processes) : int(
                            (i + 1) * len(list_video_ids) / num_processes
                        )
                    ],
                )
                for i in range(num_processes)
            ],
        )
        # save the video_dict
        with Path(cfg_dataset.paths.dataset_path, "video_dict.pkl").open("wb") as f:
            pickle.dump(video_dict, f)
    else:
        with Path(cfg_dataset.paths.dataset_path, "video_dict.pkl").open("rb") as f:
            video_dict = joblib.load(f)

    captions, sen_ids = [], []
    video_info_sen_order = []
    for sentence_json in sentences:
        video_id = sentence_json["video_id"]
        video_info_sen_order.append(video_dict[video_id])
        sen_ids.append(sentence_json["sen_id"])
        captions.append(sentence_json["caption"])

    return sen_ids, captions, video_info_sen_order, video_dict


def load_kitti(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the KITTI dataset (https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

    Args:
        cfg_dataset: configuration file

    Returns:
        img_paths: list of image absolute paths
        lidar_paths: list of LIDAR absolute paths
        text_descriptions: list of text descriptions
    """
    filenames = load_eval_filenames()
    transforms = []
    dataset = KITTIBothDataset(
        transforms=transforms,
        CFG=CFG,
        filenames=filenames,
    )
    img_paths, lidar_paths = dataset.get_image_lidar_paths()
    # load pickle file
    with Path(
        cfg_dataset.paths.dataset_path + "KITTI_llava-v1.5-13b_captions.pkl"
    ).open("rb") as f:
        path_text_descriptions = joblib.load(f)
    # merge outputs of multiprocesses
    merged_path_text_descriptions = []
    if len(path_text_descriptions) <= 10:  # noqa: PLR2004
        for multi_thread_results in path_text_descriptions:
            merged_path_text_descriptions += multi_thread_results
    else:
        merged_path_text_descriptions = path_text_descriptions
    text_descriptions = [x[1] for x in merged_path_text_descriptions]
    return img_paths, lidar_paths, text_descriptions


def load_leafy_spurge(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the mpg-ranch/leafy_spurge dataset (https://huggingface.co/datasets/mpg-ranch/leafy_spurge).

    Args:
        cfg_dataset: configuration file

    Returns:
        images: list of image (in PIL format) (train + test)
        labels: list of binary labels (train + test)
        idx2label: a dict of index to label
    """
    # We only take the crop set of 39x39 pixel images
    # load the dataset from huggingface
    if Path(cfg_dataset.paths.dataset_path + "train").exists():
        trains_ds = datasets.load_from_disk(cfg_dataset.paths.dataset_path + "train")
        test_ds = datasets.load_from_disk(cfg_dataset.paths.dataset_path + "test")
    else:
        trains_ds = datasets.load_dataset(
            "mpg-ranch/leafy_spurge",
            # "crop",
            "context",
            split="train",
        )  # 800
        test_ds = datasets.load_dataset(
            "mpg-ranch/leafy_spurge",
            # "crop",
            "context",
            split="test",
        )  # 100

    idx2label = {0: "not leafy spurge", 1: "leafy spurge"}
    return (
        trains_ds["image"] + test_ds["image"],
        trains_ds["label"] + test_ds["label"],
        idx2label,
    )


def load_flickr(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the Flickr dataset (https://huggingface.co/datasets/nlphuji/flickr30k).

    Args:
        cfg_dataset: configuration file

    Returns:
        img_paths: list of image absolute paths
        text_descriptions: list of text descriptions
        splits: list of splits [train, test, val] (str)
        obj_ids: list of object ids (str)
    """
    # load Flickr train json filee. columns: [raw, sentids, split, filename, img_id]
    flickr = pd.read_csv(cfg_dataset.paths.dataset_path + "flickr_annotations_30k.csv")
    img_paths, text_descriptions, splits, img_ids = [], [], [], []
    for _, row in flickr.iterrows():
        texts = row["raw"].replace("[", "").replace("]", "").split('", "')
        assert len(texts) == 5, f"Not 5 captions: {len(texts)}"  # noqa: PLR2004
        for text in texts:
            text_descriptions.append(text.replace('"', ""))
            img_paths.append(
                str(
                    Path(cfg_dataset.paths.dataset_path)
                    / "flickr30k-images"
                    / row["filename"]
                )
            )
            splits.append(row["split"])
            img_ids.append(row["img_id"])
    # ['test', 'train', 'val'] = [5000, 145000, 5070]
    return img_paths, text_descriptions, np.array(splits), np.array(img_ids)


def load_pitts(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the PITTS dataset.

    Args:
        cfg_dataset: configuration file

    Returns:
        img_paths: list of image absolute paths
        text_descriptions: list of text descriptions
        obj_ids: list of object ids
    """
    # load PITTS train json files
    # train set
    with Path(
        cfg_dataset.paths.dataset_path + "pitts_llava-v1.5-13b_captions.pkl"
    ).open("rb") as f:
        path_text_descriptions_threads = joblib.load(f)
    path_text_descriptions = []
    for i in range(len(path_text_descriptions_threads)):
        path_text_descriptions.extend(path_text_descriptions_threads[i])
    img_paths = [x[0] for x in path_text_descriptions]
    text_descriptions = [x[1] for x in path_text_descriptions]
    obj_ids = [img_path.split("/")[-1].split("_")[0] for img_path in img_paths]
    return img_paths, text_descriptions, obj_ids


def load_cosmos(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Load the COSMOS dataset (https://github.com/shivangi-aneja/COSMOS?tab=readme-ov-file).

    Args:
        cfg_dataset: configuration file

    Returns:
        img_paths: list of image absolute paths
        text_descriptions: list of text descriptions
        inconsistency: list of labels (True: inconsistent, False: consistent)
        article_urls: list of article urls
    """
    img_paths = []
    text_descriptions = []
    inconsistency = []
    article_urls = []

    # load COSMOS val data json files
    with Path(cfg_dataset.paths.dataset_path + "val_data.json").open("r") as f:
        for line in f:
            data = ast.literal_eval(line)
            # caption 1: 41,006
            # since the first caption is the original caption from the website, thus inconsistency is always False
            # and since we do not have labels for the val/train data, we do not consider the other captions
            img_paths.append(
                str(Path(cfg_dataset.paths.dataset_path + data["img_local_path"]))
            )
            text_descriptions.append(data["articles"][0]["caption_modified"])
            inconsistency.append(0)
            article_urls.append(data["articles"][0]["article_url"])

    # load COSMOS test data json files
    with Path(cfg_dataset.paths.dataset_path + "test_data.json").open("r") as f:
        for line in f:
            data = ast.literal_eval(line)
            # caption 1: 1700
            # the first caption is the original caption from the website, thus inconsistency is always False
            img_paths.append(
                str(Path(cfg_dataset.paths.dataset_path + data["img_local_path"]))
            )
            text_descriptions.append(data["caption1_modified"])
            inconsistency.append(0)
            article_urls.append(data["article_url"])
            # caption 2: 1700
            # the second caption is the google-searched caption, thus inconsistency can be True
            img_paths.append(
                str(Path(cfg_dataset.paths.dataset_path + data["img_local_path"]))
            )
            text_descriptions.append(data["caption2_modified"])
            inconsistency.append(
                data["context_label"]
            )  # (1=Out-of-Context, 0=Not-Out-of-Context )
            article_urls.append(data["article_url"])
    inconsistency = np.array(inconsistency, dtype=bool)
    return img_paths, text_descriptions, inconsistency, article_urls


def load_tiil(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], np.ndarray, list[str | None]]:
    """Load the TIIL dataset (https://github.com/Mingzhen-Huang/D-TIIL).

    Args:
        cfg_dataset: configuration file

    Returns:
        img_paths: list of image absolute paths
        text_descriptions: list of text descriptions
        inconsistent_labels: list of labels (True: inconsistent, False: consistent)
        original words:
    """
    # load TIIL json files
    with Path(cfg_dataset.paths.dataset_path + "consistent.json").open("rb") as f:
        consistent_json = json.load(f)
    with Path(cfg_dataset.paths.dataset_path + "inconsistent.json").open("rb") as f:
        inconsistent_json = json.load(f)
    dataset_size = len(consistent_json["images"]) + len(inconsistent_json["images"])

    # load TIIL images path
    img_paths = [None] * dataset_size
    text_descriptions = [None] * dataset_size
    inconsistent_labels = [None] * dataset_size
    original_words = [None] * dataset_size

    # load the data from the json files
    for idx, (img_dict, annot_dict) in enumerate(
        zip(consistent_json["images"], consistent_json["annotations"], strict=False)
    ):
        img_paths[idx] = Path(cfg_dataset.paths.dataset_path + img_dict["file_name"])
        text_descriptions[idx] = annot_dict["caption"]
        inconsistent_labels[idx] = False
        assert (
            img_dict["id"] == annot_dict["image_id"]
        ), f"ID mismatch: {img_dict['id']} != {annot_dict['image_id']}"
        assert img_dict["id"] == idx + 1, f"ID mismatch: {img_dict['id']} != {idx}"

    for idx, (img_dict, annot_dict) in enumerate(
        zip(inconsistent_json["images"], inconsistent_json["annotations"], strict=False)
    ):
        idx_shifted = idx + len(consistent_json["images"])
        img_paths[idx_shifted] = Path(
            cfg_dataset.paths.dataset_path, img_dict["file_name"]
        )
        text_descriptions[idx_shifted] = annot_dict["caption"]
        inconsistent_labels[idx_shifted] = True
        original_words[idx_shifted] = annot_dict["ori_word"]
        assert (
            img_dict["id"] == annot_dict["image_id"]
        ), f"ID mismatch: {img_dict['id']} != {annot_dict['image_id']}"
        assert img_dict["id"] == idx_shifted + 1 - len(
            consistent_json["images"]
        ), f"ID mismatch: {img_dict['id']} != {idx_shifted}"
    return (
        img_paths,
        text_descriptions,
        np.array(inconsistent_labels, dtype=bool),
        original_words,
    )


def load_imagenet(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[int], list[int], dict[int, str]]:
    """Load the ImageNet dataset (https://github.com/google-research/imagenet-mistakes?tab=readme-ov-file).

    Args:
        cfg_dataset: configuration file
    Returns:
        img_path: list of image absolute paths
        mturks_idx: MTurk verified classe indices (int)
        orig_idx: ground truth class indices (int)
        clsidx_to_labels: a dict of class idx to str.
    """
    # load json file
    with Path(cfg_dataset.paths.dataset_path, "imagenet_mturk.json").open() as f:
        mturks = json.load(f)  # 5440
    with Path(
        cfg_dataset.paths.dataset_path, "imagenet_val_set_index_to_filepath.json"
    ).open() as f:
        idx2path = json.load(
            f
        )  # ["val/n01440764/ILSVRC2012_val_00000293.JPEG", ...] # 50000
    img_path = [None] * len(idx2path)
    for idx, path in enumerate(idx2path):
        img_path[idx] = Path(cfg_dataset.paths.dataset_path, path)
    orig_idx = np.load(
        Path(cfg_dataset.paths.dataset_path, "imagenet_val_set_original_labels.npy")
    )

    mturks_idx = orig_idx.copy()
    # correct the labels to the MTurk labels
    for mturk in mturks:
        orig_label = mturk["given_original_label"]
        guessed_label = mturk["our_guessed_label"]
        img_name = mturk["url"].replace("https://labelerrors.com//static/imagenet/", "")
        img_index = idx2path.index(img_name)
        mturks_idx[img_index] = int(guessed_label)
        assert (
            orig_idx[img_index] == orig_label
        ), f"Mismatch at {img_index}: {orig_idx[img_index]} != {orig_label}"
    assert np.sum(orig_idx != mturks_idx) == len(
        mturks
    ), f"Relabel num mismatch: {np.sum(orig_idx != mturks_idx)}"

    # convert the labels to string. Obtained from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    with Path(
        cfg_dataset.paths.dataset_path, "ImageNet_clsidx_to_labels.txt"
    ).open() as f:
        clsidx_to_labels_txt = f.readlines()
    clsidx_to_labels = {}
    for ln in clsidx_to_labels_txt:  # example: {0: 'tench, Tinca tinca',
        line = ln.replace("{", "").replace("}", "").rstrip(",\n")
        idx, label = line.split(":")
        idx, label = int(idx.strip()), label.strip()
        label = label.replace("'", "")
        clsidx_to_labels[idx] = label
    return img_path, mturks_idx, orig_idx, clsidx_to_labels


def load_musiccaps(cfg_dataset: DictConfig) -> pd.DataFrame:
    """Load the Google MusicCaps dataset (https://huggingface.co/datasets/google/MusicCaps).

    Args:
        cfg_dataset: configuration file
    Returns:
        dataframe: A dataframe containing the following columns:
            youtube id: list of youtube ids
            audio paths: list of audio absolute paths
            caption: list of text descriptions
            aspect_list: list of aspects (str)
            audioset_positive_labels (str)
            start_time: list of start time (int, sec)
            end_time: list of end time (int, sec)
    """
    parent_dir = Path(cfg_dataset.paths.dataset_path).parent.absolute()
    df_path = Path(parent_dir, "MusicCaps_parsed.csv")
    if Path.exists(df_path):
        return pd.read_csv(df_path)

    # if the parsed file does not exist, load the dataset and parse it
    dataset = datasets.load_dataset("google/MusicCaps", split="train")
    dataframe = pd.DataFrame(
        columns=[
            "ytid",
            "audio_path",
            "caption",
            "aspect_list",
            "audioset_positive_labels",
            "start_time",
            "end_time",
        ]
    )
    rows_list = []
    for data in dataset:
        audio_path = Path(cfg_dataset.paths.dataset_path, f"{data['ytid']}.wav")
        ### check if the audio file exists
        if not Path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist. Skipping.")
            continue
        row_in_dict = {
            "ytid": data["ytid"],
            "audio_path": audio_path,
            "caption": data["caption"],
            "aspect_list": data["aspect_list"],
            "audioset_positive_labels": data["audioset_positive_labels"][0],
            "start_s": data["start_s"],
            "end_s": data["end_s"],
        }
        rows_list.append(row_in_dict)

    dataframe = pd.DataFrame(rows_list)
    dataframe.to_csv(df_path, index=False)
    return dataframe


def load_sop(
    cfg_dataset: DictConfig,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load the Stanford Online Products dataset.

    Args:
        cfg_dataset: configuration file
    Returns:
        image paths: list of image absolute paths
        text descriptions: list of text descriptions
        classes: list of classes (str)
        object ids: list of object ids (str)
    """
    # load SOP images path
    with Path(cfg_dataset.paths.dataset_path + "text_descriptions_SOP.pkl").open(
        "rb"
    ) as f:
        # '/store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG'
        # "The image features a close-up view of a bicycle's suspension system,
        # specifically focusing on the front fork and the shock absorber. "
        path_text_descriptions = joblib.load(f)
    for path_text in path_text_descriptions:
        path_text[0] = path_text[0].replace("/store/", "/nas/")
        path_text[1] = path_text[1].replace(" ", "")
    img_paths = [x[0] for x in path_text_descriptions]
    text_descriptions = [x[1] for x in path_text_descriptions]
    ### img_path example: /store/omama/datasets/Stanford_Online_Products/bicycle_final/251952414262_2.JPG
    classes = [img_path.split("/")[-2].split("_")[0] for img_path in img_paths]
    obj_ids = [img_path.split("/")[-1].split("_")[0] for img_path in img_paths]
    return img_paths, text_descriptions, classes, obj_ids


def get_train_test_split_index(
    train_test_ration: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the index of the training and validation set.

    Args:
        train_test_ration: ratio of training set
        n: number of samples
    Returns:
        index of the training and validation set
    """
    arange = np.arange(n)
    np.random.shuffle(arange)
    train_idx = arange[: int(n * train_test_ration)]
    val_idx = arange[int(n * train_test_ration) :]
    return train_idx, val_idx


def train_test_split(
    data: np.ndarray, train_idx: list[int], val_idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Split the data into training and validation set.

    Args:
        data: data
        train_idx: index of the training set
        val_idx: index of the validation set
    Return:
        training and validation set
    """
    if data is not np.ndarray:
        data = np.array(data)
    return data[train_idx], data[val_idx]


def filter_str_label(ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    """Filter the data based on the provided ground truth.

    Args:
        ground_truth: ground truth. shape: (N, )

    Return:
        a dict of index filter. keys: unique ground truth, values: indices of the data
    """
    ground_truth = ground_truth.astype(str)
    unique_classes = np.unique(ground_truth)
    filter_idx = {}
    for cls in unique_classes:
        filter_idx[cls] = np.where(ground_truth == cls)[0]
    return filter_idx


def shuffle_data_by_indices(
    data: np.ndarray, filter_idx: dict[str, np.ndarray]
) -> np.ndarray:
    """Shuffle the data by classes.

    Args:
        data: data
        filter_idx: a dict of index filter. keys: unique ground truth, values: indices of the data
        seed: random seed
    Return:
        shuffled data
    """
    for _key, val in filter_idx.items():
        c = data[val]
        np.random.shuffle(c)
        data[val] = c
    return data


def shuffle_percentage_of_data(data: np.ndarray, x: float) -> np.ndarray:
    """Shuffle a percentage of the data.

    Args:
        data: data
        x: percentage of data to shuffle
    Return:
        shuffled_data
    """
    # Calculate the number of elements to shuffle
    num_to_shuffle = int(data.shape[0] * x)
    # Get the indices of the elements to shuffle
    shuffled_indices = np.random.choice(data.shape[0], num_to_shuffle, replace=False)
    # Create a copy of the data to avoid modifying the original array
    shuffled_data = data.copy()
    # Apply the shuffle to the selected indices
    shuffled_data[shuffled_indices, :] = np.random.permutation(
        shuffled_data[shuffled_indices, :]
    )
    return shuffled_data


def filter_outliers(
    scores: np.ndarray, threshold: float, right_tail: bool = False
) -> np.ndarray:
    """Return the indices of the outliers (either the right or left tails) filtered from the given the scores.

    Args:
        scores: scores of data. shape: (N, ).
        threshold: threshold of similarity score for outliers.
        right_tail: right tail or left tail of the data.

    Return:
        indices of the outliers.
    """
    if right_tail:
        index = np.where(scores > threshold)[0]
        # sort the index from high to low by the similarity score
        index = index[np.argsort(scores[index])[::-1]]  # descending order
    else:
        index = np.where(scores < threshold)[0]
        # sort the index from low to high by the similarity score
        index = index[np.argsort(scores[index])]  # ascending order
    return index


def shuffle_by_level(  # noqa: PLR0912, C901, ANN201
    cfg: DictConfig,
    shuffle_level: str,
    traindata2unalign: np.ndarray,
    valdata2unalign: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
):
    """Shuffle the data by dataset, class, or object level.

    Args:
        cfg: configuration file
        dataset: dataset name
        shuffle_level: shuffle level. It can only be "dataset", "class", or "object".
        traindata2unalign: unaligned data
        valdata2unalign: unaligned data
        train_idx: training indices
        val_idx: validation indices
    Returns:
        traindata2unalign: shuffled training data for modal 2
        valdata2unalign: shuffled validation data for modal 2
    """
    assert shuffle_level in [
        "dataset",
        "class",
        "object",
    ], f"shuffle_level {shuffle_level} not supported."
    # all datasets can shuffle by dataset level
    if shuffle_level == "dataset":
        np.random.shuffle(traindata2unalign)
        np.random.shuffle(valdata2unalign)
        return traindata2unalign, valdata2unalign
    dataset = cfg.dataset
    cfg_dataset = cfg[dataset]
    # shuffle by class or object level
    if dataset == "sop":
        _, _, classes, obj_ids = load_sop(cfg_dataset)
        if shuffle_level == "class":
            train_gts, val_gts = train_test_split(classes, train_idx, val_idx)
        elif shuffle_level == "object":
            train_gts, val_gts = train_test_split(obj_ids, train_idx, val_idx)
        else:
            msg = f"Dataset {dataset} does not have {shuffle_level} information."
            raise ValueError(msg)
    elif dataset == "musiccaps":
        dataframe = load_musiccaps(cfg_dataset)
        if shuffle_level == "class":
            gts = dataframe["audioset_positive_labels"].tolist()
            train_gts, val_gts = train_test_split(gts, train_idx, val_idx)
        else:
            msg = f"Dataset {dataset} does not have {shuffle_level} information."
            raise ValueError(msg)
    elif dataset == "imagenet":
        _, _, orig_idx, clsidx_to_labels = load_imagenet(cfg_dataset)
        orig_labels = [clsidx_to_labels[i] for i in orig_idx]
        if shuffle_level == "class":
            train_gts, val_gts = train_test_split(orig_labels, train_idx, val_idx)
        else:
            msg = f"Dataset {dataset} does not have {shuffle_level} information."
            raise ValueError(msg)
    elif dataset == "pitts":
        _, _, obj_ids = load_pitts(cfg_dataset)
        if shuffle_level == "object":
            train_gts, val_gts = train_test_split(obj_ids, train_idx, val_idx)
        else:
            msg = f"Dataset {dataset} does not have {shuffle_level} information."
            raise ValueError(msg)
    # TODO: add more datasets
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)
    val_dict_filter = filter_str_label(val_gts)
    valdata2unalign = shuffle_data_by_indices(valdata2unalign, val_dict_filter)
    train_dict_filter = filter_str_label(train_gts)
    traindata2unalign = shuffle_data_by_indices(traindata2unalign, train_dict_filter)
    return traindata2unalign, valdata2unalign


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: D103
    load_msrvtt(cfg.MSRVTT)


if __name__ == "__main__":
    main()
