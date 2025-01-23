"""Extract tsfresh features from the Handwriting dataset."""

import pickle
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from PIL import Image
from tsfresh import extract_features

PATH = "/nas/pohan/datasets/Handwriting/"
PATH_SAVE = "/nas/pohan/datasets/Handwriting/embeddings/"


def load_handwriting() -> tuple[np.ndarray, np.ndarray, dict[str, tuple[str, str]]]:
    """Load the Handwriting dataset (https://github.com/amazon-science/aeon).

    Args:
        cfg_dataset: configuration file
    Returns:
        data: data. shape: (num_samples, 3, 152)
        labels: labels. e.g. "1.0"
        num2alphabet: a dict of index to alphabet
        alphabets_hand: list of PIL images
    """
    # train_x.shape: (150, 3, 152), test_x.shape: (850, 3, 152)
    train_x, train_y = load_classification(
        "Handwriting", split="train"
    )  # np.ndarray, list[str]
    test_x, test_y = load_classification("Handwriting", split="test")
    # merge train and test
    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    num2alphabet = {f"{i+1}.0": chr(65 + i) for i in range(26)}
    idx = np.arange(x.shape[0])
    x = x[idx]
    y = y[idx]

    def load_alphabets_img() -> tuple[np.ndarray, np.ndarray]:
        """Load the MNIST dataset.

        Returns:
            data: data
            labels: labels
        """
        # Download latest version
        path = kagglehub.dataset_download(
            "sachinpatel21/az-handwritten-alphabets-in-csv-format"
        )
        df = pd.read_csv(path + "/A_Z Handwritten Data.csv")
        labels = df.iloc[:, 0]
        data = df.iloc[:, 1:]
        return data, labels

    alphabets_x, alphabets_y = load_alphabets_img()
    alphabets_img = {}
    for i in range(26):
        alphabets_img[i + 1] = alphabets_x[alphabets_y == i][:100]

    alphabets_hand = []
    for i in range(x.shape[0]):
        label = int(y[i].split(".")[0])
        random_idx = np.random.choice(alphabets_img[label].shape[0])
        random_df = alphabets_img[label].iloc[random_idx].to_numpy()
        random_df = random_df.reshape(28, 28).astype(np.uint8)
        # save image to png
        path = Path(PATH, f"alphabet_{label}_{random_idx}.png")
        Image.fromarray(random_df, mode="L").save(path)
        alphabets_hand.append(path)
    return (
        x,
        y,
        num2alphabet,
        alphabets_hand,
    )


def tsfresh_features() -> np.ndarray:
    """Extract tsfresh features from the data.

    Returns:
        features: features
    """
    data, labels, num2alphabet, alphabets_hand = load_handwriting()

    path = Path(PATH_SAVE, "Handwriting_tsfresh.csv")

    if path.exists():
        df = pd.read_csv(path)
    else:
        # convert data to a df
        # column_id: id, column_sort: time, values: 3 channels
        df = pd.DataFrame(columns=["id", "time", "channel_1", "channel_2", "channel_3"])
        for idx in range(data.shape[0]):
            for time in range(data.shape[2]):  # 152
                df.loc[idx, "id"] = idx
                df.loc[idx, "time"] = time
                df.loc[idx, "channel_1"] = data[idx, 0, time]
                df.loc[idx, "channel_2"] = data[idx, 1, time]
                df.loc[idx, "channel_3"] = data[idx, 2, time]
        print(df.head())
        print(df.tail())

        df.to_csv(path, index=False)
    ts_features = extract_features(df, column_id="id", column_sort="time")
    ts_features = ts_features.dropna(axis=1)
    print(type(ts_features))
    print(ts_features.shape)
    print(ts_features.head())
    print("ts_features shape:", ts_features.shape)
    with Path(PATH_SAVE, "Handwriting_emb_tsfresh.pkl.pkl").open("wb") as f:
        pickle.dump(ts_features, f)
    print("TSFresh features saved")


if __name__ == "__main__":
    tsfresh_features()
