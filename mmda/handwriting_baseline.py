"""This script is for the handwriting baseline."""

import numpy as np
from aeon.classification.deep_learning import InceptionTimeClassifier
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

import hydra
from mmda.utils.dataset_utils import load_handwriting


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Train the handwriting baseline."""
    x, labels, _ = load_handwriting(cfg_dataset=cfg.handwriting)
    inception = InceptionTimeClassifier()
    for train_test_ratio in cfg.handwriting.train_test_ratios:
        np.random.seed(42)
        train_size = int(train_test_ratio * x.shape[0])
        print(x.shape, labels.shape)
        inception.fit(x[:train_size], labels[:train_size])
        y_pred = inception.predict(x[train_size:])
        accuracy = accuracy_score(labels[train_size:], y_pred)
        print(f"train_test_ratio: {train_test_ratio}, accuracy: {accuracy}")


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES="" poetry run python mmda/handwriting_baseline.py
