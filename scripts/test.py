import json
import os

from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):  # noqa: D103
    full_test_json = json.load(open(os.path.join(cfg.tiil.paths.dataset_path + "full_test.json")))
    print(full_test_json.keys())
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
