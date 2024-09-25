# Multi-modal Data Alignment (MMDA)

## Reproducing the results
### Adding more datasets
To add more dataset, just add the dataset configs in `configs/main.yaml` and fill in the code with the "TODO: add more dataset" comment.
To reproduce the results, download the datasets and change their corresponding paths in the `configs/main.yaml` file.
Then, run the following command according to what experiment you want to run (see the description of the experiments in the head of each .py file):
```bash
poetry run python mmda/<experiment>.py
```

### Core code of CSA
To see the core code of CSA, see "class NormalizedCCA" in mmda.utils.cca_class.py and "fn weighted_corr_sim" in mmda.utils.sim_utils.py.



## Disclaimer
Some of the code are modified from the [ASIF](https://github.com/noranta4/ASIF) and [Lip-loc](https://github.com/Shubodh/lidar-image-pretrain-VPR) repositories and leverage datasets and models from Hugginface.
We tried our best to cite the source of all code, models, and datasets used. If we missed any, please let us know.
