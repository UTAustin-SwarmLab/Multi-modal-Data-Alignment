# Multi-modal Data Alignment (MMDA)
The is the repo of CSA: Data-efficient Mapping of Unimodal Features to Multimodal Features and Any2Any: Incomplete Multimodal Retrieval with Conformal Prediction.

Link to papers: [CSA](https://openreview.net/forum?id=6Mg7pjG7Sw&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)) and
[Any2Any](https://arxiv.org/abs/2411.10513)

Link to blogs: [CSA](https://utaustin-swarmlab.github.io/2025/01/24/CSA.html)

## TL;DR
Canonical Similarity Analysis (CSA) matches CLIP in multimodal tasks with far less data, mapping unimodal features into a multimodal space without extensive GPU training.

Any2Any effectively retrieves from incomplete multimodal data, achieving 35% Recall@5 on the KITTI dataset, matching baseline models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Adding more datasets](#adding-more-datasets)
- [Reproducing the results](#reproducing-the-results)
- [Core code of CSA](#core-code-of-csa)
- [Core code of Any2Any](#core-code-of-any2any)
- [Disclaimer](#disclaimer)
- [Citation](#citation)
## Prerequisites
To run the code, you need to install the packages using poetry:
```bash
poetry lock && poetry install
```
Or, you can install the packages using pip (check the pyproject.toml for the dependencies).

## Adding more datasets
To add more dataset, just add the dataset configs in `configs/main.yaml` and fill in the code wherever with the "TODO: add more dataset" comment.
To reproduce the results, download the datasets and change their corresponding paths in the `configs/main.yaml` file.
Then, run the following command according to what experiment you want to run (see the description of the experiments in the head of each .py file):
```bash
poetry run python mmda/<experiment>.py
```

## CSA: System plot and major results
![CSA_system_graph](https://github.com/UTAustin-SwarmLab/Multi-modal-Data-Alignment/blob/main/assets/SystemGraph.png)

![COSMOS_results](https://github.com/UTAustin-SwarmLab/Multi-modal-Data-Alignment/blob/main/assets/COSMOS.png)

![ImageNet_results](https://github.com/UTAustin-SwarmLab/Multi-modal-Data-Alignment/blob/main/assets/imagenet.png)

## Reproducing the results

### Core code of CSA
To see the core code of CSA, see "class NormalizedCCA" in mmda.utils.cca_class.py and "fn weighted_corr_sim" in mmda.utils.sim_utils.py.
We have a notebook to show the usage of the core code of CSA in `mmda/csa_example.ipynb`.


### Core code of Any2Any
To see the core code of Any2Any, see in `mmda/any2any_conformal_retrieval.py` and `mmda/exps/any2any_retrieval.py`. We put the configs of KITTI (Image-to-LiDAR retrieval) in `mmda/utils/liploc_model.py`, which is modified from the [Lip-loc](https://github.com/Shubodh/lidar-image-pretrain-VPR) repository.



## Disclaimer
Some of the code are modified from the [ASIF](https://github.com/noranta4/ASIF) and [Lip-loc](https://github.com/Shubodh/lidar-image-pretrain-VPR) repositories and leverage datasets and models from Hugginface.
We tried our best to cite the source of all code, models, and datasets used. If we missed any, please let us know.

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@inproceedings{li2025csa,
    title={{CSA}: Data-efficient Mapping of Unimodal Features to Multimodal Features},
    author={Li, Po-han and Chinchali, Sandeep P and Topcu, Ufuk},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=6Mg7pjG7Sw}
}

@misc{li2024any2anyincompletemultimodalretrieval,
      title={Any2Any: Incomplete Multimodal Retrieval with Conformal Prediction}, 
      author={Po-han Li and Yunhao Yang and Mohammad Omama and Sandeep Chinchali and Ufuk Topcu},
      year={2024},
      eprint={2411.10513},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10513}, 
}
```
