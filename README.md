# OTER: Optimal Transport and Entity-Anchored Evidence Routing for Few-Shot Multimodal Relation Extraction

Official implementation of OTER (submitted to **IEEE ICASSP 2027**), a few-shot multimodal relation extraction framework that combines **Optimal Transport Alignment (OTA)** with **Entity-Anchored Text Evidence Routing (EATER)**. OTER aligns global-image and entity-region tokens and adaptively routes evidence from the original text, grounding phrases, and image captions.

This repository contains the training code and configuration used for the experiments in the paper. The implementation supports two datasets: the MNRE benchmark and FewREL-small.

## Overview

OTER formulates few-shot multimodal relation extraction as an episodic classification task. Given an entity pair, the model combines the original text, grounding phrases, image captions, the global image, and entity-level images. OTA establishes fine-grained correspondence between global and entity-level visual features, while EATER treats the entity pair as a semantic anchor and dynamically fuses the available textual evidence. The resulting representation is used for prototype-based relation classification.

## Installation

Clone the repository and create the recommended Conda environment:

```bash
git clone https://github.com/Echo2572/OTER.git
cd OTER

conda create -n oter python=3.10
conda activate oter
pip install -r requirements.txt
```

When training with CUDA, install a PyTorch and torchvision build compatible with the target CUDA version before running the scripts.

## Data preparation

Download and place the two datasets in the repository root. After extraction, the directory layout should follow the structure below.

```text
OTER/
├── configs/
├── encoder/
├── scripts/
│   ├── train_mnre_full.sh
│   └── train_fewrel_full.sh
├── train_full.py
├── requirements.txt
├── MNRE/
│   ├── caption/
│   ├── img_org/
│   ├── img_vg/
│   ├── txt/
│   └── ours_rel2id.json
└── FewRel_small/
    ├── image/
    ├── output/
    │   ├── crops_mixed_fallback/
    │   ├── crops_mixed_fallback_manifest.jsonl
    │   └── crops_text_boxes.jsonl
    ├── fewrel_caption.txt
    ├── rel2id.json
    ├── train_small.json
    ├── val_small.json
    ├── train_rel2scope.json
    ├── val_rel2scope.json
    └── train_wiki_official.json
```

### MNRE

The MNRE benchmark was introduced by:

> Zheng, C., Wu, Z., Feng, J., et al. “MNRE: A Challenge Multimodal Dataset for Neural Relation Extraction with Visual Evidence in Social Media Posts.” *2021 IEEE International Conference on Multimedia and Expo (ICME)*, IEEE, 2021, pp. 1–6.

The caption information used in this project follows:

> Zhang, Z., Zhang, W., Li, Y., et al. “Caption-Aware Multimodal Relation Extraction with Mutual Information Maximization.” *Proceedings of the 32nd ACM International Conference on Multimedia*, 2024, pp. 1148–1157.

The entity-crop visual evidence follows:

> Chen, X., Zhang, N., Li, L., et al. “Good Visual Guidance Make a Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction.” *Findings of the Association for Computational Linguistics: NAACL 2022*, 2022, pp. 1607–1618.

Place the processed MNRE files under `MNRE/` as shown above.

### FewREL-small

The original FewRel benchmark was introduced by:

> Han, X., Zhu, H., Yu, P., et al. “FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation.” *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2018, pp. 4803–4809.

This repository uses **FewREL-small**, the multimodal subset of FewRel proposed in:

> Gong, J. and Eldardiry, H. “Few-Shot Relation Extraction with Hybrid Visual Evidence.” *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, 2024, pp. 7232–7247.

Because the original FewRel data do not provide the required auxiliary visual modalities, the FewREL-small preparation used in this project supplements the multimodal subset with image captions and entity crops. Captions are generated with vision-language models, and the head/tail entities are grounded in the original images to obtain entity crops. The generated outputs are manually verified, and the preparation does not use relation labels, avoiding label leakage. The supplementary files are available here:

[Download the FewRel-small supplementary data](https://1drv.ms/f/c/02848db892a0e8dd/IgCWCeWM4GPsSZUhghpJC2mgAWqhuqlpPWbq_yOTeISf0yc?e=EVNQqs)

After downloading, place the files under `FewRel_small/` and preserve the names and subdirectories shown in the tree above. The `output/crops_mixed_fallback/` directory contains the generated entity crops; its accompanying manifest and text-box files must remain in `output/`.

## Training

From the repository root, run the corresponding script.

### MNRE

```bash
bash scripts/train_mnre_full.sh
```

### FewREL-small

```bash
bash scripts/train_fewrel_full.sh
```

Both scripts use GPU `0`, a 5-way episode, and 1-shot training by default. For FewREL-small, the support set size can also be changed to the 5-shot setting. These settings can be overridden without editing the scripts:

```bash
GPU=1 N=10 bash scripts/train_mnre_full.sh
GPU=0 N=5 K=5 bash scripts/train_fewrel_full.sh
```

Checkpoints are written to `ckpt/`, and evaluation metrics are written to `logs/`. These output directories are created or populated by the training workflow as needed.

We provided the checkpoint and logs for the MNRE and FewRel-small datasets under the 5W1S data setup, the link is:

[Provided Checkpoint and Logs](https://1drv.ms/f/c/02848db892a0e8dd/IgD3MOOiPgCyT42oh0lPd4lhAQD7xtYfPmT8Yp9VxQARQn8?e=7PNfeU)

## License and attribution

Copyright © 2027 Echo2572. Unless a separate license file is added to this repository, the source code is provided for academic research and evaluation purposes only. The datasets, pretrained components, and supplementary files remain subject to their original authors’ and providers’ licenses and terms of use. Please consult the corresponding dataset papers and download pages before redistributing any data.

## Acknowledgements

We thank the authors of MNRE, FewRel, and the cited multimodal relation extraction studies for making the underlying benchmarks and research resources available to the community.
